"""Validation and inference utilities

Key fixes for generation with inputs_embeds:
1. generate() with inputs_embeds returns ONLY generated tokens (not input+generated)
2. Use greedy decoding (do_sample=False) when both modalities disabled for stability
3. Decode outputs directly as they are already the generated portion
4. Only set max_new_tokens (not max_length) to avoid HuggingFace warnings
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from deepencoder.deepencoder_infer import multiview_tokens_from_sample_token, batch_multiview_tokens_from_sample_tokens
from configs.default_config import (
    DEFAULT_VIEW_ORDER,
    NUM_VIEWS,
    TOKENS_PER_VIEW,
    PROJECTOR_DIM,
)
from ..data.utils import validate_image_paths
from ..utils import calculate_metrics_by_type, calculate_sample_level_metrics
from ..utils.sequence_builder import build_training_sequence, build_inference_sequence, ModalityPosition


@torch.no_grad()
def run_validation(dl, device, tok, base, vat_lidar, vat_vision, vision_adapter, runtime, nusc, config, use_amp=False, amp_dtype=torch.float32):
    """
    Run validation on validation dataloader.
    
    Args:
        dl: Validation DataLoader
        device: Device to run on
        tok: Tokenizer
        base: Base LLM model
        vat_lidar: LiDAR VAT model
        vat_vision: Vision VAT model (optional)
        vision_adapter: Vision adapter (optional)
        runtime: DeepEncoder runtime (optional)
        nusc: NuScenes instance (optional)
        config: Training configuration
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (torch.float16 or torch.bfloat16)
        
    Returns:
        Average validation loss
    """
    # Log validation toggles
    use_vision_validation = config.get("validation_use_vision", True)
    use_lidar_validation = config.get("validation_use_lidar", True)
    print(f"[validation] Component toggles: Vision={use_vision_validation}, LiDAR={use_lidar_validation}")
    
    # Unwrap DDP if needed
    def unwrap(model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    base_model = unwrap(base)
    vat_lidar_model = unwrap(vat_lidar)

    base_model.eval()
    vat_lidar_model.eval()

    if config["use_vision"]:
        vat_vision_model = unwrap(vat_vision)
        vision_adapter_model = unwrap(vision_adapter)
        # Note: runtime is not DDP-wrapped, use directly

        vat_vision_model.eval()
        vision_adapter_model.eval()
        runtime.eval()
    else:
        vat_vision_model = vision_adapter_model = None

    total_loss = 0.0
    count = 0

    for batch in dl:
        # Use non_blocking for async CPU→GPU transfers
        bev = batch["bev"].to(device, non_blocking=True)
        p_ids = batch["prompt_ids"].to(device, non_blocking=True)
        a_ids = batch["answer_ids"].to(device, non_blocking=True)
        sample_tokens = batch["sample_tokens"]

        # Check validation toggles
        use_vision_in_validation = config.get("validation_use_vision", True)
        use_lidar_in_validation = config.get("validation_use_lidar", True)

        # Vision pipeline - use batched processing for efficiency
        vision_kv = None
        if config["use_vision"] and use_vision_in_validation:
            try:
                # Batched vision encoding (much faster than sequential)
                batch_view_tokens = batch_multiview_tokens_from_sample_tokens(
                    sample_tokens, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                )
                
                # Move all tensors to device
                batch_view_tokens_device = [
                    [t.to(device) for t in view_tokens]
                    for view_tokens in batch_view_tokens
                ]
                
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    vision_kv = vision_adapter_model.forward_batch(batch_view_tokens_device)
                    
            except Exception as e:
                # Fallback to sequential processing
                print(f"[validation] Batched vision encoding failed, using sequential: {e}")
                vision_kvs = []
                for tok_str in sample_tokens:
                    mv = multiview_tokens_from_sample_token(
                        tok_str, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                    )
                    # Safety check
                    if not mv.get("tokens") or len(mv["tokens"]) != NUM_VIEWS:
                        dummy_shape = (TOKENS_PER_VIEW, PROJECTOR_DIM)
                        mv["tokens"] = [torch.zeros(dummy_shape, device=device) for _ in range(NUM_VIEWS)]

                    vt_list = [t.to(device) for t in mv["tokens"]]

                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        kv = vision_adapter_model(vt_list)  # [TOTAL_VISION_TOKENS, PROJECTOR_DIM]
                        kv = kv.unsqueeze(0)  # Add batch dimension
                    vision_kvs.append(kv)
                vision_kv = torch.cat(vision_kvs, dim=0)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            E = base_model.get_input_embeddings()

            def emb_token(txt):
                ids = tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                return E(ids)

            # Process LiDAR (if enabled)
            # Note: VAT models now include learned output_scale internally
            prefix_lidar = None
            if use_lidar_in_validation:
                prefix_lidar = vat_lidar_model(bev)  # Learned scale applied inside VAT
            
            # Process vision (if enabled and available)
            prefix_vision = None
            if vision_kv is not None and use_vision_in_validation:
                prefix_vision = vat_vision_model(vision_kv)  # Learned scale applied inside VAT

            tok_emb = E(p_ids)
            ans_emb = E(a_ids)
            batch_size = p_ids.size(0)

            # Build input sequence with explicit position markers using helper function
            # Order: [vision_start, vision_tokens, vision_end, lidar_start, lidar_tokens, lidar_end, text_prompt, answer_tokens]
            inp, sequence_metadata = build_training_sequence(
                E=E,
                device=device,
                dtype=amp_dtype,
                batch_size=batch_size,
                tok_emb=tok_emb,
                ans_emb=ans_emb,
                prefix_vision=prefix_vision,
                prefix_lidar=prefix_lidar,
                get_special_token_emb=emb_token,
            )

            B = inp.size(0)
            total_len = inp.size(1)
            labels = torch.full((B, total_len), -100, dtype=torch.long, device=device)
            labels[:, -a_ids.size(1) :] = a_ids
            attn = torch.ones((B, total_len), dtype=torch.long, device=device)

            out = base_model(inputs_embeds=inp, attention_mask=attn, labels=labels)
            total_loss += float(out.loss.item())
            count += 1

    # Restore training mode
    base_model.train()
    vat_lidar_model.train()
    if config["use_vision"]:
        vat_vision_model.train()
        vision_adapter_model.train()
        # Restore runtime to training mode
        runtime.train()

    return total_loss / max(1, count)


@torch.no_grad()
def save_val_inference_samples(
    ds_val, tok, base, vat_lidar, vat_vision, vision_adapter, runtime, nusc, config, out_dir, epoch, n=10
):
    """
    Generate and save validation inference samples.
    
    Args:
        ds_val: Validation dataset
        tok: Tokenizer
        base: Base LLM model
        vat_lidar: LiDAR VAT model
        vat_vision: Vision VAT model (optional)
        vision_adapter: Vision adapter (optional)
        runtime: DeepEncoder runtime (optional)
        nusc: NuScenes instance (optional)
        config: Training configuration
        out_dir: Output directory
        epoch: Current epoch
        n: Number of samples to save
    """
    if not config["use_vision"]:
        print("[skip] inference samples require vision pipeline")
        return

    if vat_vision is None or vision_adapter is None or runtime is None or nusc is None:
        print("[skip] vision pipeline not initialized")
        return

    assert n % 2 == 0, "n must be even"

    # Unwrap DDP
    def unwrap(model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    base_model = unwrap(base)
    vat_lidar_model = unwrap(vat_lidar)
    vat_vision_model = unwrap(vat_vision)
    vision_adapter_model = unwrap(vision_adapter)

    # Set to eval mode
    was_training_base = base_model.training
    was_training_lidar = vat_lidar_model.training
    was_training_vision = vat_vision_model.training
    was_training_adapter = vision_adapter_model.training

    base_model.eval()
    vat_lidar_model.eval()
    vat_vision_model.eval()
    vision_adapter_model.eval()

    device = next(base_model.parameters()).device

    captions = []
    grounding = []

    # Classify validation rows
    for r in ds_val.dataset.rows:
        src = r.get("dataset_source", "")
        if not src:
            src = "grounding"
            if "caption" in r.get("question", "").lower():
                src = "caption"
        if "caption" in src.lower():
            captions.append(r)
        else:
            grounding.append(r)

    random.shuffle(captions)
    random.shuffle(grounding)

    chosen = captions[: n // 2] + grounding[: n // 2]
    results = []

    for row in chosen:
        try:
            sample_token = row["sample_token"]
            question = row.get("question", "")
            ground_truth = row.get(config["target_field"], "")

            # Load BEV feature
            bev_path = ds_val.dataset.token2path.get(sample_token)
            if not bev_path:
                continue

            import numpy as np

            bev = np.load(bev_path)
            bev = torch.from_numpy(bev).float().unsqueeze(0).to(device)

            # Generate prediction
            msgs = [
                {
                    "role": "system",
                    "content": "You are a driving assistant. Use LiDAR and camera context provided via prefix tokens.",
                },
                {"role": "user", "content": question},
            ]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            results.append(
                {
                    "sample_token": sample_token,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": "[generation not implemented - add your inference code here]",
                }
            )
        except Exception as e:
            print(f"[warn] failed to process sample {row.get('sample_token', 'unknown')}: {e}")
            continue

    fname = out_dir / f"inference_epoch{epoch}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] validation inference → {fname}")

    # Restore training mode
    if was_training_base:
        base_model.train()
    if was_training_lidar:
        vat_lidar_model.train()
    if was_training_vision:
        vat_vision_model.train()


@torch.no_grad()
def run_inference_sampling(
    base, vat_lidar, vat_vision, vision_adapter, runtime, nusc,
    tok, config, out_dir, epoch, device, token2path, best_step,
    use_amp=False, amp_dtype=torch.float32
):
    """
    Generate predictions on validation samples with evaluation metrics.
    
    Samples n/2 from caption validation and n/2 from grounding validation,
    generates predictions using the current best model, and calculates metrics.
    
    Args:
        base: Base LLM model
        vat_lidar: LiDAR VAT model
        vat_vision: Vision VAT model (optional)
        vision_adapter: Vision adapter (optional)
        runtime: DeepEncoder runtime (optional)
        nusc: NuScenes instance (optional)
        tok: Tokenizer
        config: Training configuration
        out_dir: Output directory
        epoch: Current epoch number
        device: Device to run on
        token2path: Mapping from sample_token to BEV feature path
        best_step: Best step number so far
    """
    print(f"\n[inference_sampling] Generating predictions at epoch {epoch}...")
    
    # Log inference component toggles
    use_vision_toggle = config.get("inference_use_vision", True)
    use_lidar_toggle = config.get("inference_use_lidar", True)
    use_system_toggle = config.get("inference_use_system", True)
    
    print(f"[inference_sampling] Component toggles: Vision={use_vision_toggle}, LiDAR={use_lidar_toggle}, System={use_system_toggle}")
    
    # Inform about decoding strategy
    if not use_vision_toggle and not use_lidar_toggle:
        print(f"[inference_sampling] Using GREEDY decoding (both modalities disabled, more stable for early training)")
    else:
        print(f"[inference_sampling] Using SAMPLING decoding (temp={config.get('inference_temperature', 0.7)}, top_p={config.get('inference_top_p', 0.9)})")

    
    # Unwrap DDP if needed
    def unwrap(model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    base_model = unwrap(base)
    vat_lidar_model = unwrap(vat_lidar)
    
    # CRITICAL: Determine model dtype ONCE at the start to avoid double casting
    # The model weights dtype is the ground truth - use this consistently throughout
    model_dtype = next(base_model.parameters()).dtype
    print(f"[inference_sampling] Model dtype: {model_dtype}, AMP dtype: {amp_dtype}, AMP enabled: {use_amp}")
    
    # Set to eval mode
    was_training_base = base_model.training
    was_training_lidar = vat_lidar_model.training
    
    base_model.eval()
    vat_lidar_model.eval()
    
    # Disable gradient checkpointing for generation (important!)
    if hasattr(base_model, 'gradient_checkpointing_disable'):
        base_model.gradient_checkpointing_disable()
        gradient_checkpointing_was_enabled = True
    else:
        gradient_checkpointing_was_enabled = False
    
    # Enable cache for generation (important for performance and correctness)
    original_use_cache = base_model.config.use_cache
    base_model.config.use_cache = True
    
    if config["use_vision"]:
        vat_vision_model = unwrap(vat_vision)
        vision_adapter_model = unwrap(vision_adapter)
        # Note: runtime is not DDP-wrapped, use directly
        
        was_training_vision = vat_vision_model.training
        was_training_adapter = vision_adapter_model.training
        
        vat_vision_model.eval()
        vision_adapter_model.eval()
        runtime.eval()
    else:
        vat_vision_model = vision_adapter_model = None
    
    # =========================================================================
    # DATASET MODE: Load JSONs based on config setting
    # =========================================================================
    dataset_mode = config.get("dataset_mode", "both")
    print(f"[inference_sampling] Dataset mode: {dataset_mode}")
    
    caption_data = []
    grounding_data = []
    
    # Load caption data if mode includes caption
    if dataset_mode in ("caption", "both"):
        caption_json_path = config.get("inference_caption_json")
        fallback_caption = False
        if not caption_json_path:
            caption_json_path = config.get("caption_json")
            if caption_json_path:
                fallback_caption = True
        if caption_json_path:
            try:
                with open(caption_json_path, "r", encoding="utf-8") as f:
                    caption_data = json.load(f)
                split_tag = "training" if fallback_caption else "inference"
                print(
                    f"[inference_sampling] Loaded {len(caption_data)} caption samples from {split_tag} split → {caption_json_path}"
                )
            except Exception as e:
                print(f"[inference_sampling] Warning: Could not load caption JSON: {e}")
                caption_data = []
        else:
            print(f"[inference_sampling] Warning: No caption JSON configured (inference or training)")
    
    # Load grounding data ONLY if mode includes grounding
    if dataset_mode in ("grounding", "both"):
        grounding_json_path = config.get("inference_grounding_json")
        fallback_grounding = False
        if not grounding_json_path:
            grounding_json_path = config.get("grounding_json")
            if grounding_json_path:
                fallback_grounding = True
        if grounding_json_path:
            try:
                with open(grounding_json_path, "r", encoding="utf-8") as f:
                    grounding_data = json.load(f)
                split_tag = "training" if fallback_grounding else "inference"
                print(
                    f"[inference_sampling] Loaded {len(grounding_data)} grounding samples from {split_tag} split → {grounding_json_path}"
                )
            except Exception as e:
                print(f"[inference_sampling] Warning: Could not load grounding JSON: {e}")
                grounding_data = []
        else:
            print(f"[inference_sampling] Warning: No grounding JSON configured (inference or training)")
    
    # ===== INFERENCE DATA VALIDATION =====
    # Validate camera image paths for inference data (when using vision)
    # This ensures test data has the same validation as training data
    if config.get("use_vision", False) and nusc is not None and config.get("validate_image_paths", True):
        all_inference_tokens = list(set(
            [s.get("sample_token") for s in caption_data if s.get("sample_token")] +
            [s.get("sample_token") for s in grounding_data if s.get("sample_token")]
        ))
        if all_inference_tokens:
            print(f"\n[inference_sampling] Validating camera image paths for {len(all_inference_tokens)} unique tokens...")
            image_validation = validate_image_paths(
                nusc=nusc,
                sample_tokens=all_inference_tokens,
                view_order=DEFAULT_VIEW_ORDER,
                num_workers=config.get("bev_validation_workers", 16),
                # Check all inference samples (they're typically fewer than training)
            )
            if image_validation.get('tokens_with_missing', 0) > 0:
                print(f"[inference_sampling] ⚠️  {image_validation['tokens_with_missing']} samples have missing camera views")
                print(f"[inference_sampling] ⚠️  Missing views will be filled with zeros, which may affect evaluation quality.")
    
    # =========================================================================
    # SAMPLING STRATEGY BASED ON DATASET MODE
    # =========================================================================
    total_n = config["inference_samples_n"]
    
    # Filter for samples that have BEV features
    caption_available = [s for s in caption_data if s.get("sample_token") in token2path]
    
    # For grounding: Use BOTH types for comprehensive evaluation
    # det_area: Descriptive questions (no coords in Q) → Text quality + Bbox accuracy
    # det_object: Coordinate questions (coords in Q) → Text quality only (skip bbox to avoid copying)
    grounding_det_area = [
        s for s in grounding_data 
        if s.get("sample_token") in token2path 
        and s.get("template_type") == "det_area"
    ]
    
    grounding_det_object = [
        s for s in grounding_data 
        if s.get("sample_token") in token2path 
        and s.get("template_type") == "det_object"
    ]
    
    # Log filtering statistics
    total_grounding_with_bev = len([s for s in grounding_data if s.get("sample_token") in token2path])
    
    # Calculate sample distribution based on dataset_mode
    if dataset_mode == "caption":
        # Caption only: all samples from caption dataset
        n_caption = total_n
        n_det_area = 0
        n_det_object = 0
        
        print(f"[inference_sampling] Sampling strategy (caption only, total={total_n}):")
        print(f"  Caption: {n_caption} samples (100%)")
        
        # Validate sufficient samples
        assert len(caption_available) >= n_caption, (
            f"Insufficient caption samples: need {n_caption}, have {len(caption_available)}. "
            f"Reduce inference_samples_n or add more caption data."
        )
        
        caption_samples = random.sample(caption_available, n_caption)
        det_area_samples = []
        det_object_samples = []
        
    elif dataset_mode == "grounding":
        # Grounding only: split evenly between det_area and det_object
        assert total_n % 2 == 0, (
            f"inference_samples_n must be divisible by 2 for grounding mode. "
            f"Got {total_n}. Recommended values: 2, 4, 6, 8, 10, etc."
        )
        n_caption = 0
        n_det_area = total_n // 2
        n_det_object = total_n // 2
        
        print(f"[inference_sampling] Sampling strategy (grounding only, total={total_n}):")
        print(f"  Grounding det_area: {n_det_area} samples (50%)")
        print(f"  Grounding det_object: {n_det_object} samples (50%)")
        
        # Validate sufficient samples
        assert len(grounding_det_area) >= n_det_area, (
            f"Insufficient det_area samples: need {n_det_area}, have {len(grounding_det_area)}. "
            f"Reduce inference_samples_n or add more det_area data."
        )
        assert len(grounding_det_object) >= n_det_object, (
            f"Insufficient det_object samples: need {n_det_object}, have {len(grounding_det_object)}. "
            f"Reduce inference_samples_n or add more det_object data."
        )
        
        caption_samples = []
        det_area_samples = random.sample(grounding_det_area, n_det_area)
        det_object_samples = random.sample(grounding_det_object, n_det_object)
        
    else:  # "both" mode
        # Original behavior: 50% caption, 25% det_area, 25% det_object
        assert total_n % 4 == 0, (
            f"inference_samples_n must be divisible by 4 for 'both' mode. "
            f"Got {total_n}. Recommended values: 4, 8, 12, 16, 20, 24, 28, 32, etc."
        )
        n_caption = total_n // 2
        n_det_area = total_n // 4
        n_det_object = total_n // 4
        
        print(f"[inference_sampling] Sampling strategy (both, total={total_n}):")
        print(f"  Caption: {n_caption} samples (50%)")
        print(f"  Grounding det_area: {n_det_area} samples (25%)")
        print(f"  Grounding det_object: {n_det_object} samples (25%)")
        
        # Validate sufficient samples
        assert len(caption_available) >= n_caption, (
            f"Insufficient caption samples: need {n_caption}, have {len(caption_available)}. "
            f"Reduce inference_samples_n or add more caption data."
        )
        assert len(grounding_det_area) >= n_det_area, (
            f"Insufficient det_area samples: need {n_det_area}, have {len(grounding_det_area)}. "
            f"Reduce inference_samples_n or add more det_area data."
        )
        assert len(grounding_det_object) >= n_det_object, (
            f"Insufficient det_object samples: need {n_det_object}, have {len(grounding_det_object)}. "
            f"Reduce inference_samples_n or add more det_object data."
        )
        
        caption_samples = random.sample(caption_available, n_caption)
        det_area_samples = random.sample(grounding_det_area, n_det_area)
        det_object_samples = random.sample(grounding_det_object, n_det_object)
    
    # Log available samples
    print(f"\n[inference_sampling] Available samples:")
    print(f"  Caption: {len(caption_available)} available")
    print(f"  Grounding total: {total_grounding_with_bev} available")
    print(f"    det_area: {len(grounding_det_area)} available → text quality + bbox accuracy")
    print(f"    det_object: {len(grounding_det_object)} available → text quality only")
    
    print(f"\n[inference_sampling] ✓ Sampled exactly:")
    print(f"  Caption: {len(caption_samples)} samples")
    print(f"  det_area: {len(det_area_samples)} samples")
    print(f"  det_object: {len(det_object_samples)} samples")
    print(f"  Total: {len(caption_samples) + len(det_area_samples) + len(det_object_samples)} samples")
    
    # Verify equal distribution
    assert len(caption_samples) == n_caption
    assert len(det_area_samples) == n_det_area
    assert len(det_object_samples) == n_det_object
    assert len(caption_samples) + len(det_area_samples) + len(det_object_samples) == total_n
    
    all_samples = [
        {**s, "dataset_type": "caption"} for s in caption_samples
    ] + [
        {**s, "dataset_type": "grounding_det_area"} for s in det_area_samples
    ] + [
        {**s, "dataset_type": "grounding_det_object"} for s in det_object_samples
    ]
    
    # ===== BATCHED ENCODING PHASE =====
    # Pre-encode all BEV and vision features in batches for efficiency
    # This avoids redundant per-sample encoding during generation
    
    inference_batch_size = max(1, int(config.get("inference_batch_size", 8)))  # Encode in batches
    print(f"\n[inference_sampling] Pre-encoding {len(all_samples)} samples (batch_size={inference_batch_size})...")
    
    # Filter valid samples and prepare for batched encoding
    valid_samples = []
    for sample in all_samples:
        sample_token = sample["sample_token"]
        bev_path = token2path.get(sample_token)
        if bev_path:
            sample["_bev_path"] = bev_path
            valid_samples.append(sample)
        else:
            print(f"[inference_sampling] Warning: No BEV feature for {sample_token}")
    
    # Pre-compute all encodings in batches
    encoded_samples = []
    
    for batch_start in range(0, len(valid_samples), inference_batch_size):
        batch_end = min(batch_start + inference_batch_size, len(valid_samples))
        batch_samples = valid_samples[batch_start:batch_end]
        batch_tokens = [s["sample_token"] for s in batch_samples]
        
        # Batch load BEV features
        batch_bevs = []
        for s in batch_samples:
            bev = np.load(s["_bev_path"])
            batch_bevs.append(torch.from_numpy(bev).float())
        batch_bev = torch.stack(batch_bevs, dim=0).to(device, non_blocking=True)  # [B, C, H, W]
        
        # Batched LiDAR encoding (learned scale applied inside VAT)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            batch_prefix_lidar = vat_lidar_model(batch_bev)  # [B, n_queries, d_model]
        
        # Batched vision encoding
        batch_prefix_vision = None
        if config["use_vision"] and nusc is not None:
            batch_view_tokens = batch_multiview_tokens_from_sample_tokens(
                batch_tokens, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
            )
            try:
                batch_view_tokens_device = [
                    [t.to(device) for t in view_tokens]
                    for view_tokens in batch_view_tokens
                ]
                
                # Learned scale applied inside VAT
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    batch_kv = vision_adapter_model.forward_batch(batch_view_tokens_device)  # [B, 1536, 2048]
                    batch_prefix_vision = vat_vision_model(batch_kv)  # [B, n_queries, d_model]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[inference_sampling] Batched vision encoding OOM, retrying sequentially (batch_size=1)...")
                    torch.cuda.empty_cache()
                    sequential_outputs = []
                    for single_views in batch_view_tokens:
                        single_views_device = [t.to(device) for t in single_views]
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            single_kv = vision_adapter_model.forward_batch([single_views_device])
                            single_prefix = vat_vision_model(single_kv)
                        sequential_outputs.append(single_prefix)
                    batch_prefix_vision = torch.cat(sequential_outputs, dim=0)
                else:
                    print(f"[inference_sampling] Batched vision encoding failed: {e}")
                    batch_prefix_vision = None
            except Exception as e:
                print(f"[inference_sampling] Batched vision encoding failed: {e}")
                batch_prefix_vision = None
        
        # Store encoded features for each sample
        for i, sample in enumerate(batch_samples):
            sample["_prefix_lidar"] = batch_prefix_lidar[i:i+1]  # [1, n_queries, d_model]
            if batch_prefix_vision is not None:
                sample["_prefix_vision"] = batch_prefix_vision[i:i+1]  # [1, n_queries, d_model]
            else:
                sample["_prefix_vision"] = None
            encoded_samples.append(sample)
        
        if (batch_start // inference_batch_size) % 2 == 0:
            print(f"[inference_sampling] Encoded {batch_end}/{len(valid_samples)} samples...")
    
    print(f"[inference_sampling] ✓ Pre-encoding complete: {len(encoded_samples)} samples ready")
    
    # ===== GENERATION PHASE =====
    # Generate predictions using pre-encoded features
    results = []
    
    for sample in encoded_samples:
        try:
            sample_token = sample["sample_token"]
            question = sample.get("question", "").strip()
            ground_truth = sample.get(config["target_field"], "").strip()
            dataset_type = sample["dataset_type"]
            
            # Use pre-encoded features
            prefix_lidar = sample["_prefix_lidar"]  # [1, n_queries, d_model]
            prefix_vision = sample.get("_prefix_vision")  # [1, n_queries, d_model] or None
            
            # Format prompt with configurable system prompt and toggles
            # CRITICAL: Match the exact order used during training!
            # BUT allow toggling components for debugging/ablation studies
            
            # Check inference toggles (default to True for backward compatibility)
            use_vision_in_inference = config.get("inference_use_vision", True)
            use_lidar_in_inference = config.get("inference_use_lidar", True)
            use_system_in_inference = config.get("inference_use_system", True)
            
            # Build prompt based on system toggle
            if use_system_in_inference:
                system_prompt = config.get(
                    "system_prompt", 
                    "You are an expert autonomous driving assistant. Analyze the 3D LiDAR point cloud and camera images to understand the driving scene. Provide accurate, concise descriptions of objects, their locations, distances, and spatial relationships. Use directional terms like 'ahead', 'left', 'right', 'behind' and specify distances in meters when describing object locations."
                )
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
            else:
                # Skip system prompt, only user question
                msgs = [
                    {"role": "user", "content": question},
                ]
            
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            
            # Build inputs_embeds using the SAME order as training
            # Training order: VISION → LIDAR → SYSTEM+QUESTION
            # 
            # CRITICAL: Use model_dtype directly instead of relying on autocast output dtype.
            # This avoids double-casting: autocast may produce fp16/bf16, then we'd cast again.
            # Instead, we explicitly cast to model_dtype once after all operations.
            with torch.autocast(device_type=device.type, dtype=model_dtype, enabled=use_amp):
                E = base_model.get_input_embeddings()
                
                def emb_token(txt):
                    ids = tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    return E(ids)
                
                # Tokenize the prompt (system + user + generation prompt)
                prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                prompt_embeds = E(prompt_ids)  # [1, L, d_model]
            
                # Build the full input with explicit position markers
                # Order: [vision_start, vision_tokens, vision_end, lidar_start, lidar_tokens, lidar_end, text_prompt]
                # Use toggles to conditionally include vision/lidar
                effective_vision = prefix_vision if (prefix_vision is not None and use_vision_in_inference) else None
                effective_lidar = prefix_lidar if use_lidar_in_inference else None
                
                # Safety check: ensure we have at least some content
                if effective_vision is None and effective_lidar is None and prompt_embeds is None:
                    print(f"[inference_sampling] Error: No components enabled for {sample_token}, skipping...")
                    print(f"[inference_sampling]   Check: inference_use_vision, inference_use_lidar, or ensure prompt is not empty")
                    continue
                
                inputs_embeds, sequence_metadata = build_inference_sequence(
                    device=device,
                    dtype=model_dtype,  # Use model_dtype directly, not amp_dtype
                    batch_size=1,
                    tok_emb=prompt_embeds,
                    prefix_vision=effective_vision,
                    prefix_lidar=effective_lidar,
                    get_special_token_emb=emb_token,
                )
            
            # inputs_embeds is now already in model_dtype - no redundant cast needed
            # Verify dtype matches (debug assertion)
            assert inputs_embeds.dtype == model_dtype, \
                f"dtype mismatch: inputs_embeds={inputs_embeds.dtype}, model={model_dtype}"
            
            attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)
            
            # Debug logging
            if config.get("debug_shapes", False):
                print(f"[inference_sampling] inputs_embeds shape: {inputs_embeds.shape}")
                print(f"[inference_sampling] attention_mask shape: {attention_mask.shape}")
            
            # Check model's max position embeddings
            max_position_embeddings = base_model.config.max_position_embeddings
            input_length = inputs_embeds.shape[1]
            max_new_tokens_config = config.get("inference_max_tokens", 64)
            
            # Calculate safe max_new_tokens to avoid exceeding model's context length
            max_new_tokens = min(max_new_tokens_config, max_position_embeddings - input_length - 10)
            
            if max_new_tokens < 10:
                print(f"[inference_sampling] Warning: Very limited generation space for {sample_token}")
                print(f"[inference_sampling]   Input: {input_length}, Max pos: {max_position_embeddings}, Max new: {max_new_tokens}")
                max_new_tokens = max(1, max_new_tokens)  # Force at least 1 token
            
            if config.get("debug_shapes", False):
                print(f"[inference_sampling] Generation params: input_len={input_length}, max_new={max_new_tokens}")
            
            # Use greedy decoding only when BOTH modalities are disabled (for debugging/early training)
            # When at least one modality is enabled, use sampling for more diverse outputs
            use_greedy = not use_vision_in_inference and not use_lidar_in_inference
            
            # Common generation kwargs for speed optimization
            common_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tok.pad_token_id,
                "eos_token_id": tok.eos_token_id,
                "bos_token_id": tok.bos_token_id,
                "use_cache": True,  # Enable KV cache for faster generation
            }
            
            if use_greedy:
                generation_kwargs = {
                    **common_kwargs,
                    "do_sample": False,  # Greedy decoding
                    "num_beams": 1,
                }
            else:
                generation_kwargs = {
                    **common_kwargs,
                    "temperature": config.get("inference_temperature", 0.7),
                    "top_p": config.get("inference_top_p", 0.9),
                    "top_k": config.get("inference_top_k", 50),
                    "do_sample": config.get("inference_do_sample", True),
                    "num_beams": config.get("inference_num_beams", 1),
                    "repetition_penalty": 1.0,
                }
            
            # Generate with torch.inference_mode for speed
            try:
                with torch.inference_mode():
                    outputs = base_model.generate(**generation_kwargs)
                
                # CRITICAL: generate() with inputs_embeds behavior:
                # - Returns ONLY the generated tokens (not input + generated)
                # - The output length will be <= max_new_tokens
                # - We decode the entire output as the prediction
                
                actual_output_length = outputs.shape[1]
                
                # Decode the generated tokens directly
                prediction = tok.decode(outputs[0], skip_special_tokens=True).strip()
                
                if prediction:
                    print(f"[inference_sampling] ✓ Generated {actual_output_length} tokens for {sample_token}")
                    print(f"[inference_sampling]   '{prediction[:100]}{'...' if len(prediction) > 100 else ''}'")
                else:
                    # Empty after decoding - likely only special tokens
                    print(f"[inference_sampling] Warning: Generated {actual_output_length} tokens but empty after decoding")
                    raw_decoded = tok.decode(outputs[0], skip_special_tokens=False)
                    print(f"[inference_sampling]   Raw: '{raw_decoded[:100]}...'")
                
            except Exception as gen_error:
                print(f"[inference_sampling] Generation failed for {sample_token}: {gen_error}")
                import traceback
                traceback.print_exc()
                prediction = ""
            
            sample_result = {
                "sample_token": sample_token,
                "dataset_type": dataset_type,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
            sample_result["metrics"] = calculate_sample_level_metrics(sample_result, config)
            results.append(sample_result)
        
        except Exception as e:
            print(f"[inference_sampling] Error processing {sample.get('sample_token', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate metrics
    print(f"\n[inference_sampling] Calculating metrics for {len(results)} samples...")
    
    if not results:
        print("[inference_sampling] Warning: No results generated, skipping metrics calculation")
        metrics = {}
    else:
        metrics = calculate_metrics_by_type(results, config)
    
    # Save detailed sample manifest and aggregate metrics separately
    samples_manifest = {
        "epoch": epoch,
        "best_step": best_step,
        "timestamp": datetime.now().isoformat(),
        "samples": results,
    }
    samples_file = out_dir / f"inference_sampling_epoch{epoch}.json"
    with open(samples_file, "w") as f:
        json.dump(samples_manifest, f, indent=2)

    summary_payload = {
        "epoch": epoch,
        "best_step": best_step,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(results),
        "metrics": metrics,
        "samples_file": samples_file.name,
    }
    metrics_file = out_dir / f"metrics_summary_epoch{epoch}.json"
    with open(metrics_file, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[inference_sampling] Sample manifest saved to {samples_file}")
    print(f"[inference_sampling] Metrics summary saved to {metrics_file}")
    print(f"\n{'='*60}")
    print("INFERENCE SAMPLING METRICS")
    print('='*60)
    
    if "caption_dashboard" in metrics:
        cap = metrics['caption_dashboard']
        print(f"\nCaption Dashboard ({cap['num_samples']} samples):")
        if "bleu4" in cap:
            print(f"  BLEU-4:       {cap['bleu4']:.4f}")
        if "rouge_l" in cap:
            print(f"  ROUGE-L:      {cap['rouge_l']:.4f}")
        if "meteor" in cap:
            print(f"  METEOR:       {cap['meteor']:.4f}")
        if "cider" in cap:
            print(f"  CIDEr:        {cap['cider']:.4f}")
        if "spice" in cap:
            print(f"  SPICE:        {cap['spice']:.4f}")
        if "bertscore_f1" in cap:
            print(f"  BERTScore-F1: {cap['bertscore_f1']:.4f}")
    
    if "grounding_det_area_dashboard" in metrics:
        gnd = metrics['grounding_det_area_dashboard']
        print(f"\nGrounding det_area Dashboard ({gnd['num_samples']} samples):")
        
        # Print text quality metrics if any are enabled
        text_metrics = [k for k in ["bleu4", "rouge_l", "meteor", "cider", "spice", "bertscore_f1"] if k in gnd]
        if text_metrics:
            print(f"  Text Quality:")
            if "bleu4" in gnd:
                print(f"    BLEU-4:       {gnd['bleu4']:.4f}")
            if "rouge_l" in gnd:
                print(f"    ROUGE-L:      {gnd['rouge_l']:.4f}")
            if "meteor" in gnd:
                print(f"    METEOR:       {gnd['meteor']:.4f}")
            if "cider" in gnd:
                print(f"    CIDEr:        {gnd['cider']:.4f}")
            if "spice" in gnd:
                print(f"    SPICE:        {gnd['spice']:.4f}")
            if "bertscore_f1" in gnd:
                print(f"    BERTScore-F1: {gnd['bertscore_f1']:.4f}")
        
        # Print bbox accuracy metrics if any are enabled
        bbox_metrics = [k for k in ["top1_accuracy", "bev_iou"] if k in gnd]
        if bbox_metrics:
            bbox_valid = gnd.get('bbox_valid_samples', gnd['num_samples'])
            print(f"  Bbox Accuracy ({bbox_valid} valid bbox parses):")
            if "top1_accuracy" in gnd:
                print(f"    Top-1 Acc:    {gnd['top1_accuracy']:.4f}")
            if "bev_iou" in gnd:
                print(f"    BEV IoU:      {gnd['bev_iou']:.4f}")
    
    if "grounding_det_object_dashboard" in metrics:
        obj = metrics['grounding_det_object_dashboard']
        print(f"\nGrounding det_object Dashboard ({obj['num_samples']} samples):")
        
        # Print text quality metrics if any are enabled
        text_metrics = [k for k in ["bleu4", "rouge_l", "meteor", "cider", "spice", "bertscore_f1"] if k in obj]
        if text_metrics:
            print(f"  Text Quality:")
            if "bleu4" in obj:
                print(f"    BLEU-4:       {obj['bleu4']:.4f}")
            if "rouge_l" in obj:
                print(f"    ROUGE-L:      {obj['rouge_l']:.4f}")
            if "meteor" in obj:
                print(f"    METEOR:       {obj['meteor']:.4f}")
            if "cider" in obj:
                print(f"    CIDEr:        {obj['cider']:.4f}")
            if "spice" in obj:
                print(f"    SPICE:        {obj['spice']:.4f}")
            if "bertscore_f1" in obj:
                print(f"    BERTScore-F1: {obj['bertscore_f1']:.4f}")
        
        print(f"  Note: Bbox evaluation skipped (coords in question)")
    
    print('='*60 + '\n')
    
    # Restore training mode and gradient checkpointing
    if was_training_base:
        base_model.train()
    if was_training_lidar:
        vat_lidar_model.train()
    
    # Re-enable gradient checkpointing if it was enabled
    if gradient_checkpointing_was_enabled and hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
    
    # Restore use_cache setting
    base_model.config.use_cache = original_use_cache
    
    if config["use_vision"]:
        if was_training_vision:
            vat_vision_model.train()
        if was_training_adapter:
            vision_adapter_model.train()
        # Restore runtime to training mode (CLIP/Projector train, SAM stays frozen)
        runtime.train()
    
    # Return metrics for live plotting
    return metrics
