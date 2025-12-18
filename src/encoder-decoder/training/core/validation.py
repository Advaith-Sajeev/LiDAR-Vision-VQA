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
from configs.constants import (
    DEFAULT_VIEW_ORDER,
    NUM_VIEWS,
    TOKENS_PER_VIEW,
    PROJECTOR_DIM,
)
from ..data.utils import validate_image_paths
from ..utils import calculate_metrics_by_type, calculate_sample_level_metrics
from ..utils.sequence_builder import build_training_sequence, build_inference_sequence, ModalityPosition


@torch.no_grad()
def run_validation(dl, device, tok, base, vision_adapter, runtime, nusc, config, use_amp=False, amp_dtype=torch.float32):
    """
    Run validation on validation dataloader.
    
    Args:
        dl: Validation DataLoader
        device: Device to run on
        tok: Tokenizer
        base: Base LLM model
        vision_adapter: Vision adapter
        runtime: DeepEncoder runtime
        nusc: NuScenes instance
        config: Training configuration
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (torch.float16 or torch.bfloat16)
        
    Returns:
        Average validation loss
    """
    # Log validation toggles
    use_vision_validation = config.get("validation_use_vision", True)
    print(f"[validation] Vision enabled: {use_vision_validation}")
    
    # Unwrap DDP if needed
    def unwrap(model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    base_model = unwrap(base)
    base_model.eval()

    vision_adapter_model = unwrap(vision_adapter)
    # Note: runtime is not DDP-wrapped, use directly

    vision_adapter_model.eval()
    runtime.eval()

    total_loss = 0.0
    count = 0

    for batch in dl:
        # Use non_blocking for async CPU→GPU transfers
        # Check validation toggles
        use_vision_in_validation = config.get("validation_use_vision", True)

        # Unpack batch
        p_ids = batch["prompt_ids"].to(device, non_blocking=True)
        a_ids = batch["answer_ids"].to(device, non_blocking=True)
        sample_tokens = batch["sample_tokens"]

        # Vision pipeline - use batched processing for efficiency
        # Vision pipeline - use batched processing for efficiency
        vision_kv = None
        
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
            # Process vision (if enabled and available)
            batched_view_tokens = None
            if vision_kv is not None:
                 try:
                    num_views = len(vision_kv[0])
                    batched_view_tokens = []
                    for v_idx in range(num_views):
                        v_tokens = torch.stack([sample[v_idx] for sample in vision_kv], dim=0)
                        batched_view_tokens.append(v_tokens)
                 except Exception:
                    batched_view_tokens = None

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
                view_tokens_list=batched_view_tokens,
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
    
    vision_adapter_model.train()
    # Restore runtime to training mode
    runtime.train()

    return total_loss / max(1, count)


@torch.no_grad()
def save_val_inference_samples(
    ds_val, tok, base, vision_adapter, runtime, nusc, config, out_dir, epoch, n=10
):
    """
    Generate and save validation inference samples.
    
    Args:
        ds_val: Validation dataset
        tok: Tokenizer
        base: Base LLM model
        vision_adapter: Vision adapter
        runtime: DeepEncoder runtime
        nusc: NuScenes instance
        config: Training configuration
        out_dir: Output directory
        epoch: Current epoch
        n: Number of samples to save
    """


    assert n % 2 == 0, "n must be even"

    # Unwrap DDP
    def unwrap(model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    base_model = unwrap(base)
    was_training_base = base_model.training
    base_model.eval()

    vision_adapter_model = unwrap(vision_adapter)
    was_training_adapter = vision_adapter_model.training
    vision_adapter_model.eval()

    device = next(base_model.parameters()).device

    # Classify validation rows
    samples = list(ds_val.dataset.rows)
    random.shuffle(samples)
    chosen = samples[:n]
    results = []

    for row in chosen:
        try:
            sample_token = row["sample_token"]
            question = row.get("question", "")
            ground_truth = row.get(config["target_field"], "")



            # Vision encoding
            vision_views = None
            if config.get("validation_use_vision", True) and nusc is not None:
                try:
                    mv = multiview_tokens_from_sample_token(
                        sample_token, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                    )
                    if mv.get("tokens") and len(mv["tokens"]) == NUM_VIEWS:
                        view_tokens = [t.to(device) for t in mv["tokens"]]
                        # Use forward_batch for consistency
                        encoded_batch = vision_adapter_model.forward_batch([view_tokens])
                        vision_views = encoded_batch[0]
                except Exception as e:
                    print(f"[warn] Vision encoding failed for {sample_token}: {e}")
                    vision_views = None

            # Generate prediction
            msgs = [
                {
                    "role": "system",
                    "content": "You are a driving assistant. Use camera context provided via prefix tokens.",
                },
                {"role": "user", "content": question},
            ]
            text_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            
            # Build input sequence
            with torch.inference_mode():
                E = base_model.get_input_embeddings()
                prompt_ids = tok(text_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                prompt_embeds = E(prompt_ids)
                
                def emb_token(txt):
                    ids = tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    return E(ids)

                inputs_embeds, _ = build_inference_sequence(
                    device=device,
                    dtype=prompt_embeds.dtype,
                    batch_size=1,
                    tok_emb=prompt_embeds,
                    view_tokens_list=vision_views,
                    get_special_token_emb=emb_token
                )
                
                attn = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)
                
                outputs = base_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn,
                    max_new_tokens=128,
                    do_sample=False, # Use greedy for validation stability
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True
                )
                prediction = tok.decode(outputs[0], skip_special_tokens=True).strip()

            results.append(
                {
                    "sample_token": sample_token,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
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
    if was_training_adapter:
        vision_adapter_model.train()


@torch.no_grad()
def run_inference_sampling(
    base, vision_adapter, runtime, nusc,
    tok, config, out_dir, epoch, device, best_step,
    use_amp=False, amp_dtype=torch.float32,
    val_dataset=None
):
    """
    Generate predictions on validation samples with evaluation metrics.
    
    Samples n/2 from caption validation and n/2 from grounding validation,
    generates predictions using the current best model, and calculates metrics.
    
    Args:
        base: Base LLM model
        vision_adapter: Vision adapter
        runtime: DeepEncoder runtime
        nusc: NuScenes instance
        tok: Tokenizer
        config: Training configuration
        out_dir: Output directory
        epoch: Current epoch number
        device: Device to run on
        best_step: Best step number so far
        val_dataset: Validation dataset (Subset) from the trainer.
    """
    print(f"\n[inference_sampling] Generating predictions at epoch {epoch}...")
    debug.start_timer("validation", "inference_sampling")
    print(f"\n[inference_sampling] Starting inference sampling...")
    
    # Log inference component toggles
    use_vision_toggle = config.get("inference_use_vision", True)
    use_system_toggle = config.get("inference_use_system", True)
    
    print(f"[inference_sampling] Component toggles: Vision={use_vision_toggle}, System={use_system_toggle}")
    
    # Inform about decoding strategy
    if not use_vision_toggle:
        print(f"[inference_sampling] Using GREEDY decoding (modalities disabled, more stable for early training)")
    else:
        print(f"[inference_sampling] Using SAMPLING decoding (temp={config.get('inference_temperature', 0.7)}, top_p={config.get('inference_top_p', 0.9)})")

    
    # Unwrap DDP if needed
    def unwrap(model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    base_model = unwrap(base)
    
    # CRITICAL: Determine model dtype ONCE at the start to avoid double casting
    # The model weights dtype is the ground truth - use this consistently throughout
    model_dtype = next(base_model.parameters()).dtype
    print(f"[inference_sampling] Model dtype: {model_dtype}, AMP dtype: {amp_dtype}, AMP enabled: {use_amp}")
    
    # Set to eval mode
    was_training_base = base_model.training
    
    base_model.eval()
    
    # Disable gradient checkpointing for generation (important!)
    if hasattr(base_model, 'gradient_checkpointing_disable'):
        base_model.gradient_checkpointing_disable()
        gradient_checkpointing_was_enabled = True
    else:
        gradient_checkpointing_was_enabled = False
    
    # Enable cache for generation (important for performance and correctness)
    original_use_cache = base_model.config.use_cache
    base_model.config.use_cache = True
    
    vision_adapter_model = unwrap(vision_adapter)
    # Note: runtime is not DDP-wrapped, use directly
    
    was_training_adapter = vision_adapter_model.training
    
    vision_adapter_model.eval()
    runtime.eval()
    
    # Load caption data logic
    # Priority 1: Explicit inference_caption_json (e.g. test set)
    # Priority 2: Validation split from trainer (val_dataset)
    # Priority 3: Fallback to full caption_json (training + validation mixture)
    
    caption_data = []
    inference_source = "unknown"
    
    caption_json_path = config.get("inference_caption_json")
    
    if caption_json_path:
        # Case 1: Explicit file provided
        try:
            with open(caption_json_path, "r", encoding="utf-8") as f:
                caption_data = json.load(f)
            inference_source = f"file: {caption_json_path}"
            print(f"[inference_sampling] Loaded {len(caption_data)} samples from: {inference_source}")
        except Exception as e:
            print(f"[inference_sampling] Warning: Could not load inference JSON: {e}")
            caption_data = []
            
    elif val_dataset is not None:
        # Case 2: Use validation split (Subset)
        try:
            # Check if it's a Subset (standard from random_split)
            if hasattr(val_dataset, 'dataset') and hasattr(val_dataset, 'indices'):
                # access rows from parent dataset using indices
                caption_data = [val_dataset.dataset.rows[i] for i in val_dataset.indices]
                inference_source = "validation_split"
                print(f"[inference_sampling] Using validation split ({len(caption_data)} samples) from trainer.")
            elif hasattr(val_dataset, 'rows'):
                # fallback if passed full dataset directly
                caption_data = val_dataset.rows
                inference_source = "validation_dataset_full"
                print(f"[inference_sampling] Using provided dataset ({len(caption_data)} samples).")
        except Exception as e:
            print(f"[inference_sampling] Warning: Failed to extract samples from val_dataset: {e}")
            caption_data = []

    if not caption_data:
        # Case 3: Fallback to full training JSON
        caption_json_path = config.get("caption_json")
        if caption_json_path:
            print(f"[inference_sampling] Warning: No inference file or validation set available. Falling back to FULL training data.")
            try:
                with open(caption_json_path, "r", encoding="utf-8") as f:
                    caption_data = json.load(f)
                inference_source = f"fallback_full: {caption_json_path}"
                print(f"[inference_sampling] Loaded {len(caption_data)} samples from: {inference_source}")
            except Exception as e:
                print(f"[inference_sampling] Error: Could not load fallback JSON: {e}")
        else:
            print(f"[inference_sampling] Error: No data source available for inference sampling!")
    
    # ===== INFERENCE DATA VALIDATION =====
    # Validate camera image paths for inference data (when using vision)
    # This ensures test data has the same validation as training data
    if nusc is not None and config.get("validate_image_paths", True):
        all_inference_tokens = list(set([s.get("sample_token") for s in caption_data if s.get("sample_token")]))
        if all_inference_tokens:
            print(f"\n[inference_sampling] Validating camera image paths for {len(all_inference_tokens)} unique tokens...")
            image_validation = validate_image_paths(
                nusc=nusc,
                sample_tokens=all_inference_tokens,
                view_order=DEFAULT_VIEW_ORDER,
                num_workers=config.get("image_validation_workers", 16),
                # Check all inference samples (they're typically fewer than training)
            )
            if image_validation.get('tokens_with_missing', 0) > 0:
                print(f"[inference_sampling] ⚠️  {image_validation['tokens_with_missing']} samples have missing camera views")
                print(f"[inference_sampling] ⚠️  Missing views will be filled with zeros, which may affect evaluation quality.")
    
    # =========================================================================
    # SAMPLING STRATEGY BASED ON DATASET MODE
    # =========================================================================
    total_n = config["inference_samples_n"]
    
    # Filter for samples that have valid sample_token
    caption_available = [s for s in caption_data if s.get("sample_token")]

    
    # Calculate sample distribution based on dataset_mode
    print(f"[inference_sampling] Sampling strategy (caption only, total={total_n}):")
    print(f"  Caption: {total_n} samples (100%)")
    
    # Validate sufficient samples
    if len(caption_available) < total_n:
        print(f"[inference_sampling] Warning: Insufficient caption samples: need {total_n}, have {len(caption_available)}.")
        print(f"[inference_sampling] Using all available samples.")
        caption_samples = caption_available
    else:
        caption_samples = random.sample(caption_available, total_n)
        
    print(f"\n[inference_sampling] ✓ Sampled exactly:")
    print(f"  Caption: {len(caption_samples)} samples")
    print(f"  Total: {len(caption_samples)} samples")
    
    all_samples = [
        {**s, "dataset_type": "caption"} for s in caption_samples
    ]
    
    # ===== BATCHED ENCODING PHASE =====
    # Pre-encode vision features in batches for efficiency
    # This avoids redundant per-sample encoding during generation
    
    inference_batch_size = max(1, int(config.get("inference_batch_size", 8)))  # Encode in batches
    print(f"\n[inference_sampling] Pre-encoding {len(all_samples)} samples (batch_size={inference_batch_size})...")
    
    # Filter valid samples and prepare for batched encoding
    valid_samples = []
    for sample in all_samples:
        sample_token = sample["sample_token"]
        valid_samples.append(sample)
    
    # Pre-compute all encodings in batches
    encoded_samples = []
    
    for batch_start in range(0, len(valid_samples), inference_batch_size):
        batch_end = min(batch_start + inference_batch_size, len(valid_samples))
        batch_samples = valid_samples[batch_start:batch_end]
        batch_tokens = [s["sample_token"] for s in batch_samples]
        
        # Batched vision encoding
        batch_prefix_vision = None
        if nusc is not None:
            batch_view_tokens = batch_multiview_tokens_from_sample_tokens(
                batch_tokens, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
            )
            try:
                batch_view_tokens_device = [
                    [t.to(device) for t in view_tokens]
                    for view_tokens in batch_view_tokens
                ]
                
                # VisionAdapter output goes directly to LLM (no VAT compression)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    batch_prefix_vision = vision_adapter_model.forward_batch(batch_view_tokens_device)  # [B, 1536, d_model]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[inference_sampling] Batched vision encoding OOM, retrying sequentially (batch_size=1)...")
                    torch.cuda.empty_cache()
                    sequential_outputs = []
                    for single_views in batch_view_tokens:
                        single_views_device = [t.to(device) for t in single_views]
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            single_prefix = vision_adapter_model.forward_batch([single_views_device])
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
            # Collate vision tokens properly:
            # batch_prefix_vision is List[List[Tensor]] [Batch][View]
            if batch_prefix_vision:
                sample["_vision_views"] = batch_prefix_vision[i] # List[Tensor] (for this sample)
            else:
                sample["_vision_views"] = None
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
            vision_views_list = sample.get("_vision_views")  # List[Tensor] or None

            
            # Format prompt with configurable system prompt and toggles
            # CRITICAL: Match the exact order used during training!
            # BUT allow toggling components for debugging/ablation studies
            
            # Check inference toggles (default to True for backward compatibility)
            use_system_in_inference = config.get("inference_use_system", True)
            
            # Build prompt based on system toggle
            if use_system_in_inference:
                system_prompt = config.get(
                    "system_prompt", 
                    "You are an expert autonomous driving assistant. Analyze the camera images to understand the driving scene. Provide accurate, concise descriptions of objects, their locations, distances, and spatial relationships. Use directional terms like 'ahead', 'left', 'right', 'behind' and specify distances in meters when describing object locations."
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
                effective_vision = vision_views_list
                
                # Safety check: ensure we have at least some content
                if effective_vision is None and prompt_embeds is None:
                    print(f"[inference_sampling] Error: No components enabled for {sample_token}, skipping...")
                    print(f"[inference_sampling]   Check: inference_use_vision or ensure prompt is not empty")
                    continue
                
                inputs_embeds, sequence_metadata = build_inference_sequence(
                    device=device,
                    dtype=model_dtype,  # Use model_dtype directly, not amp_dtype
                    batch_size=1,
                    tok_emb=prompt_embeds,
                    view_tokens_list=effective_vision,
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
            
            # Use greedy decoding only when Vision is disabled (for debugging)
            # When vision is enabled, use sampling for more diverse outputs
            use_greedy = False
            
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
    

    
    print('='*60 + '\n')
    
    # Restore training mode and gradient checkpointing
    if was_training_base:
        base_model.train()

    
    # Re-enable gradient checkpointing if it was enabled
    if gradient_checkpointing_was_enabled and hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
    
    # Restore use_cache setting
    base_model.config.use_cache = original_use_cache
    
    if was_training_adapter:
        vision_adapter_model.train()
    # Restore runtime to training mode (CLIP/Projector train, SAM stays frozen)
    runtime.train()
    
    # Return metrics for live plotting
    return metrics
