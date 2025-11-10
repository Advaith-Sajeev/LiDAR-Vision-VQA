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

from deepencoder.deepencoder_infer import DEFAULT_VIEW_ORDER, multiview_tokens_from_sample_token
from ..utils import calculate_metrics_by_type


@torch.no_grad()
def run_validation(dl, device, tok, base, vat_lidar, vat_vision, vision_adapter, runtime, nusc, config):
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
        # Use runtime.eval() to properly set all components to eval mode
        runtime_unwrapped = unwrap(runtime)

        vat_vision_model.eval()
        vision_adapter_model.eval()
        runtime_unwrapped.eval()
    else:
        vat_vision_model = vision_adapter_model = None

    total_loss = 0.0
    count = 0
    use_amp = config["fp16"] and device.type == "cuda"

    for batch in dl:
        bev = batch["bev"].to(device)
        p_ids = batch["prompt_ids"].to(device)
        a_ids = batch["answer_ids"].to(device)
        sample_tokens = batch["sample_tokens"]

        # Check validation toggles
        use_vision_in_validation = config.get("validation_use_vision", True)
        use_lidar_in_validation = config.get("validation_use_lidar", True)

        # Vision pipeline
        vision_kv = None
        if config["use_vision"] and use_vision_in_validation:
            vision_kvs = []
            for tok_str in sample_tokens:
                mv = multiview_tokens_from_sample_token(
                    tok_str, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                )
                # Safety check
                if not mv.get("tokens") or len(mv["tokens"]) != 6:
                    dummy_shape = (400, 1280)
                    mv["tokens"] = [torch.zeros(dummy_shape, device=device) for _ in range(6)]
                    mv["grid"] = [20, 20]

                vt_list = [t.to(device) for t in mv["tokens"]]

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    kv = vision_adapter_model(vt_list)  # [1536, 2048]
                    kv = kv.unsqueeze(0)  # Add batch dimension: [1, 1536, 2048]
                vision_kvs.append(kv)
            vision_kv = torch.cat(vision_kvs, dim=0)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            E = base_model.get_input_embeddings()

            def emb_token(txt):
                ids = tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                return E(ids)

            # Process LiDAR (if enabled)
            prefix_lidar = None
            if use_lidar_in_validation:
                prefix_lidar = vat_lidar_model(bev) * config["prefix_scale"]
            
            # Process vision (if enabled and available)
            prefix_vision = None
            if vision_kv is not None and use_vision_in_validation:
                prefix_vision = vat_vision_model(vision_kv) * config["prefix_scale"]

            tok_emb = E(p_ids)

            # Build input pieces based on toggles
            pieces = []
            
            # Add vision tokens if enabled
            if prefix_vision is not None:
                pieces += [
                    emb_token("<vision_start>").expand(prefix_vision.size(0), -1, -1),
                    prefix_vision,
                    emb_token("<vision_end>").expand(prefix_vision.size(0), -1, -1),
                ]

            # Add LiDAR tokens if enabled
            if prefix_lidar is not None:
                pieces += [
                    emb_token("<lidar_start>").expand(prefix_lidar.size(0), -1, -1),
                    prefix_lidar,
                    emb_token("<lidar_end>").expand(prefix_lidar.size(0), -1, -1),
                ]
            
            # Always add text prompt
            pieces.append(tok_emb)

            inp = torch.cat(pieces, dim=1)
            ans_emb = E(a_ids)
            inp = torch.cat([inp, ans_emb], dim=1)

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
        # Use runtime.train() to properly restore training mode
        runtime_unwrapped.train()

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
    tok, config, out_dir, epoch, device, token2path, best_step
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
        runtime_unwrapped = unwrap(runtime)
        
        was_training_vision = vat_vision_model.training
        was_training_adapter = vision_adapter_model.training
        
        vat_vision_model.eval()
        vision_adapter_model.eval()
        runtime_unwrapped.eval()
    else:
        vat_vision_model = vision_adapter_model = runtime_unwrapped = None
    
    # Load validation JSONs
    try:
        with open(config["inference_caption_json"], "r", encoding="utf-8") as f:
            caption_data = json.load(f)
        print(f"[inference_sampling] Loaded {len(caption_data)} caption samples")
    except Exception as e:
        print(f"[inference_sampling] Warning: Could not load caption JSON: {e}")
        caption_data = []
    
    try:
        with open(config["inference_grounding_json"], "r", encoding="utf-8") as f:
            grounding_data = json.load(f)
        print(f"[inference_sampling] Loaded {len(grounding_data)} grounding samples")
    except Exception as e:
        print(f"[inference_sampling] Warning: Could not load grounding JSON: {e}")
        grounding_data = []
    
    # Sample n/2 from each
    n_per_type = config["inference_samples_n"] // 2
    
    # Filter for samples that have BEV features
    caption_available = [s for s in caption_data if s.get("sample_token") in token2path]
    grounding_available = [s for s in grounding_data if s.get("sample_token") in token2path]
    
    caption_samples = random.sample(caption_available, min(n_per_type, len(caption_available)))
    grounding_samples = random.sample(grounding_available, min(n_per_type, len(grounding_available)))
    
    print(f"[inference_sampling] Selected {len(caption_samples)} caption + {len(grounding_samples)} grounding samples")
    
    all_samples = [
        {**s, "dataset_type": "caption"} for s in caption_samples
    ] + [
        {**s, "dataset_type": "grounding"} for s in grounding_samples
    ]
    
    # Generate predictions
    results = []
    
    for sample in all_samples:
        try:
            sample_token = sample["sample_token"]
            question = sample.get("question", "").strip()
            ground_truth = sample.get(config["target_field"], "").strip()
            dataset_type = sample["dataset_type"]
            
            # Load BEV feature
            bev_path = token2path.get(sample_token)
            if not bev_path:
                print(f"[inference_sampling] Warning: No BEV feature for {sample_token}")
                continue
            
            bev = np.load(bev_path)
            bev = torch.from_numpy(bev).float().unsqueeze(0).to(device)  # [1, C, H, W]
            
            # Process LiDAR
            prefix_lidar = vat_lidar_model(bev) * config["prefix_scale"]  # [1, n_queries, d_model]
            
            # Process vision
            prefix_vision = None
            if config["use_vision"] and nusc is not None:
                try:
                    mv = multiview_tokens_from_sample_token(
                        sample_token, nusc, runtime=runtime_unwrapped, view_order=DEFAULT_VIEW_ORDER, strict=False
                    )
                    
                    if mv.get("tokens") and len(mv["tokens"]) == 6:
                        vt = [t.to(device) for t in mv["tokens"]]
                        kv = vision_adapter_model(vt)  # [1536, 2048]
                        kv = kv.unsqueeze(0)  # [1, 1536, 2048]
                        prefix_vision = vat_vision_model(kv) * config["prefix_scale"]  # [1, n_queries, d_model]
                except Exception as e:
                    print(f"[inference_sampling] Vision processing failed for {sample_token}: {e}")
            
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
            E = base_model.get_input_embeddings()
            
            def emb_token(txt):
                ids = tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                return E(ids)
            
            # Tokenize the prompt (system + user + generation prompt)
            prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            prompt_embeds = E(prompt_ids)  # [1, L, d_model]
            
            # Build the full input with toggleable components
            # Order matches training: VISION → LIDAR → TEXT
            embeds_list = []
            
            # 1. Vision tokens (if available AND enabled)
            if prefix_vision is not None and use_vision_in_inference:
                embeds_list.append(emb_token("<vision_start>"))
                embeds_list.append(prefix_vision)
                embeds_list.append(emb_token("<vision_end>"))
            
            # 2. LiDAR tokens (if enabled)
            if use_lidar_in_inference:
                embeds_list.append(emb_token("<lidar_start>"))
                embeds_list.append(prefix_lidar)
                embeds_list.append(emb_token("<lidar_end>"))
            
            # 3. Text prompt (user question + optional system prompt + generation prompt)
            embeds_list.append(prompt_embeds)
            
            # Safety check: ensure we have at least some embeddings
            if not embeds_list:
                print(f"[inference_sampling] Error: No components enabled for {sample_token}, skipping...")
                print(f"[inference_sampling]   Check: inference_use_vision, inference_use_lidar, or ensure prompt is not empty")
                continue
            
            # Concatenate all pieces
            inputs_embeds = torch.cat(embeds_list, dim=1)  # [1, total_len, d_model]
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
            
            # Use greedy decoding if both modalities are disabled (for debugging/early training)
            # This is more stable for untrained models
            use_greedy = not (use_vision_in_inference and use_lidar_in_inference)
            if use_greedy:
                generation_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,  # Greedy decoding
                    "num_beams": 1,
                    "pad_token_id": tok.pad_token_id,
                    "eos_token_id": tok.eos_token_id,
                    "bos_token_id": tok.bos_token_id,
                }
            else:
                generation_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "temperature": config.get("inference_temperature", 0.7),
                    "top_p": config.get("inference_top_p", 0.9),
                    "top_k": config.get("inference_top_k", 50),
                    "do_sample": config.get("inference_do_sample", True),
                    "num_beams": config.get("inference_num_beams", 1),
                    "pad_token_id": tok.pad_token_id,
                    "eos_token_id": tok.eos_token_id,
                    "bos_token_id": tok.bos_token_id,
                    "repetition_penalty": 1.0,
                }
            
            # Generate
            try:
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
            
            results.append({
                "sample_token": sample_token,
                "dataset_type": dataset_type,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
            })
        
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
        metrics = calculate_metrics_by_type(results)
    
    # Save results
    output = {
        "epoch": epoch,
        "best_step": best_step,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "samples": results,
    }
    
    output_file = out_dir / f"inference_sampling_epoch{epoch}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[inference_sampling] Results saved to {output_file}")
    print(f"\n{'='*60}")
    print("INFERENCE SAMPLING METRICS")
    print('='*60)
    
    if "caption_dashboard" in metrics:
        print(f"\nCaption Dashboard ({metrics['caption_dashboard']['num_samples']} samples):")
        print(f"  BLEU-4:       {metrics['caption_dashboard']['bleu4']:.4f}")
        print(f"  CIDEr:        {metrics['caption_dashboard']['cider']:.4f}")
        print(f"  SPICE:        {metrics['caption_dashboard']['spice']:.4f}")
        print(f"  BERTScore-F1: {metrics['caption_dashboard']['bertscore_f1']:.4f}")
    
    if "grounding_dashboard" in metrics:
        print(f"\nGrounding Dashboard ({metrics['grounding_dashboard']['num_samples']} samples):")
        print(f"  Top-1 Acc:    {metrics['grounding_dashboard']['top1_accuracy']:.4f}")
        print(f"  BEV IoU:      {metrics['grounding_dashboard']['bev_iou']:.4f}")
    
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
