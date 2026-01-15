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
from ..utils import calculate_metrics_by_type, calculate_sample_level_metrics, debug
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
            # Wrap with autocast: FP32 trainable weights receive auto-cast FP16 inputs
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Batched vision encoding (much faster than sequential)
                batch_view_tokens = batch_multiview_tokens_from_sample_tokens(
                    sample_tokens, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                )
                
                # Move all tensors to device
                batch_view_tokens_device = [
                    [t.to(device) for t in view_tokens]
                    for view_tokens in batch_view_tokens
                ]
                
                vision_kv = vision_adapter_model.forward_batch(batch_view_tokens_device)
                
        except Exception as e:
            # Fallback to sequential processing
            print(f"[validation] Batched vision encoding failed, using sequential: {e}")
            vision_kvs = []
            for tok_str in sample_tokens:
                # Wrap with autocast
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    mv = multiview_tokens_from_sample_token(
                        tok_str, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                    )
                    # Safety check
                    if not mv.get("tokens") or len(mv["tokens"]) != NUM_VIEWS:
                        dummy_shape = (TOKENS_PER_VIEW, PROJECTOR_DIM)
                        mv["tokens"] = [torch.zeros(dummy_shape, device=device) for _ in range(NUM_VIEWS)]

                    vt_list = [t.to(device) for t in mv["tokens"]]

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
    val_dataset=None,
    train_dataset=None,
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

    def _restore_state():
        # Restore training mode and gradient checkpointing
        if was_training_base:
            base_model.train()

        if gradient_checkpointing_was_enabled and hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

        base_model.config.use_cache = original_use_cache

        if was_training_adapter:
            vision_adapter_model.train()

        # Restore runtime to training mode (CLIP stays frozen by requires_grad=False)
        runtime.train()
    
    def _extract_rows(ds):
        if ds is None:
            return None
        # torch.utils.data.Subset
        if hasattr(ds, "dataset") and hasattr(ds, "indices") and hasattr(ds.dataset, "rows"):
            return [ds.dataset.rows[i] for i in ds.indices]
        # full dataset
        if hasattr(ds, "rows"):
            return list(ds.rows)
        return None

    def _load_json_rows(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of records in {path}, got {type(data)}")
        return data

    def _run_one_source(*, caption_data, source_label: str, out_tag: str, n_samples: int):
        """Run inference sampling on a specific caption_data list and save under out_tag."""
        if n_samples <= 0:
            return [], {}

        # ===== INFERENCE DATA VALIDATION =====
        if nusc is not None and config.get("validate_image_paths", True):
            all_tokens = list(set([s.get("sample_token") for s in caption_data if s.get("sample_token")] ))
            if all_tokens:
                print(f"\n[inference_sampling] Validating camera image paths ({out_tag}) for {len(all_tokens)} unique tokens...")
                image_validation = validate_image_paths(
                    nusc=nusc,
                    sample_tokens=all_tokens,
                    view_order=DEFAULT_VIEW_ORDER,
                    num_workers=config.get("image_validation_workers", 16),
                )
                if image_validation.get("tokens_with_missing", 0) > 0:
                    print(f"[inference_sampling] ⚠️  {image_validation['tokens_with_missing']} samples have missing camera views ({out_tag})")

        caption_available = [s for s in caption_data if s.get("sample_token")]
        if len(caption_available) < n_samples:
            print(f"[inference_sampling] Warning ({out_tag}): need {n_samples}, have {len(caption_available)}. Using all available.")
            caption_samples = caption_available
        else:
            caption_samples = random.sample(caption_available, n_samples)

        all_samples = [{**s, "dataset_type": "caption", "split": out_tag} for s in caption_samples]

        inference_batch_size = max(1, int(config.get("inference_batch_size", 8)))
        print(f"\n[inference_sampling] Pre-encoding {len(all_samples)} samples ({out_tag}, batch_size={inference_batch_size})...")

        encoded_samples = []
        for batch_start in range(0, len(all_samples), inference_batch_size):
            batch_end = min(batch_start + inference_batch_size, len(all_samples))
            batch_samples = all_samples[batch_start:batch_end]
            batch_tokens = [s["sample_token"] for s in batch_samples]

            batch_prefix_vision = None
            if nusc is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    batch_view_tokens = batch_multiview_tokens_from_sample_tokens(
                        batch_tokens, nusc, runtime=runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                    )
                try:
                    batch_view_tokens_device = [[t.to(device) for t in view_tokens] for view_tokens in batch_view_tokens]
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        batch_prefix_vision = vision_adapter_model.forward_batch(batch_view_tokens_device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[inference_sampling] Batched vision encoding OOM ({out_tag}), retrying sequentially...")
                        torch.cuda.empty_cache()
                        sequential_outputs = []
                        for single_views in batch_view_tokens:
                            single_views_device = [t.to(device) for t in single_views]
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                single_prefix = vision_adapter_model.forward_batch([single_views_device])
                            sequential_outputs.append(single_prefix)
                        batch_prefix_vision = torch.cat(sequential_outputs, dim=0)
                    else:
                        print(f"[inference_sampling] Batched vision encoding failed ({out_tag}): {e}")
                        batch_prefix_vision = None
                except Exception as e:
                    print(f"[inference_sampling] Batched vision encoding failed ({out_tag}): {e}")
                    batch_prefix_vision = None

            for i, sample in enumerate(batch_samples):
                if batch_prefix_vision is not None:
                    sample["_vision_views"] = batch_prefix_vision[i]
                else:
                    sample["_vision_views"] = None
                encoded_samples.append(sample)

        print(f"[inference_sampling] ✓ Pre-encoding complete ({out_tag}): {len(encoded_samples)} samples ready")

        results = []
        for sample in encoded_samples:
            try:
                sample_token = sample["sample_token"]
                question = sample.get("question", "").strip()
                ground_truth = sample.get(config["target_field"], "").strip()

                vision_views_list = sample.get("_vision_views")
                use_system_in_inference = config.get("inference_use_system", True)

                if use_system_in_inference:
                    system_prompt = config.get(
                        "system_prompt",
                        "You are an expert autonomous driving assistant. Analyze the camera images to understand the driving scene.",
                    )
                    msgs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]
                else:
                    msgs = [{"role": "user", "content": question}]

                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

                with torch.autocast(device_type=device.type, dtype=model_dtype, enabled=use_amp):
                    E = base_model.get_input_embeddings()

                    def emb_token(txt):
                        ids = tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                        return E(ids)

                    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                    prompt_embeds = E(prompt_ids)

                    inputs_embeds, _ = build_inference_sequence(
                        device=device,
                        dtype=model_dtype,
                        batch_size=1,
                        tok_emb=prompt_embeds,
                        view_tokens_list=vision_views_list,
                        get_special_token_emb=emb_token,
                    )

                inputs_embeds = inputs_embeds.to(dtype=model_dtype)
                attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)

                max_position_embeddings = base_model.config.max_position_embeddings
                input_length = inputs_embeds.shape[1]
                max_new_tokens_config = config.get("inference_max_tokens", 64)
                max_new_tokens = min(max_new_tokens_config, max_position_embeddings - input_length - 10)
                if max_new_tokens < 1:
                    max_new_tokens = 1

                do_sample = config.get("inference_do_sample", True)
                temperature = config.get("inference_temperature", 0.7)
                if temperature <= 1e-5:
                    do_sample = False

                generation_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tok.pad_token_id,
                    "eos_token_id": tok.eos_token_id,
                    "bos_token_id": tok.bos_token_id,
                    "use_cache": True,
                    "do_sample": do_sample,
                    "num_beams": config.get("inference_num_beams", 1),
                    "repetition_penalty": 1.0,
                }
                if do_sample:
                    generation_kwargs.update(
                        {
                            "temperature": temperature,
                            "top_p": config.get("inference_top_p", 0.9),
                            "top_k": config.get("inference_top_k", 50),
                        }
                    )

                with torch.inference_mode():
                    outputs = base_model.generate(**generation_kwargs)

                prediction = tok.decode(outputs[0], skip_special_tokens=True).strip()

                sample_result = {
                    "source": source_label,
                    "split": out_tag,
                    "sample_token": sample_token,
                    "dataset_type": "caption",
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                }
                sample_result["metrics"] = calculate_sample_level_metrics(sample_result, config)
                results.append(sample_result)
            except Exception as e:
                print(f"[inference_sampling] Error ({out_tag}) processing {sample.get('sample_token', 'unknown')}: {e}")
                continue

        if results:
            metrics = calculate_metrics_by_type(results, config)
        else:
            metrics = {}

        samples_manifest = {
            "epoch": epoch,
            "best_step": best_step,
            "timestamp": datetime.now().isoformat(),
            "source": source_label,
            "split": out_tag,
            "num_samples": len(results),
            "samples": results,
        }
        samples_file = out_dir / f"inference_sampling_{out_tag}_epoch{epoch}.json"
        with open(samples_file, "w") as f:
            json.dump(samples_manifest, f, indent=2)

        summary_payload = {
            "epoch": epoch,
            "best_step": best_step,
            "timestamp": datetime.now().isoformat(),
            "source": source_label,
            "split": out_tag,
            "num_samples": len(results),
            "metrics": metrics,
            "samples_file": samples_file.name,
        }
        metrics_file = out_dir / f"metrics_summary_{out_tag}_epoch{epoch}.json"
        with open(metrics_file, "w") as f:
            json.dump(summary_payload, f, indent=2)

        print(f"\n[inference_sampling] ({out_tag}) Sample manifest saved to {samples_file}")
        print(f"[inference_sampling] ({out_tag}) Metrics summary saved to {metrics_file}")
        return results, metrics

    # ==============================
    # Mode A: Split-aware train/val (optionally external test file)
    # ==============================
    train_n = int(config.get("inference_train_samples_n", 0) or 0)
    # NOTE: "test" here means "second split".
    # If inference_test_caption_json is not provided, we fall back to the validation split.
    test_n = int(config.get("inference_test_samples_n", 0) or 0)
    test_json_path = config.get("inference_test_caption_json")

    if train_n > 0 or test_n > 0:
        all_results = []

        if train_n > 0:
            train_rows = _extract_rows(train_dataset)
            if not train_rows:
                print("[inference_sampling] Warning: train_dataset not available; skipping train inference sampling.")
            else:
                r_train, _ = _run_one_source(
                    caption_data=train_rows,
                    source_label="train_split",
                    out_tag="train",
                    n_samples=train_n,
                )
                all_results.extend(r_train)

        if test_n > 0:
            if test_json_path:
                # External test set provided
                try:
                    test_rows = _load_json_rows(test_json_path)
                    r_test, _ = _run_one_source(
                        caption_data=test_rows,
                        source_label=f"file: {test_json_path}",
                        out_tag="test",
                        n_samples=test_n,
                    )
                    all_results.extend(r_test)
                except Exception as e:
                    print(f"[inference_sampling] Warning: Could not load test JSON ({test_json_path}): {e}")
            else:
                # No test set available: fall back to validation split
                val_rows = _extract_rows(val_dataset) if val_dataset is not None else None
                if not val_rows:
                    print("[inference_sampling] Warning: No test JSON and no val_dataset available; skipping val/test sampling.")
                else:
                    r_val, _ = _run_one_source(
                        caption_data=val_rows,
                        source_label="validation_split",
                        out_tag="val",
                        n_samples=test_n,
                    )
                    all_results.extend(r_val)

        # Return combined metrics for plotting compatibility
        combined_metrics = calculate_metrics_by_type(all_results, config) if all_results else {}
        _restore_state()
        return combined_metrics

    # ==============================
    # Mode B: Legacy val/file mixing
    # ==============================
    # Priority 1: Explicit inference_caption_json
    # Priority 2: Validation split from trainer (val_dataset)
    # Priority 3: Fallback to full caption_json

    caption_data = []
    inference_source = "unknown"

    caption_json_path = config.get("inference_caption_json")
    if caption_json_path:
        try:
            caption_data = _load_json_rows(caption_json_path)
            inference_source = f"file: {caption_json_path}"
            print(f"[inference_sampling] Loaded {len(caption_data)} samples from: {inference_source}")
        except Exception as e:
            print(f"[inference_sampling] Warning: Could not load inference JSON: {e}")
            caption_data = []
    elif val_dataset is not None:
        try:
            caption_data = _extract_rows(val_dataset) or []
            if caption_data:
                inference_source = "validation_split"
                print(f"[inference_sampling] Using validation split ({len(caption_data)} samples) from trainer.")
        except Exception as e:
            print(f"[inference_sampling] Warning: Failed to extract samples from val_dataset: {e}")
            caption_data = []

    if not caption_data:
        caption_json_path = config.get("caption_json")
        if caption_json_path:
            print("[inference_sampling] Warning: No inference file or validation set available. Falling back to FULL training data.")
            try:
                caption_data = _load_json_rows(caption_json_path)
                inference_source = f"fallback_full: {caption_json_path}"
                print(f"[inference_sampling] Loaded {len(caption_data)} samples from: {inference_source}")
            except Exception as e:
                print(f"[inference_sampling] Error: Could not load fallback JSON: {e}")
        else:
            print("[inference_sampling] Error: No data source available for inference sampling!")
    
    # For legacy mode, run using the existing output naming.
    total_n = int(config.get("inference_samples_n", 0) or 0)
    if total_n <= 0:
        print("[inference_sampling] inference_samples_n<=0; skipping inference sampling.")
        _restore_state()
        return {}

    legacy_results, legacy_metrics = _run_one_source(
        caption_data=caption_data,
        source_label=inference_source,
        out_tag="val",
        n_samples=total_n,
    )

    # Preserve older filenames as an alias for tooling expectations
    if legacy_results:
        samples_manifest = {
            "epoch": epoch,
            "best_step": best_step,
            "timestamp": datetime.now().isoformat(),
            "samples": legacy_results,
        }
        samples_file = out_dir / f"inference_sampling_epoch{epoch}.json"
        with open(samples_file, "w") as f:
            json.dump(samples_manifest, f, indent=2)

        summary_payload = {
            "epoch": epoch,
            "best_step": best_step,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(legacy_results),
            "metrics": legacy_metrics,
            "samples_file": samples_file.name,
        }
        metrics_file = out_dir / f"metrics_summary_epoch{epoch}.json"
        with open(metrics_file, "w") as f:
            json.dump(summary_payload, f, indent=2)

    _restore_state()
    return legacy_metrics
