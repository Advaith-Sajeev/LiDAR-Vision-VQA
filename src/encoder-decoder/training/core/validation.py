"""Validation and inference utilities"""

import json
import random
import torch
from pathlib import Path
from typing import Dict, Optional

from deepencoder.deepencoder_infer import DEFAULT_VIEW_ORDER, multiview_tokens_from_sample_token


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
        projector_model = unwrap(runtime.projector)
        clip_model = unwrap(runtime.clip_vit)

        vat_vision_model.eval()
        vision_adapter_model.eval()
        projector_model.eval()
        clip_model.eval()
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

        # Vision pipeline
        if config["use_vision"]:
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
                grid_side = mv["grid"][0]

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    kv = vision_adapter_model(vt_list, grid_side)
                vision_kvs.append(kv)
            vision_kv = torch.cat(vision_kvs, dim=0)
        else:
            vision_kv = None

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            E = base_model.get_input_embeddings()

            def emb_token(txt):
                ids = tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                return E(ids)

            prefix_lidar = vat_lidar_model(bev) * config["prefix_scale"]
            prefix_vision = (
                vat_vision_model(vision_kv) * config["prefix_scale"] if vision_kv is not None else None
            )

            tok_emb = E(p_ids)

            pieces = []
            if prefix_vision is not None:
                pieces += [
                    emb_token("<vision_start>").expand(prefix_vision.size(0), -1, -1),
                    prefix_vision,
                    emb_token("<vision_end>").expand(prefix_vision.size(0), -1, -1),
                ]

            pieces += [
                emb_token("<lidar_start>").expand(prefix_lidar.size(0), -1, -1),
                prefix_lidar,
                emb_token("<lidar_end>").expand(prefix_lidar.size(0), -1, -1),
                tok_emb,
            ]

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
        projector_model.train()
        clip_model.train()

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
    if was_training_adapter:
        vision_adapter_model.train()
