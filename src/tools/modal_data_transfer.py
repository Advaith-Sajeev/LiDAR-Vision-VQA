#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modal Data Copy Tool :: src/tools/modal_data_transfer.py

This script runs on Modal to copy inference samples (images only) from the attached volume
into a structured batch format on the volume.

It does NOT:
- Load the model
- Run inference
- Save sample.json
- Save LiDAR/BEV files

Usage:
    modal run src/tools/modal_data_transfer.py
"""

import math
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Optional, Sequence

import modal

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "out_dir": "/data/DATA",  # Output on the volume
    "num_batches": 5,
    "dataset_mode": "caption",  # "caption", "grounding", "both"
    "dataroot": "/data/Datasets/nuScenes", # Path on the volume
    "version": "v1.0-trainval",
}
# ============================================================================

app = modal.App("lidar-vision-data-copy")
volume = modal.Volume.from_name("lidar-llm", create_if_missing=False)

# Lightweight image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "nuscenes-devkit>=1.1.0",
        "numpy",
        "tqdm",
    )
    .add_local_dir(
        local_path="./src",
        remote_path="/root/src",
        copy=False,
    )
)

# Helper function to avoid importing from deepencoder
def resolve_cam_image_paths(
    nusc,
    sample_token: str,
    view_order: Sequence[str],
) -> List[Optional[Path]]:
    """Resolve absolute image paths for the specified views from a nuScenes sample token."""
    sample = nusc.get("sample", sample_token)
    out: List[Optional[Path]] = []
    for cam in view_order:
        sd_tok = sample["data"].get(cam, None)
        if not sd_tok:
            out.append(None)
            continue
        sd = nusc.get("sample_data", sd_tok)
        p = (Path(nusc.dataroot) / sd["filename"]).resolve()
        out.append(p if p.exists() else None)
    return out

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,  # 2 hour timeout
    cpu=40.0,      # 40 CPU cores
    memory=16384,  # 16GB Memory
)
def copy_samples_remote():
    import json
    import sys
    import concurrent.futures
    from tqdm import tqdm
    
    # Add src to sys.path
    src_path = "/root/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from nuscenes import NuScenes
    from configs.modal_config import get_modal_training_config
    # We can import DEFAULT_VIEW_ORDER from configs.default_config as it has no heavy deps
    from configs.default_config import DEFAULT_VIEW_ORDER

    print("=" * 80)
    print("🚀 STARTING REMOTE DATA COPY (PARALLELIZED)")
    print("=" * 80)

    # 1. Load Configuration
    print("Loading configuration...")
    modal_config = get_modal_training_config()
    
    # Override config with local CONFIG
    modal_config["dataset_mode"] = CONFIG["dataset_mode"]
    
    # Re-run the dynamic config logic for jsons
    mode = CONFIG["dataset_mode"]
    if mode == "caption":
        modal_config["jsons"] = [modal_config["caption_json"]]
    elif mode == "grounding":
        modal_config["jsons"] = [modal_config["grounding_json"]]
    elif mode == "both":
        modal_config["jsons"] = [modal_config["caption_json"], modal_config["grounding_json"]]
        
    # 2. Initialize nuScenes
    dataroot = CONFIG["dataroot"]
    version = CONFIG["version"]
    
    print(f"Initializing nuScenes (root={dataroot}, version={version})...")
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    except Exception as e:
        print(f"Failed to initialize nuScenes: {e}")
        print("Please ensure 'dataroot' is correct and points to the nuScenes dataset.")
        return

    # 3. Load Data
    print("Loading dataset JSONs...")
    
    def load_json_data(json_paths: List[str]) -> List[Dict]:
        all_data = []
        for path in json_paths:
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                print(f"Warning: JSON file not found: {p}")
                continue
            
            print(f"Loading {p}...")
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict) and "data" in data:
                    all_data.extend(data["data"])
        return all_data

    raw_samples = load_json_data(modal_config["jsons"])
    print(f"Total JSON entries loaded: {len(raw_samples)}")

    # Deduplicate: Get list of unique tokens
    seen_tokens = set()
    samples = []
    for s in raw_samples:
        t = s.get("sample_token")
        if t and t not in seen_tokens:
            seen_tokens.add(t)
            samples.append(t)
    
    total_samples = len(samples)
    unique_tokens = seen_tokens
    print(f"Unique sample tokens to process: {total_samples}")

    if total_samples == 0:
        print("No samples found. Exiting.")
        return

    # 4. Prepare Batches
    num_batches = CONFIG["num_batches"]
    batch_size = math.ceil(total_samples / num_batches)
    print(f"Splitting {total_samples} samples into {num_batches} batches (approx {batch_size} per batch)...")

    # 5. Prepare Output Directory
    base_output_dir = Path(CONFIG["out_dir"])
    if base_output_dir.exists():
        print(f"Warning: Output directory {base_output_dir} already exists.")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

    # 6. Prepare Tasks for Parallel Execution
    tasks = []
    global_idx = 0
    
    print(f"Preparing tasks for {total_samples} samples across {num_batches} batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        batch_samples = samples[start_idx:end_idx]
        
        if not batch_samples:
            break
            
        batch_dir_name = f"batch_{batch_idx + 1}"
        batch_dir = base_output_dir / batch_dir_name
        batch_dir.mkdir(exist_ok=True)
        
        for i, sample_token in enumerate(batch_samples):
            sample_idx_in_batch = i + 1
            tasks.append({
                "sample_token": sample_token,
                "batch_dir": batch_dir,
                "sample_idx_in_batch": sample_idx_in_batch,
                "global_idx": global_idx + i + 1
            })
        
        global_idx += len(batch_samples)

    # Define worker function
    def process_sample(task):
        sample_token = task["sample_token"]
        batch_dir = task["batch_dir"]
        sample_idx_in_batch = task["sample_idx_in_batch"]
        
        if not sample_token:
            return 0
            
        sample_dir_name = f"sample_{sample_idx_in_batch:03d}_{sample_token}"
        sample_dir = batch_dir / sample_dir_name
        sample_dir.mkdir(exist_ok=True)
        
        try:
            # Resolve image paths
            image_paths = resolve_cam_image_paths(nusc, sample_token, view_order=DEFAULT_VIEW_ORDER)
            
            copied_any = False
            for view_idx, (view_name, img_path) in enumerate(zip(DEFAULT_VIEW_ORDER, image_paths)):
                if img_path is None:
                    continue
                
                img_src = Path(img_path)
                if not img_src.exists():
                    # Try prepending dataroot if path is relative
                    if not img_src.is_absolute():
                        img_src = Path(dataroot) / img_path
                
                if not img_src.exists():
                    # print(f"  [Warning] Image not found: {img_src}") # Reduce spam in parallel
                    continue
                
                # Destination filename: {idx:02d}_{view_name}_{original_name}
                dest_name = f"{view_idx:02d}_{view_name}_{img_src.name}"
                img_dest = sample_dir / dest_name
                
                shutil.copy2(img_src, img_dest)
                copied_any = True
            
            if copied_any:
                return 1
            else:
                # Cleanup empty dir
                try:
                    sample_dir.rmdir()
                except:
                    pass
                return 0

        except Exception as e:
            print(f"  Error processing sample {sample_token}: {e}")
            return 0

    # 7. Execute in Parallel
    print(f"Starting parallel execution with 256 workers...")
    total_success_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(process_sample, task) for task in tasks]
        
        # Use tqdm for progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="sample", mininterval=5.0):
            total_success_count += future.result()
            
    volume.commit()

    print("=" * 80)
    print(f"All batches completed. Artifacts location: {base_output_dir}")
    print(f"Total unique samples in JSON: {len(unique_tokens)}")
    print(f"Total sample folders created: {total_success_count}")
    
    if len(unique_tokens) == total_success_count:
        print("✅ SUCCESS: Counts match exactly.")
    else:
        print(f"⚠️ WARNING: Count mismatch! ({len(unique_tokens) - total_success_count} difference)")
    print("=" * 80)


@app.local_entrypoint()
def main():
    copy_samples_remote.remote()

if __name__ == "__main__":
    print("Run with: modal run src/tools/modal_data_transfer.py")
