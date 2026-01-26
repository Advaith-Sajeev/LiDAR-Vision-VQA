#!/usr/bin/env python3
"""
Extract BEV features for a random sample from a CSV file.
Saves features (.npy) and optional visualization (.png) to ./output directory.

Designed to run on HPC with existing paths.
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from nuscenes.nuscenes import NuScenes

# ======================== USER CONFIG ========================
CONFIG = {
    # Path to the CSV file containing tokens (can be absolute or relative)
    "CSV_PATH": "/home/j_bindu/fyp-26-grp-38/nuscenes_goal_tokens_refined.csv",

    # HPC Data Roots (from existing config)
    "SPLIT_DIRS": {
        "train": "/home/j_bindu/fyp-26-grp-38/Dataset_subset",
    },

    # Output directory for this script
    "OUTPUT_DIR": "./output",
    
    # Visualization toggles
    "VISUALIZE": True,
    
    # PCDet config & checkpoint
    "PCDET_CFG":  "src/lidar-encoder/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml",
    "PCDET_CKPT": "src/lidar-encoder/models/voxelnext_nuscenes_kernel1.pth",
}
# ============================================================

# Setup paths for PCDet imports
REPO_ROOT = Path(__file__).resolve().parent
LIDAR_ENCODER_ROOT = REPO_ROOT / "src" / "lidar-encoder"
sys.path.insert(0, str(LIDAR_ENCODER_ROOT))

# Now we can import pcdet modules
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate

# --------------------------- Helper Classes ---------------------------

class FeatureCatcher:
    """
    Hooks into the model to capture the feature map before the head.
    Matches the logic in precompute_bev_features.py
    """
    def __init__(self, model):
        self.model = model
        self.last = None
        self.h1 = None
        self.h2 = None
        self.capture_key = None

    def _pre_head(self, _m, inp):
        try:
            if not inp: return
            bd = inp[0]
            if isinstance(bd, dict):
                for k in ("spatial_features_2d", "encoded_spconv_tensor", "spatial_features"):
                    if k in bd:
                        arr = self.to_numpy_feature(bd[k])
                        if arr is not None:
                            self.capture_key = k
                            self.last = arr
                            break
        except Exception as e:
            print(f"[debug] dense_head pre-hook failed: {e}")

    def _b2d(self, _m, _inp, out):
        try:
            if isinstance(out, dict) and "spatial_features_2d" in out:
                arr = self.to_numpy_feature(out["spatial_features_2d"])
                if arr is not None:
                    self.capture_key = "spatial_features_2d"
                    self.last = arr
        except Exception as e:
            print(f"[debug] backbone_2d hook failed: {e}")

    def to_numpy_feature(self, x):
        if x is None: return None
        try:
            t = x.dense() if hasattr(x, "dense") else x
            if isinstance(t, torch.Tensor):
                return t.detach().float().cpu().numpy()
        except Exception:
            return None
        return None

    def __enter__(self):
        # Hook 1: dense head pre-hook
        if getattr(self.model, "dense_head", None) is not None:
            self.h1 = self.model.dense_head.register_forward_pre_hook(self._pre_head)
        # Hook 2: optional 2D backbone
        b2d = getattr(self.model, "backbone_2d", None)
        if b2d:
            self.h2 = b2d.register_forward_hook(self._b2d)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.h1: self.h1.remove()
        if self.h2: self.h2.remove()

class SingleFileDataset(DatasetTemplate):
    """
    Minimal dataset wrapper for a single point cloud file.
    """
    def __init__(self, dataset_cfg, class_names, lidar_path):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=None, logger=None)
        self.lidar_path = Path(lidar_path)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        points = self._load_points(self.lidar_path)
        data = self.prepare_data({"points": points, "frame_id": index})
        return data

    def _load_points(self, file_path):
        suffix = file_path.suffix.lower()
        if suffix in (".bin", ".pcd.bin"):
            arr = np.fromfile(str(file_path), dtype=np.float32)
            if arr.size % 5 == 0:
                pts = arr.reshape(-1, 5)[:, :4]
            elif arr.size % 4 == 0:
                pts = arr.reshape(-1, 4)
            else:
                raise ValueError(f"Unexpected float count: {arr.size}")
        elif suffix == ".npy":
            pts = np.load(str(file_path))
            if pts.ndim == 1: pts = pts.reshape(-1, pts.size // 4)
            if pts.shape[1] > 4: pts = pts[:, :4]
        else:
            raise NotImplementedError(file_path.suffix)
        return pts.astype(np.float32, copy=False)

def load_pcdet(logger):
    """Load the model based on config."""
    cwd = Path.cwd()
    tools_dir = LIDAR_ENCODER_ROOT / "tools"
    
    # Checkpoint path handling
    ckpt_path = Path(CONFIG["PCDET_CKPT"])
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / CONFIG["PCDET_CKPT"]).resolve()

    # Config path handling
    cfg_path = Path(CONFIG["PCDET_CFG"])
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / CONFIG["PCDET_CFG"]).resolve()

    # Change to tools dir to load config (PCDet quirk)
    os.chdir(tools_dir)
    try:
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(cwd)

    # Dummy dataset for building model
    dummy_ds = SingleFileDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, Path("dummy.bin"))
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_ds)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, cfg

def visualize_bev(feature_map, output_path, token):
    """
    Visualize the BEV feature map.
    feature_map: [C, H, W] numpy array
    """
    # Max-pool across channels to get a spatial heatmap
    heatmap = np.max(feature_map, axis=0) 
    # Alternatively: mean_map = np.mean(feature_map, axis=0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap='viridis', origin='lower')
    plt.colorbar(label='Feature Intensity')
    plt.title(f"BEV Feature Map Visualization\nToken: {token}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Viz] Saved visualization to {output_path}")

def find_lidar_path(nusc, token, split_root):
    """Find the absolute path to the LiDAR file for a sample token."""
    sample = nusc.get("sample", token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    filename = sd["filename"]
    return (split_root / filename).resolve()

def main():
    logger = common_utils.create_logger()
    
    # 1. Output Setup
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    viz_dir = out_dir / "bev_viz"
    feat_dir = out_dir / "bev_features"
    viz_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 2. Select Token
    print(f"[1/5] Loading tokens from {CONFIG['CSV_PATH']}...")
    try:
        df = pd.read_csv(CONFIG["CSV_PATH"])
        # Assuming the column is named 'token' or taking the first column
        # Look for 'token' or 'sample_token' column
        token_col = next((c for c in df.columns if 'token' in c.lower()), df.columns[0])
        token = df[token_col].sample(n=1).iloc[0]
        print(f"      Selected Random Token: {token}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Initialize NuScenes to find file path
    # We need to determine if this token is train or test to know which root to use.
    # Since we can't easily know, we might have to try both or rely on the CSV info if present.
    # For now, let's assume it's in the 'train' split directory as default, 
    # or check file existence.
    
    # NOTE: Keep the HPC logic. We'll try to init nusc for 'train' first.
    split_name = "train" # Default to train/val dataset
    split_root = Path(CONFIG["SPLIT_DIRS"][split_name])
    
    # Try to find version
    version = "v1.0-trainval"
    if not (split_root / version).exists():
        # Fallback logic if needed, or specific to HPC structure
        pass

    print(f"[2/5] Initializing NuScenes ({version} in {split_root})...")
    try:
        nusc = NuScenes(version=version, dataroot=str(split_root), verbose=True)
    except Exception as e:
        print(f"Failed to init NuScenes: {e}")
        return

    try:
        lidar_path = find_lidar_path(nusc, token, split_root)
        print(f"      Lidar Path: {lidar_path}")
    except Exception as e:
        print(f"Error finding lidar path (maybe token is in 'test' split?): {e}")
        return

    if not lidar_path.exists():
        print(f"Error: Lidar file does not exist at {lidar_path}")
        return

    # 4. Load Model
    print(f"[3/5] Loading PCDet model...")
    model, cfg_obj = load_pcdet(logger)

    # 5. Inference
    print(f"[4/5] Running Inference...")
    dataset = SingleFileDataset(cfg_obj.DATA_CONFIG, cfg_obj.CLASS_NAMES, lidar_path)
    # Collate single item
    data_dict = dataset[0]
    # Add batch dimension
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = torch.from_numpy(v).unsqueeze(0)
        elif not isinstance(v, (str, int, float)):
             # Handle other types if necessary, but usually just np arrays need unsq
            pass

    load_data_to_gpu(data_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with FeatureCatcher(model) as tap:
        with torch.no_grad():
            model.forward(data_dict)
        
        feature_map = tap.last
        
    if feature_map is None:
        print("Error: Could not capture feature map!")
        return
        
    # Remove batch dim if present [1, C, H, W] -> [C, H, W]
    if feature_map.ndim == 4:
        feature_map = feature_map[0]

    print(f"      Captured Feature Map Shape: {feature_map.shape}")

    # 6. Save & Visualize
    print(f"[5/5] Saving results...")
    
    # Save NPY
    npy_path = feat_dir / f"{token}.npy"
    np.save(npy_path, feature_map.astype(np.float16))
    print(f"      Saved features to {npy_path}")

    # Visualize
    if CONFIG["VISUALIZE"]:
        viz_path = viz_dir / f"{token}.png"
        visualize_bev(feature_map, viz_path, token)

    print("\n[Done] Process completed successfully.")

if __name__ == "__main__":
    main()
