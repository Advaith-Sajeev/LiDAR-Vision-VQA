#!/usr/bin/env python3
"""
Precompute BEV features (pre-head) for ALL NuScenes samples in two splits,
saving to:
    bev_feats/train/<sample_token>.npy
    bev_feats/test/<sample_token>.npy

- Train dataroot:  /home/j_bindu/fyp-26-grp-38/Dataset_subset
  (contains v1.0-trainval, samples/)
- Test  dataroot:  /home/j_bindu/fyp-26-grp-38/Datasets/nuscenes/test
  (must contain v1.0-test, samples/; i.e., the parent of .../test/samples)

Works with PCDet models WITH or WITHOUT a 2D BEV backbone.
Each file is a float16 numpy array of shape [C, H, W].
"""

from __future__ import annotations
import os, sys, time, gc, random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ======================== USER CONFIG ========================
CONFIG = {
    # Explicit dataroot per split (must contain v1.0-*/ and samples/)
    "SPLIT_DIRS": {
        "train": "/home/j_bindu/fyp-26-grp-38/Dataset_subset",
        # IMPORTANT: this should be the directory that contains v1.0-test and samples/
        "test":  "/home/j_bindu/fyp-26-grp-38/Datasets/nuscenes/test",
    },

    # Base output dir; per-split subfolders will be created automatically
    "FEATURES_DIR": "./bev_feats",

    # PCDet config & checkpoint (absolute or relative to repo root)
    "PCDET_CFG":  "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml",
    "PCDET_CKPT": "models/voxelnext_nuscenes_kernel1.pth",

    # Selection & batching
    "SINGLE_TOKEN": None,   # restrict to a single sample_token if set
    "LIMIT_NEW": 0,         # 0=no per-split limit
    "SHUFFLE": True,
    "SEED": 42,
    "BATCH_SIZE": 100,
    "NUM_WORKERS": 0,

    # Thread throttling (optional)
    "OMP_NUM_THREADS": "14",
    "MKL_NUM_THREADS": "14",
}
# ============================================================

# Thread throttling before heavy libs
os.environ.setdefault("OMP_NUM_THREADS", CONFIG["OMP_NUM_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", CONFIG["MKL_NUM_THREADS"])

# This file lives at <repo>/get-data/*.py, PCDet is in lidar-encoder
REPO_ROOT = Path(__file__).resolve().parents[1] / "lidar-encoder"
sys.path.insert(0, str(REPO_ROOT))

from nuscenes.nuscenes import NuScenes
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# ------------------------------ utils ------------------------------
def _find_version_name(split_dir: Path, split: str) -> Optional[str]:
    """
    Return version dir name if found, else None.
    Prefers 'v1.0-trainval' for train and 'v1.0-test' for test.
    """
    default = "v1.0-trainval" if split == "train" else "v1.0-test"
    if (split_dir / default).is_dir():
        return default
    cand = sorted(p.name for p in split_dir.iterdir()
                  if p.is_dir() and p.name.startswith("v1.0-"))
    if not cand:
        return None
    for c in cand:
        if (split == "train" and "trainval" in c) or (split == "test" and "test" in c):
            return c
    return cand[0] if cand else None


def _prepare_split_paths(split: str):
    """
    Resolve dataroot for a split and sanity-check structure.
    Requires: <dataroot>/<v1.0-*> and <dataroot>/samples
    Creates:  <FEATURES_DIR>/<split> output dir
    """
    split = split.lower().strip()
    assert split in CONFIG["SPLIT_DIRS"], f"Missing dataroot for split '{split}' in CONFIG['SPLIT_DIRS']"
    split_dir = Path(CONFIG["SPLIT_DIRS"][split]).expanduser().resolve()

    version = _find_version_name(split_dir, split)
    if version is None:
        raise RuntimeError(f"Could not find a 'v1.0-*' directory in {split_dir} for split '{split}'")

    for req in [version, "samples"]:
        if not (split_dir / req).exists():
            raise RuntimeError(f"Missing required dir: {split_dir / req}")

    features_dir = Path(CONFIG["FEATURES_DIR"]) / split
    features_dir.mkdir(parents=True, exist_ok=True)

    return split, split_dir, version, features_dir


def to_numpy_feature(x) -> Optional[np.ndarray]:
    """
    Convert common PCDet intermediate features to numpy.
    Accepts: torch.Tensor or spconv SparseConvTensor (via .dense()).
    Returns [B,C,H,W] or [C,H,W] if batch==1.
    """
    if x is None:
        return None
    try:
        t = x.dense() if hasattr(x, "dense") else x
        if isinstance(t, torch.Tensor):
            return t.detach().float().cpu().numpy()
    except Exception:
        return None
    return None


# --------------------------- dataset --------------------------
class MultiFileDataset(DatasetTemplate):
    """
    Items are tuples: (lidar_path: Path, token: str, out_path: Path)
    __getitem__ loads points, runs DATA_PROCESSOR pipeline (voxels, etc.)
    and returns:
      {
        "data": <batch_dict-ready dict>,
        "meta": {"token": str, "out_path": Path}
      }
    """
    def __init__(self, dataset_cfg, class_names, items: List[Tuple[Path, str, Path]], logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=None, logger=logger)
        self.items = items
        if not self.items:
            raise RuntimeError("No items to process.")

    def __len__(self): return len(self.items)

    def _load_points(self, file_path: Path) -> np.ndarray:
        suffix = file_path.suffix.lower()
        if suffix in (".bin", ".pcd.bin"):
            arr = np.fromfile(str(file_path), dtype=np.float32)
            if arr.size % 5 == 0:
                pts = arr.reshape(-1, 5)[:, :4]
            elif arr.size % 4 == 0:
                pts = arr.reshape(-1, 4)
            else:
                raise ValueError(f"Unexpected float count in {file_path.name}: {arr.size}")
        elif suffix == ".npy":
            pts = np.load(str(file_path))
            if pts.ndim == 1:
                pts = pts.reshape(-1, pts.size // 4)
            if pts.shape[1] > 4:
                pts = pts[:, :4]
        else:
            raise NotImplementedError(file_path.suffix)
        return pts.astype(np.float32, copy=False)

    def __getitem__(self, index):
        lidar_path, token, out_path = self.items[index]
        if not lidar_path.exists():
            raise FileNotFoundError(lidar_path)

        points = self._load_points(lidar_path)

        # Run DATA_PROCESSOR pipeline to produce voxels/batch_dict
        data = self.prepare_data({"points": points, "frame_id": index})

        # Drop any accidental string arrays that can confuse collate
        if isinstance(data, dict):
            for k in list(data.keys()):
                v = data[k]
                if isinstance(v, np.ndarray) and v.dtype.kind in ("U", "S"):
                    data.pop(k)

        meta = {"token": token, "out_path": out_path}
        return {"data": data, "meta": meta}

    def collate_with_meta(self, items: List[Dict[str, Any]]):
        data_list = [it["data"] for it in items]
        metas     = [it["meta"] for it in items]
        batch = self.collate_batch(data_list)  # DatasetTemplate's collate
        return batch, metas


# ---------------------------- PCDet load ---------------------------
def load_pcdet(logger):
    """
    Load PCDet model once (frozen). Build from YAML & checkpoint.
    """
    cwd = Path.cwd()
    tools = REPO_ROOT / "tools"
    if not tools.exists():
        tools = REPO_ROOT

    ckpt_path = Path(CONFIG["PCDET_CKPT"])
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / CONFIG["PCDET_CKPT"]).resolve()

    os.chdir(tools)
    try:
        cfg_from_yaml_file(CONFIG["PCDET_CFG"], cfg)
    finally:
        os.chdir(cwd)

    # Dummy dataset only to satisfy DatasetTemplate during build
    dummy_items = [(Path(__file__), "dummy", Path(__file__))]
    dataset = MultiFileDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, dummy_items)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, cfg


# --------------------------- hook helper ---------------------------
class FeatureCatcher:
    """
    Robust feature tap:
      - Always tries a dense_head pre-hook (works for sparse models like VoxelNeXt:
        encoded_spconv_tensor / spatial_features)
      - Optionally hooks backbone_2d if present (SECOND/CenterPoint/etc.)
      - After forward, falls back to reading from batch_dict if needed

    After each forward() call, read self.last as numpy [B,C,H,W] (or [C,H,W] if B==1).
    """
    def __init__(self, model):
        self.model = model
        self.last = None
        self.h1 = None
        self.h2 = None
        self.capture_key = None

    def _pre_head(self, _m, inp):
        try:
            if not inp:
                return
            bd = inp[0]
            if isinstance(bd, dict):
                for k in ("spatial_features_2d", "encoded_spconv_tensor", "spatial_features"):
                    if k in bd:
                        arr = to_numpy_feature(bd[k])
                        if arr is not None:
                            self.capture_key = k
                            self.last = arr
                            break
        except Exception as e:
            print(f"[debug] dense_head pre-hook failed: {e}")

    def _b2d(self, _m, _inp, out):
        try:
            if isinstance(out, dict) and "spatial_features_2d" in out:
                arr = to_numpy_feature(out["spatial_features_2d"])
                if arr is not None:
                    self.capture_key = "spatial_features_2d"
                    self.last = arr
        except Exception as e:
            print(f"[debug] backbone_2d hook failed: {e}")

    def __enter__(self):
        # Hook 1: dense head pre-hook (works for sparse models like VoxelNeXt)
        if getattr(self.model, "dense_head", None) is not None:
            self.h1 = self.model.dense_head.register_forward_pre_hook(self._pre_head)

        # Hook 2: optional 2D backbone (only if present)
        b2d = getattr(self.model, "backbone_2d", None)
        if b2d:
            try:
                self.h2 = b2d.register_forward_hook(self._b2d)
            except Exception as e:
                print(f"[info] skipping backbone_2d hook: {e}")
                self.h2 = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.h1: self.h1.remove()
        if self.h2: self.h2.remove()


# ----------------------------- main -------------------------------
def main():
    # seeds
    random.seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])

    logger = common_utils.create_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[device] CPU mode")

    # Build model once and reuse across splits
    model, cfg_obj = load_pcdet(logger)

    total_created = 0
    for split in ("train", "test"):
        split, SPLIT_DIR, VERSION, FEATURES_DIR_SPLIT = _prepare_split_paths(split)
        print(f"\n=== Processing split: {split} (version: {VERSION}) ===")
        print(f"[dataroot] {SPLIT_DIR}")

        # Initialize NuScenes with the split's dataroot
        nusc = NuScenes(version=VERSION, dataroot=str(SPLIT_DIR), verbose=False)

        # tokens — all samples in this split (or SINGLE_TOKEN)
        if CONFIG["SINGLE_TOKEN"]:
            tokens = [CONFIG["SINGLE_TOKEN"]]
        else:
            tokens = [s["token"] for s in nusc.sample]
            if CONFIG["SHUFFLE"]:
                random.shuffle(tokens)

        # build item list (skip existing)
        items: List[Tuple[Path, str, Path]] = []
        for tok in tokens:
            out_path = FEATURES_DIR_SPLIT / f"{tok}.npy"
            if out_path.exists():
                continue
            try:
                sample = nusc.get("sample", tok)
                sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                lidar_path = (SPLIT_DIR / sd["filename"]).resolve()
                if not lidar_path.is_file():
                    raise FileNotFoundError(lidar_path)
                items.append((lidar_path, tok, out_path))
                if CONFIG["LIMIT_NEW"] and len(items) >= int(CONFIG["LIMIT_NEW"]):
                    break
            except Exception as e:
                print(f"[skip] {tok}: {e}")

        if not items:
            print(f"[info] Nothing to do for split '{split}' (all features exist or no valid tokens).")
            continue

        print(f"[plan] {len(items)} samples → {FEATURES_DIR_SPLIT}")

        ds = MultiFileDataset(cfg_obj.DATA_CONFIG, cfg_obj.CLASS_NAMES, items, logger=logger)
        loader = DataLoader(
            ds, batch_size=int(CONFIG["BATCH_SIZE"]), shuffle=bool(CONFIG["SHUFFLE"]),
            num_workers=int(CONFIG["NUM_WORKERS"]), pin_memory=(device.type == "cuda"),
            collate_fn=ds.collate_with_meta, drop_last=False
        )

        created = 0
        with FeatureCatcher(model) as tap, tqdm(total=len(ds), unit="frame", desc=f"Extract[{split}]") as pbar:
            for (batch, metas) in loader:
                # Move numeric tensors to GPU
                load_data_to_gpu(batch)

                tap.last = None
                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = model.forward(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0

                arr = tap.last
                # Fallback: read from mutated batch_dict if hook missed
                if arr is None:
                    for k in ("spatial_features_2d", "encoded_spconv_tensor", "spatial_features"):
                        if k in batch:
                            arr = to_numpy_feature(batch[k])
                            if arr is not None:
                                tap.capture_key = k
                                break

                if arr is None:
                    raise RuntimeError("Failed to capture BEV features for a batch.")

                # Ensure [B,C,H,W]
                if arr.ndim == 3:
                    arr = arr[None, ...]
                B = arr.shape[0]
                assert B == len(metas), f"Mismatched batch: arr.B={B} metas={len(metas)}"

                # Save each sample
                for i in range(B):
                    out_path: Path = metas[i]["out_path"]
                    bev = arr[i]  # [C,H,W]
                    np.save(out_path, bev.astype(np.float16, copy=False))
                    created += 1

                # Optional: show capture source & perf
                src = tap.capture_key or "unknown"
                pbar.set_postfix_str(f"{arr.shape[1:]}  {dt:.1f} ms/batch  from:{src}")
                pbar.update(B)

                # free transient memory
                del arr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"[done:{split}] created={created} saved to {FEATURES_DIR_SPLIT}")
        total_created += created

    print(f"\n[ALL DONE] total_created={total_created} saved under {CONFIG['FEATURES_DIR']}")


if __name__ == "__main__":
    main()
