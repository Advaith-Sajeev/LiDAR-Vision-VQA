#!/usr/bin/env python3
# tools/print_arch_voxelnext.py
"""
Comprehensive architecture & tensor printer for VoxelNeXt (OpenPCDet).

What it does:
- Prints a hierarchical module tree with parameter counts.
- Runs a forward pass over 1 sample and:
  * Logs batch_dict key/shape after every top-level module in model.module_list.
  * Hooks EVERY LEAF LAYER to print input/output shapes as they execute.
  * Hooks the internal SHC / map-to-BEV inside the head and flags the exact BEV "tap" tensor.

Run:
  python tools/print_arch_voxelnext.py

Tweak CONFIG below if needed.
"""

from __future__ import annotations
from pathlib import Path
import glob
import os
import time
from typing import Dict, Any, List, Tuple, Iterable, Union

import numpy as np
import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# ========================= CONFIG =========================
CONFIG = {
    'cfg_file': 'cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml',  # relative to tools/
    'data_path': './300Samples',               # a dir or single file (.bin/.npy)
    'ext': '.bin',                             # '.bin' or '.npy'
    'ckpt': '../models/voxelnext_nuscenes_kernel1.pth',
    'max_samples': 1,                          # print for one sample
    'print_leaf_io_maxlen': 6,                 # cap number of tensors printed per hook side
    'suppress_empty_tensors': True,            # hide empty tensors in hook logging
}
# ==========================================================


# ---------------------- Dataset -------------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=training, root_path=root_path, logger=logger)
        self.root_path = Path(root_path)
        self.ext = ext
        if self.root_path.is_dir():
            data_file_list = sorted(glob.glob(str(self.root_path / f'*{self.ext}')))
        else:
            data_file_list = [str(self.root_path)]
        if not data_file_list:
            raise FileNotFoundError(f"No input files found under {self.root_path} with ext={self.ext}")
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        p = self.sample_file_list[index]
        if self.ext == '.bin':
            pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            pts = np.load(p)
            if pts.ndim == 1:
                pts = pts.reshape(-1, pts.size // 4)
        else:
            raise NotImplementedError(f"Unsupported ext: {self.ext}")
        input_dict = {
            'points': pts,
            'frame_id': index,
            'file_name': Path(p).stem,   # kept for logging; removed before GPU
            'lidar_path': p,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


# ------------------- Utility: shapes/summary --------------------
def is_sparse_conv_tensor(x) -> bool:
    return hasattr(x, "features") and hasattr(x, "spatial_shape")

def fmt_shape(x) -> str:
    try:
        if is_sparse_conv_tensor(x):
            ss = tuple(int(v) for v in x.spatial_shape)
            nf = tuple(x.features.shape)
            return f"SparseConvTensor(features={nf}, spatial_shape={ss})"
        if isinstance(x, torch.Tensor):
            return f"Tensor{tuple(x.shape)}"
        if isinstance(x, np.ndarray):
            return f"ndarray{tuple(x.shape)}"
        if isinstance(x, (int, float, bool)):
            return f"{type(x).__name__}({x})"
        return type(x).__name__
    except Exception:
        return type(x).__name__

def summarize_batch_keys(bd: Dict[str, Any]) -> str:
    # Show common keys and anything interesting
    keys_priority = {
        'points', 'voxels', 'voxel_coords', 'voxel_num_points', 'voxel_features',
        'encoded_spconv_tensor', 'encoded_spconv_tensor_stride',
        'multi_scale_3d_features', 'multi_scale_3d_strides',
        'spatial_features', 'spatial_features_2d', 'multi_scale_2d_features',
        'pred_dicts', 'final_box_dicts', 'batch_size'
    }
    parts = []
    for k in sorted(bd.keys()):
        if k.startswith('_'):  # internal
            continue
        v = bd[k]
        if isinstance(v, dict):
            # summarize nested dict keys
            if k in ('multi_scale_3d_features', 'multi_scale_2d_features'):
                sub = ", ".join([f"{kk}:{fmt_shape(vv)}" for kk, vv in v.items()])
                parts.append(f"{k}={{ {sub} }}")
            else:
                parts.append(f"{k}=dict[{len(v)}]")
        else:
            if k in keys_priority or isinstance(v, (torch.Tensor, np.ndarray)) or is_sparse_conv_tensor(v):
                parts.append(f"{k}={fmt_shape(v)}")
    return " | ".join(parts) if parts else "(no notable keys)"

# ------------------- GPU/Batch sanitize -------------------
def drop_nontensor_fields(batch: Dict[str, Any]) -> None:
    to_drop = []
    for k, v in batch.items():
        if isinstance(v, (str, list, dict)):
            to_drop.append(k)
        elif isinstance(v, np.ndarray) and v.dtype.kind in ('U', 'S', 'O'):
            to_drop.append(k)
    for k in to_drop:
        batch.pop(k, None)

# ------------------- CKPT resolve -------------------
def resolve_ckpt_path(arg_path: str) -> Path:
    p = Path(arg_path).expanduser()
    if p.is_file():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    cand = repo_root / "models" / p.name
    if cand.is_file():
        return cand
    hits = list((repo_root / "models").glob(f"**/{p.name}"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Checkpoint not found: {p}")

# ------------------- Module tree printing -------------------
def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def module_tree_lines(model: nn.Module, prefix: str = "", name: str = "") -> List[str]:
    """Recursively build a pretty module tree with param counts."""
    cls = model.__class__.__name__
    params = count_parameters(model)
    head = f"{prefix}{name}({cls}) [params: {params:,}]"
    lines = [head]
    children = list(model.named_children())
    if not children:
        return lines
    # next prefix pieces
    for i, (child_name, child_mod) in enumerate(children):
        is_last = (i == len(children) - 1)
        branch = "└─ " if is_last else "├─ "
        pad = "   " if is_last else "│  "
        lines.extend(module_tree_lines(child_mod, prefix + pad, f"{branch}{child_name}"))
    return lines

def print_module_tree(logger, model: nn.Module):
    logger.info("===== Full Module Tree (with trainable param counts) =====")
    for line in module_tree_lines(model, "", "model"):
        logger.info(line)
    logger.info("=========================================================")

# ------------------- Leaf I/O hook logging -------------------
def is_leaf(m: nn.Module) -> bool:
    return len(list(m.children())) == 0

def summarize_tensors(seq: Iterable[Any], maxlen: int, suppress_empty=True) -> str:
    parts = []
    cnt = 0
    for x in seq:
        if isinstance(x, (list, tuple)):
            # flatten one level
            parts.append(summarize_tensors(x, maxlen, suppress_empty))
            continue
        if isinstance(x, dict):
            keys = ", ".join(list(x.keys())[:maxlen])
            parts.append(f"dict[{len(x)}] keys=({keys})")
            continue
        if isinstance(x, (torch.Tensor, np.ndarray)) or is_sparse_conv_tensor(x):
            if suppress_empty and hasattr(x, "numel") and int(x.numel()) == 0:
                continue
            parts.append(fmt_shape(x))
            cnt += 1
            if cnt >= maxlen:
                break
        else:
            parts.append(fmt_shape(x))
            cnt += 1
            if cnt >= maxlen:
                break
    return " ; ".join(parts) if parts else "-"

def register_leaf_hooks(model: nn.Module, logger) -> List[Any]:
    handles: List[Any] = []
    maxlen = CONFIG['print_leaf_io_maxlen']
    suppress = CONFIG['suppress_empty_tensors']

    def hook_fn(path):
        def _hook(module, inputs, output):
            in_sum = summarize_tensors(inputs if isinstance(inputs, (list, tuple)) else (inputs,), maxlen, suppress)
            out_sum = ""
            if isinstance(output, dict):
                # show common keys and specific tap keys if present
                keys = list(output.keys())
                out_sum = f"dict[{len(keys)}] keys={keys[:10]}"
                if 'spatial_features_2d' in output:
                    logger.info(f"[LEAF:{path}] >>> TAP DETECTED (BEV in dict): spatial_features_2d -> {fmt_shape(output['spatial_features_2d'])}")
            else:
                out_sum = summarize_tensors((output,), maxlen, suppress)
            logger.info(f"[LEAF:{path}] in: {in_sum}")
            logger.info(f"[LEAF:{path}] out: {out_sum}")
        return _hook

    for name, m in model.named_modules():
        if name == "" or name == "":  # skip root registered twice
            pass
        if is_leaf(m):
            try:
                h = m.register_forward_hook(hook_fn(name))
                handles.append(h)
            except Exception:
                # Some modules might not allow hooks; skip safely
                continue
    logger.info(f"[hooks] Registered on {len(handles)} leaf layers")
    return handles

# -------------------- Hook SHC / map-to-BEV (explicit) -------------------
def attach_bev_hooks(model: nn.Module, logger) -> List[Any]:
    handles: List[Any] = []

    def mk(name):
        def _fn(module, _inp, out):
            bev = None
            if isinstance(out, dict):
                bev = out.get('spatial_features_2d', None)
                ms = out.get('multi_scale_2d_features', None)
                if bev is not None:
                    logger.info(f"[HOOK:{name}] spatial_features_2d -> {fmt_shape(bev)}")
                    out['_tap_spatial_features_2d'] = bev  # mirror for visibility
                if isinstance(ms, dict):
                    logger.info(f"[HOOK:{name}] multi_scale_2d_features keys -> {list(ms.keys())}")
            elif isinstance(out, torch.Tensor):
                # Rare case
                bev = out
                logger.info(f"[HOOK:{name}] (tensor) -> {fmt_shape(bev)}")
            if bev is not None:
                logger.info("      >>> TAP HERE (BEV): 'spatial_features_2d'")
        return _fn

    candidates: List[Tuple[str, nn.Module]] = []
    for n, m in model.named_modules():
        nlow = n.lower()
        clow = m.__class__.__name__.lower()
        if any(tag in nlow for tag in ('map_to_bev', 'map2bev', 'height_compress', 'heightcompression', 'shc')) \
           or any(tag in clow for tag in ('maptobev', 'heightcompression', 'sparseheightcompression')):
            candidates.append((n, m))

    # Also hook the head itself, which usually owns SHC for VoxelNeXt
    if hasattr(model, 'dense_head'):
        candidates.append(('dense_head', model.dense_head))

    seen = set()
    for n, m in candidates:
        if id(m) in seen:
            continue
        seen.add(id(m))
        try:
            handles.append(m.register_forward_hook(mk(n)))
            logger.info(f"[hook] Attached to: {n} ({m.__class__.__name__})")
        except Exception:
            continue

    if not handles:
        logger.warning("[hook] No SHC/map-to-BEV candidates found; BEV may not be exposed")

    return handles

# ------------------------ Main --------------------------
def main():
    logger = common_utils.create_logger()
    logger.info("--------------- VoxelNeXt Architecture Printer ---------------")
    logger.info(f"[env] torch={torch.__version__}")

    # Load cfg from repo root/tools/
    original_cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / 'tools'
    try:
        os.chdir(tools_dir)
        cfg_from_yaml_file(CONFIG['cfg_file'], cfg)
    finally:
        os.chdir(original_cwd)

    # Dataset
    ds = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=CONFIG['data_path'], ext=CONFIG['ext'], logger=logger
    )
    logger.info(f"[dataset] samples={len(ds)} (showing up to {CONFIG['max_samples']})")

    # Model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    ckpt = resolve_ckpt_path(CONFIG['ckpt'])
    logger.info(f"[model] cfg={CONFIG['cfg_file']}  ckpt={ckpt}")
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=True)
    model.cuda().eval()

    # Print the full hierarchical tree (with params)
    print_module_tree(logger, model)

    # Print top-level module_list (execution order)
    logger.info("----- Execution Order (model.module_list) -----")
    for i, m in enumerate(model.module_list):
        logger.info(f"[{i:02d}] {m.__class__.__name__}")
    logger.info("-----------------------------------------------")

    # Attach generic leaf hooks (print all leaf layer I/O) + explicit BEV hooks
    leaf_handles = register_leaf_hooks(model, logger)
    bev_handles = attach_bev_hooks(model, logger)

    # One-sample forward with per-stage summaries
    n = min(CONFIG['max_samples'] or len(ds), len(ds))
    with torch.no_grad():
        for i in range(n):
            data_dict = ds[i]
            file_name = data_dict.get('file_name', None)
            lidar_path = data_dict.get('lidar_path', None)
            data_dict.pop('file_name', None)
            data_dict.pop('lidar_path', None)

            batch = ds.collate_batch([data_dict])
            drop_nontensor_fields(batch)
            if file_name or lidar_path:
                logger.info(f"[meta] file_name={file_name}  lidar_path={lidar_path}")

            load_data_to_gpu(batch)

            logger.info(f"=== Forward pass on sample #{i} ===")
            logger.info("[init] " + summarize_batch_keys(batch))

            for step_idx, module in enumerate(model.module_list):
                name = module.__class__.__name__
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                batch = module(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                logger.info(f"[{step_idx:02d}] {name:<28} -> {summarize_batch_keys(batch)}  ({dt_ms:.1f} ms)")

                # If BEV mirrored by hook, call it out explicitly
                if '_tap_spatial_features_2d' in batch and isinstance(batch['_tap_spatial_features_2d'], torch.Tensor):
                    logger.info(f"      >>> TAP HERE (BEV): _tap_spatial_features_2d {fmt_shape(batch['_tap_spatial_features_2d'])}")

            # Final head output quick stat
            preds = batch.get('pred_dicts', None)
            if preds:
                p0 = preds[0]
                nb = int(p0.get('pred_boxes', torch.empty(0)).shape[0]) if isinstance(p0.get('pred_boxes', None), torch.Tensor) else 0
                logger.info(f"[done] head produced {nb} boxes")

    # Clean hooks
    for h in leaf_handles + bev_handles:
        try:
            h.remove()
        except Exception:
            pass

    logger.info("------------------------------------------------------------------")
    logger.info("Tap guidance:")
    logger.info(" - For VoxelNeXt (3D backbone + 2D head), use the BEV after SHC/map-to-BEV.")
    logger.info(" - In configs where BEV is global: batch_dict['spatial_features_2d'].")
    logger.info(" - In your cfg, BEV lives inside the head; the hook mirrors it as '_tap_spatial_features_2d'.")
    logger.info("------------------------------------------------------------------")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        raise
