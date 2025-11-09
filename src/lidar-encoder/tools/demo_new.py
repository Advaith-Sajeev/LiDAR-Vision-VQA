#!/usr/bin/env python3
# tools/demo_new.py
"""
PCDet demo with:
 - robust checkpoint path resolution + clear errors
 - rich debug logs (input points stats, preds stats, label histogram)
 - score-threshold override to avoid empty outputs
 - per-sample NPZ saved under <output_dir>/<file_stem>/pred_XXXX.npz
 - optional feature embeddings via forward hook
"""

# ========================= CONFIGURATION =========================
CONFIG = {
    'cfg_file': 'cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml',  # config yaml path relative to tools/
    'data_path': './300Samples',  # point cloud file or directory
    'ckpt': '../models/voxelnext_nuscenes_kernel1.pth',  # pretrained checkpoint
    'ext': '.bin',  # file extension for point cloud data: .bin (binary) or .npy (numpy array)
    'output_dir': 'demo_outputs',  # root directory to save prediction outputs
    'save_features': 'demo_features',  # directory to save feature embeddings (npy, one per sample) or None to disable
    'max_samples': None,  # maximum number of samples to process (None = process all files in data_path)
}
# ==================================================================

import glob
from pathlib import Path
import os
import sys
import time
from collections import Counter
from typing import Dict, Any, Tuple

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# ----------------------- Dataset -----------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = Path(root_path)
        self.ext = ext
        if self.root_path.is_dir():
            data_file_list = sorted(glob.glob(str(self.root_path / f'*{self.ext}')))
        else:
            data_file_list = [str(self.root_path)]
        self.sample_file_list = data_file_list

        if logger:
            logger.info(f"[dataset] root={self.root_path} ext={self.ext} files={len(self.sample_file_list)}")

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        sample_path = self.sample_file_list[index]
        if self.ext == '.bin':
            points = np.fromfile(sample_path, dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(sample_path)
            if points.ndim == 1:
                points = points.reshape(-1, points.size // 4)
        else:
            raise NotImplementedError(f"Unsupported ext: {self.ext}")

        file_stem = Path(sample_path).stem  # for nuScenes *.pcd.bin this becomes "... .pcd"
        input_dict = {
            'points': points,
            'frame_id': index,
            'file_name': file_stem,
            'lidar_path': sample_path,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['file_name'] = file_stem
        data_dict['lidar_path'] = sample_path
        return data_dict


# ------------------ Config / Thresholds ----------------
def log_postproc_thresholds(logger, cfg_obj):
    if not hasattr(cfg_obj.MODEL, "POST_PROCESSING"):
        return
    pp = cfg_obj.MODEL.POST_PROCESSING
    st = getattr(pp, "SCORE_THRESH", None)
    if isinstance(st, dict):
        logger.info("[config] SCORE_THRESH per class:")
        for k, v in st.items():
            logger.info(f"  - {k}: {v}")
    else:
        logger.info(f"[config] SCORE_THRESH: {st}")


# ----------------- Checkpoint resolution ----------------
def resolve_ckpt_path(arg_path: str) -> Tuple[Path, str]:
    """
    Try several locations to resolve the checkpoint:
      1) user-specified (expanded) path
      2) repo_root/models/<filename>
      3) CWD/models/<filename>
      4) glob search under repo_root/models for same basename
    Returns (found_path, debug_message)
    """
    tried = []
    p = Path(arg_path).expanduser()
    tried.append(str(p))
    if p.is_file():
        return p, f"[ckpt] Using provided path: {p}"

    # repo root: this file lives in <repo>/tools/demo_new.py
    repo_root = Path(__file__).resolve().parents[1]
    cand = repo_root / "models" / p.name
    tried.append(str(cand))
    if cand.is_file():
        return cand, f"[ckpt] Using repo models/: {cand}"

    cand2 = Path.cwd() / "models" / p.name
    tried.append(str(cand2))
    if cand2.is_file():
        return cand2, f"[ckpt] Using CWD models/: {cand2}"

    # last try: glob under repo_root/models
    hits = list((repo_root / "models").glob(f"**/{p.name}"))
    if hits:
        tried.extend(map(str, hits))
        return hits[0], f"[ckpt] Found via glob: {hits[0]}"

    msg = "[ckpt] Could not find checkpoint. Tried:\n  " + "\n  ".join(tried)
    return p, msg


# ------------------------ I/O -------------------------
def save_predictions(output_dir: Path, subfolder: str, idx: int, pred_dict: Dict[str, Any], logger):
    out_subdir = output_dir / subfolder
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_file = out_subdir / f"pred_{idx:04d}.npz"

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    pred_boxes  = to_np(pred_dict.get('pred_boxes',  np.empty((0, 7), np.float32)))
    pred_scores = to_np(pred_dict.get('pred_scores', np.empty((0,),  np.float32)))
    pred_labels = to_np(pred_dict.get('pred_labels', np.empty((0,),  np.int64)))

    np.savez(out_file, pred_boxes=pred_boxes, pred_scores=pred_scores, pred_labels=pred_labels)
    if pred_scores.size:
        logger.info(f"[save] {out_file}  boxes={pred_boxes.shape[0]}  scores=({pred_scores.min():.3f},{pred_scores.max():.3f})")
    else:
        logger.info(f"[save] {out_file}  boxes=0")


def save_features(save_features_dir: Path, subfolder: str, idx: int, features: np.ndarray, logger):
    out_subdir = save_features_dir / subfolder
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_file = out_subdir / f"embedding_{idx:04d}.npy"
    np.save(out_file, features)
    logger.info(f"[feature] {out_file}  shape={features.shape}")


# ---------------------- Debug helpers -----------------
def points_stats(points: np.ndarray) -> str:
    if points.size == 0:
        return "points=0"
    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    return (f"points={len(points)} "
            f"x=({mins[0]:.1f},{maxs[0]:.1f}) "
            f"y=({mins[1]:.1f},{maxs[1]:.1f}) "
            f"z=({mins[2]:.1f},{maxs[2]:.1f})")

def preds_stats(pred_dict: Dict[str, Any], class_names) -> str:
    boxes  = pred_dict.get('pred_boxes')
    scores = pred_dict.get('pred_scores')
    labels = pred_dict.get('pred_labels')
    n = int(boxes.shape[0]) if boxes is not None else 0
    if n == 0:
        return "preds: 0 boxes"
    smin = float(scores.min().item()) if isinstance(scores, torch.Tensor) else float(scores.min())
    smax = float(scores.max().item()) if isinstance(scores, torch.Tensor) else float(scores.max())
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    hist = Counter(labels_np.tolist())
    parts = [f"preds: {n} boxes  scores=({smin:.3f},{smax:.3f})  labels:"]
    for lb, cnt in sorted(hist.items()):
        idx = int(lb) - 1
        cname = class_names[idx] if 0 <= idx < len(class_names) else f"id{lb}"
        parts.append(f"{cname}={cnt}")
    return " ".join(parts)


# ---------------------- Config Loading -------------------
def load_config():
    """Load configuration from the CONFIG dict at top of file"""
    original_cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / 'tools'

    try:
        os.chdir(tools_dir)
        cfg_from_yaml_file(CONFIG['cfg_file'], cfg)
    finally:
        os.chdir(original_cwd)

    return CONFIG, cfg


def main():
    args, cfg_obj = load_config()
    logger = common_utils.create_logger()
    logger.info('----------------- PCDet Demo [Inference Only] with Debug Logs -------------------------')
    logger.info(f"[env] torch={torch.__version__}  numpy={np.__version__}")

    # log the original thresholds from config
    log_postproc_thresholds(logger, cfg_obj)

    # resolve checkpoint path
    ckpt_path, ckpt_msg = resolve_ckpt_path(args['ckpt'])
    logger.info(ckpt_msg)
    if not ckpt_path.is_file():
        # print a small listing of likely ckpts to help
        repo_root = Path(__file__).resolve().parents[1]
        hints = list((repo_root / "models").glob("**/*.pth"))
        hint_lines = "\n  ".join(map(str, hints[:10]))  # show up to 10
        logger.error("[ckpt] Not found. A few model files under repo models/:\n  " + (hint_lines or "(none)"))
        raise FileNotFoundError(str(ckpt_path))

    # dataset
    demo_dataset = DemoDataset(
        dataset_cfg=cfg_obj.DATA_CONFIG, class_names=cfg_obj.CLASS_NAMES, training=False,
        root_path=Path(args['data_path']), ext=args['ext'], logger=logger
    )
    logger.info(f"[dataset] Total number of samples: {len(demo_dataset)}")

    # model
    logger.info(f"[model] Building from {args['cfg_file']}")
    model = build_network(model_cfg=cfg_obj.MODEL, num_class=len(cfg_obj.CLASS_NAMES), dataset=demo_dataset)
    logger.info(f"[model] Loading checkpoint: {ckpt_path}")
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    model.cuda().eval()

    # optional feature hook
    extract_features = args['save_features'] is not None
    features_list = []
    handle = None
    if extract_features:
        def hook_fn(module, _input, output):
            if isinstance(output, dict):
                feat = output.get('spatial_features', None)
                if feat is None:
                    feat = output.get('spatial_features_2d', None)
                if feat is not None:
                    features_list.append(feat.detach().cpu().numpy())
            else:
                features_list.append(output.detach().cpu().numpy())

        backbone_2d = getattr(model, 'backbone_2d', None)
        backbone = getattr(model, 'backbone', None)
        if backbone_2d is not None:
            handle = backbone_2d.register_forward_hook(hook_fn)
            logger.info("[feature] Hooked model.backbone_2d")
        elif backbone is not None:
            handle = backbone.register_forward_hook(hook_fn)
            logger.info("[feature] Hooked model.backbone")
        else:
            logger.warning("[feature] No backbone_2d/backbone module available; will attempt fallback extraction")

    output_dir = Path(args['output_dir'])
    feature_dir = Path(args['save_features']) if extract_features else None
    feature_warned = False

    # inference loop
    with torch.no_grad():
        N = len(demo_dataset) if args['max_samples'] is None else min(args['max_samples'], len(demo_dataset))
        for idx in range(N):
            data_dict = demo_dataset[idx]
            file_name = data_dict['file_name']
            lidar_path = data_dict.get('lidar_path', '?')

            # input stats
            pts_np = data_dict['points']
            logger.info(f"[{idx+1:04d}/{N:04d}] {lidar_path}")
            logger.info(f"  -> {points_stats(pts_np)}")

            # collate & GPU
            del data_dict['file_name']
            del data_dict['lidar_path']
            data_batch = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_batch)

            # forward timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_dicts, _ = model.forward(data_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - t0

            # log predictions
            pred0 = pred_dicts[0]
            logger.info(f"  -> inference_time={infer_time * 1000:.2f} ms")
            logger.info("  -> " + preds_stats(pred0, cfg_obj.CLASS_NAMES))

            # save
            save_predictions(output_dir, file_name, idx, pred0, logger)

            # features
            if extract_features:
                feature_np = None
                if features_list:
                    feature_np = features_list.pop(0)
                    features_list.clear()
                else:
                    fallback_sources = [
                        ('spatial_features_2d', data_batch.get('spatial_features_2d')),
                        ('spatial_features', data_batch.get('spatial_features'))
                    ]
                    for name, value in fallback_sources:
                        if value is None:
                            continue
                        if isinstance(value, torch.Tensor):
                            feature_np = value.detach().cpu().numpy()
                        else:
                            try:
                                feature_np = np.asarray(value)
                            except Exception:
                                feature_np = None
                        if feature_np is not None:
                            break
                    if feature_np is None:
                        encoded = data_batch.get('encoded_spconv_tensor')
                        if encoded is not None:
                            try:
                                dense_feat = encoded.dense()
                                feature_np = dense_feat.detach().cpu().numpy()
                            except Exception:
                                feature_np = None

                if feature_np is not None:
                    save_features(feature_dir, file_name, idx, feature_np, logger)
                elif not feature_warned:
                    logger.warning("[feature] Could not capture features for saving; skipping")
                    feature_warned = True

    if handle is not None:
        handle.remove()

    logger.info(f"[done] Predictions saved to: {output_dir}")
    if extract_features:
        logger.info(f"[done] Feature embeddings saved to: {feature_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
