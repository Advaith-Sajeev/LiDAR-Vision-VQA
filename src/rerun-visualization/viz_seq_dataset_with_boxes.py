#!/usr/bin/env python3
# viz_seq_dataset_with_boxes.py — nuScenes LiDAR + per-frame boxes (+ optional cameras)
# Works with OpenPCDet-style per-frame outputs:
#   /home/jarvis/Documents/OpenPCDet/tools/my_outputs/<frame>.pcd/pred_XXXX.npz

# ── EDIT THESE ───────────────────────────────────────────────────────────────
DATASET_ROOT   = "/home/jarvis/nuscenes_train_bridge"
VERSION        = "v1.0-trainval"

# Your per-frame output root: <root>/<frame>.pcd/pred_XXXX.npz
PRED_PCD_DIR   = ""

# Optional alternate box dir (disabled by default)
BOX_DIR        = None
BOX_EXTS       = [".npz", ".npy", ".pkl"]

SHOW_CAMERAS   = False
SPAWN_VIEW     = True
MAX_FRAMES     = None           # e.g. 200 for quick pass

# ---- Visualization/Filtering knobs ----
MIN_SCORE      = 0.4           # raise to thin detections (0.3–0.5)
TOP_K          = 150            # keep only top-K by score; None=keep all
MAX_RADIUS_M   = 100.0           # drop boxes farther than this; None=disable
LABEL_ALLOW    = None           # e.g. {'car','truck','bus'} or None for all
SKIP_EMPTY_BOX_FRAMES = True    # skip timeline step if no boxes after filters

# Geometry convention fix:
#   "pcdet"      -> boxes=[x,y,z,l,w,h,yaw], yaw 0 along +X (default)
#   "voxelnext"  -> boxes=[x,y,z,w,l,h,yaw], yaw needs +π/2 (CenterPoint/VoxelNeXt style)
#   "centerpoint" same as voxelnext
MODEL_FLAVOR   = "voxelnext"    # << set to "pcdet" if you switch back to PointPillars/SECOND
# ─────────────────────────────────────────────────────────────────────────────

import os, glob, pickle, math
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import rerun as rr
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

CAM_CHANNELS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

NUSC_CLASS_NAMES = [
    "car","truck","construction_vehicle","bus","trailer","barrier",
    "motorcycle","bicycle","pedestrian","traffic_cone"
]

# ---------- helpers ----------
def _norm_base(path_or_rel: str) -> str:
    b = os.path.basename(path_or_rel.rstrip("/"))
    base = os.path.splitext(b)[0]
    if base.endswith(".pcd"):
        base = base[:-4]
    return base

def _find_sd_and_path(nusc: NuScenes, pred_path: str) -> Tuple[str, str]:
    key = _norm_base(pred_path)
    for sd in nusc.sample_data:
        if sd["channel"] != "LIDAR_TOP":
            continue
        sd_key = _norm_base(sd["filename"])
        if sd_key == key:
            lidar_path, _, _ = nusc.get_sample_data(sd["token"])
            return sd["sample_token"], lidar_path
    raise RuntimeError(f"No LIDAR_TOP sample_data found for base '{key}'")

def _load_dataset_lidar(lidar_path: str):
    pc = LidarPointCloud.from_file(lidar_path)    # (4,N): x,y,z,intensity
    pts = pc.points[:3].T.astype(np.float32)
    colors = None
    if pc.points.shape[0] >= 4:
        inten = pc.points[3]
        inten = 255 * (inten - inten.min()) / (max(inten.ptp(), 1e-6))
        colors = np.stack([inten, inten, inten], axis=-1).astype(np.uint8)
    return pts, colors

def _yaw_to_quat_xyzw(y: np.ndarray) -> np.ndarray:
    h = 0.5 * y
    return np.stack([np.zeros_like(y), np.zeros_like(y), np.sin(h), np.cos(h)], axis=-1).astype(np.float32)

def _color_from_label(labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if labels is None: return None
    pal = np.array([[230,57,70],[29,53,87],[69,123,157],[168,218,220],[42,157,143],
                    [233,196,106],[244,162,97],[231,111,81],[87,87,87]], dtype=np.uint8)
    return pal[labels % len(pal)]

# ---- convention normalization -----------------------------------------------
def _apply_model_convention(sizes: np.ndarray, yaws: np.ndarray):
    """Normalize (sizes,yaws) into PCDet viewer convention: [l,w,h], yaw 0 along +X."""
    if MODEL_FLAVOR.lower() in ("voxelnext","centerpoint"):
        # many CenterPoint/VoxelNeXt checkpoints export [w,l,h] and yaw offset +π/2
        sizes = sizes.copy()
        sizes[:, [0,1]] = sizes[:, [1,0]]      # w,l,h -> l,w,h
        yaws = yaws + np.pi/2
        # wrap to [-pi, pi]
        yaws = (yaws + np.pi) % (2*np.pi) - np.pi
    # else "pcdet": leave as-is
    return sizes, yaws

# ---- parsers ----------------------------------------------------------------
def _pick_boxes_key(d):
    # prefer common keys
    for k in ["pred_boxes","boxes_3d","boxes","box3d_lidar","bbox3d","bboxes_3d","bboxes","box_preds","boxes_lidar"]:
        if k in d:
            arr = d[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 7:
                return k
    # fallback: first numeric (N,>=7)
    for k in d.files:
        arr = d[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 7 and np.issubdtype(arr.dtype, np.number):
            return k
    return None

def _parse_boxes_npz(path: str):
    d = np.load(path, allow_pickle=True)
    key = _pick_boxes_key(d)
    if key is None:
        raise ValueError(f"Couldn't find boxes array in {path} (keys={sorted(list(d.files))})")
    boxes = np.asarray(d[key])
    centers = boxes[:,0:3].astype(np.float32)
    sizes   = boxes[:,3:6].astype(np.float32)
    yaws    = boxes[:,6].astype(np.float32)
    # normalize convention (if needed)
    sizes, yaws = _apply_model_convention(sizes, yaws)
    # extras
    scores  = None
    labels  = None
    if "pred_scores" in d: scores = d["pred_scores"].astype(np.float32)
    elif "scores" in d:     scores = d["scores"].astype(np.float32)
    if "pred_labels" in d: labels = d["pred_labels"].astype(np.int32)
    elif "labels" in d:    labels = d["labels"].astype(np.int32)
    names   = NUSC_CLASS_NAMES
    return centers, sizes, yaws, scores, labels, names, key

def _parse_boxes_npy(path: str):
    arr = np.load(path, allow_pickle=True)
    assert arr.ndim == 2 and arr.shape[1] >= 7, f"{path} must be Nx7+"
    centers = arr[:,0:3].astype(np.float32)
    sizes   = arr[:,3:6].astype(np.float32)
    yaws    = arr[:,6].astype(np.float32)
    sizes, yaws = _apply_model_convention(sizes, yaws)
    return centers, sizes, yaws, None, None, NUSC_CLASS_NAMES, "npy:array"

def _parse_boxes_pkl(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "pred_boxes" in obj:
        arr = np.asarray(obj["pred_boxes"])
        centers = arr[:,0:3].astype(np.float32)
        sizes   = arr[:,3:6].astype(np.float32)
        yaws    = arr[:,6].astype(np.float32)
        sizes, yaws = _apply_model_convention(sizes, yaws)
        scores = np.asarray(obj.get("pred_scores", None)).astype(np.float32) if "pred_scores" in obj else None
        labels = np.asarray(obj.get("pred_labels", None)).astype(np.int32)   if "pred_labels" in obj else None
        return centers, sizes, yaws, scores, labels, NUSC_CLASS_NAMES, "pkl:pred_*"
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict) and "pred_boxes" in obj[0]:
        d0 = obj[0]
        arr = np.asarray(d0["pred_boxes"])
        centers = arr[:,0:3].astype(np.float32)
        sizes   = arr[:,3:6].astype(np.float32)
        yaws    = arr[:,6].astype(np.float32)
        sizes, yaws = _apply_model_convention(sizes, yaws)
        scores = np.asarray(d0.get("pred_scores", None)).astype(np.float32) if "pred_scores" in d0 else None
        labels = np.asarray(d0.get("pred_labels", None)).astype(np.int32)   if "pred_labels" in d0 else None
        return centers, sizes, yaws, scores, labels, NUSC_CLASS_NAMES, "pkl:list[0].pred_*"
    raise ValueError(f"Don't know how to parse {path}")

def _find_boxes_for_frame(entry_path: str, frame_idx: int):
    # Prefer pred_{frame_idx:04d}.* in <entry>/<pred_*.ext>
    folder = entry_path if os.path.isdir(entry_path) else os.path.join(os.path.dirname(entry_path), os.path.basename(entry_path))
    if os.path.isdir(folder):
        pref = []
        for ext in (".npz",".npy",".pkl"):
            cand = os.path.join(folder, f"pred_{frame_idx:04d}{ext}")
            if os.path.isfile(cand):
                pref.append(cand)
        if pref:
            path = pref[0]
        else:
            hits = sorted(glob.glob(os.path.join(folder, "pred_*.*")))
            hits = [h for h in hits if os.path.splitext(h)[1].lower() in (".npz",".npy",".pkl")]
            if not hits:
                raise FileNotFoundError(f"no pred_*.npz|.npy|.pkl inside {folder}")
            path = hits[0]
        if path.endswith(".npz"): return _parse_boxes_npz(path), path
        if path.endswith(".npy"): return _parse_boxes_npy(path), path
        if path.endswith(".pkl"): return _parse_boxes_pkl(path), path

    # BOX_DIR fallback
    if BOX_DIR:
        base = _norm_base(entry_path)
        for ext in BOX_EXTS:
            cand = os.path.join(BOX_DIR, base + ext)
            if os.path.isfile(cand):
                if ext == ".npz": return _parse_boxes_npz(cand), cand
                if ext == ".npy": return _parse_boxes_npy(cand), cand
                if ext == ".pkl": return _parse_boxes_pkl(cand), cand
    raise FileNotFoundError(f"no boxes found for {entry_path}")

# ---- timeline shim ----------------------------------------------------------
def set_frame_timeline(i: int):
    try:
        rr.set_time(sequence="frame", value=i)
    except TypeError:
        try:
            rr.set_time(sequence="frame", time=i)
        except Exception:
            rr.set_time_sequence("frame", i)

# ---- filtering & drawing ----------------------------------------------------
def _apply_filters(centers, sizes, yaws, scores, labels, names):
    idx = np.arange(len(centers))

    # score
    if scores is not None and MIN_SCORE is not None:
        idx = idx[(scores[idx] >= float(MIN_SCORE))]

    # label set
    if LABEL_ALLOW is not None and labels is not None:
        keep = []
        for i in idx:
            lab = int(labels[i])
            cname = names[lab-1] if 1 <= lab <= len(names) else None  # PCDet labels are 1-based
            if cname in LABEL_ALLOW:
                keep.append(i)
        idx = np.array(keep, dtype=int)

    # radius
    if MAX_RADIUS_M is not None:
        d2 = np.sum(centers[idx,:2]**2, axis=1)
        idx = idx[d2 <= (MAX_RADIUS_M**2)]

    # top-k by score
    if TOP_K is not None and scores is not None and len(idx) > TOP_K:
        part = np.argpartition(-scores[idx], TOP_K-1)[:TOP_K]
        idx = idx[part]

    # slicing
    centers = centers[idx]
    sizes   = sizes[idx]
    yaws    = yaws[idx]
    scores  = (scores[idx] if scores is not None else None)
    labels  = (labels[idx] if labels is not None else None)
    return centers, sizes, yaws, scores, labels, names

def log_boxes3d(path_prefix: str, centers, sizes, yaws, scores, labels, names):
    # labels text (optional)
    labels_txt = None
    if labels is not None or scores is not None:
        labels_txt = []
        for j in range(len(centers)):
            parts = []
            if labels is not None:
                lab = int(labels[j])
                cname = names[lab-1] if 1 <= lab <= len(names) else f"id{lab}"
                parts.append(cname)
            if scores is not None:
                parts.append(f"{float(scores[j]):.2f}")
            labels_txt.append(" ".join(parts))

    colors_boxes = _color_from_label(labels)
    quats = _yaw_to_quat_xyzw(yaws)
    try:
        rr.log(path_prefix, rr.Boxes3D(
            centers=centers, sizes=sizes,
            rotations=rr.Quaternion(xyzw=quats),
            colors=colors_boxes, labels=labels_txt
        ))
        return "sizes"
    except Exception:
        rr.log(path_prefix, rr.Boxes3D(
            centers=centers, half_sizes=(np.asarray(sizes)*0.5),
            rotations=rr.Quaternion(xyzw=quats),
            colors=colors_boxes, labels=labels_txt
        ))
        return "half_sizes"

# -----------------------------------------------------------------------------
def main():
    # Frame list = folders named *.pcd in PRED_PCD_DIR
    preds = sorted(glob.glob(os.path.join(PRED_PCD_DIR, "*.pcd")))
    if not preds:
        raise SystemExit(f"No entries matching *.pcd in {PRED_PCD_DIR}")

    print(f"[setup] MODEL_FLAVOR={MODEL_FLAVOR}  MIN_SCORE={MIN_SCORE}  TOP_K={TOP_K}  MAX_RADIUS_M={MAX_RADIUS_M}")
    nusc = NuScenes(version=VERSION, dataroot=DATASET_ROOT, verbose=False)
    rr.init("nuScenes seq (dataset LiDAR + per-frame boxes)", spawn=SPAWN_VIEW)

    frames = 0
    for i, entry in enumerate(preds):
        if MAX_FRAMES is not None and frames >= MAX_FRAMES:
            print(f"[stop] Reached MAX_FRAMES={MAX_FRAMES}")
            break

        # match dataset lidar file
        try:
            sample_token, lidar_path = _find_sd_and_path(nusc, entry)
        except Exception as e:
            print(f"[skip] {os.path.basename(entry)}: {e}")
            continue
        sample = nusc.get("sample", sample_token)
        scene  = nusc.get("scene", sample["scene_token"])["name"]

        set_frame_timeline(i)

        # LiDAR cloud
        pts, cols = _load_dataset_lidar(lidar_path)
        rr.log("world/LIDAR_TOP", rr.Points3D(pts, colors=cols))

        # Boxes for this frame index
        try:
            (centers, sizes, yaws, scores, labels, names, src_key), box_path = _find_boxes_for_frame(entry, i)
            before = len(centers)
            centers, sizes, yaws, scores, labels, names = _apply_filters(centers, sizes, yaws, scores, labels, names)
            after = len(centers)

            if after == 0:
                msg = (f"[boxes] {os.path.basename(entry)} -> {os.path.basename(box_path)} "
                       f"(0 after filters; had {before}, key={src_key})")
                if SKIP_EMPTY_BOX_FRAMES:
                    print("[skip] " + msg)
                    continue
                else:
                    print(msg)
            else:
                how = log_boxes3d("world/preds", centers, sizes, yaws, scores, labels, names)
                s_min = float(scores.min()) if scores is not None and scores.size else float('nan')
                s_max = float(scores.max()) if scores is not None and scores.size else float('nan')
                print(f"[boxes] {os.path.basename(entry)} -> {os.path.basename(box_path)}  "
                      f"({after}/{before} kept, key={src_key}, {how}, scores=({s_min:.3f},{s_max:.3f}))")
        except FileNotFoundError as e:
            print(f"[info] no boxes for {os.path.basename(entry)}: {e}")
        except Exception as e:
            print(f"[warn] box load/draw failed for {os.path.basename(entry)}: {e}")

        # Cameras (toggle)
        if SHOW_CAMERAS:
            for ch in CAM_CHANNELS:
                tok = sample["data"].get(ch)
                if not tok: continue
                try:
                    img_path, _, _ = nusc.get_sample_data(tok)
                    img = np.asarray(Image.open(img_path))
                    rr.log(f"world/{ch}", rr.Image(img))
                except Exception as e:
                    print(f"[warn] camera {ch} failed: {e}")

        print(f"[frame {i:04d}] scene={scene} sample={sample_token} base={_norm_base(entry)}")
        frames += 1

    print(f"[done] streamed {frames} frames.")

if __name__ == "__main__":
    main()
