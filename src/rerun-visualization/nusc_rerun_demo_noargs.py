# nusc_rerun_bridge_multicam.py
# - Creates a local "bridge" dataroot with symlinks (in your home directory)
# - Validates nuScenes layout
# - Streams LiDAR_TOP + ALL six camera images to Rerun
#
# ── Edit these if needed ───────────────────────────────────────────────────────
SRC_ROOT     = "/media/jarvis/Acer/Users/mahes/MY DATA/FYP/Dataset/Train"
BLOBS_DIR    = "v1.0-trainval01_blobs"   # contains samples/ & sweeps/
META_DIR     = "v1.0-trainval_meta"      # contains maps/ & v1.0-trainval/
BRIDGE_ROOT  = "/home/jarvis/nuscenes_train_bridge"  # lives on ext4
VERSION      = "v1.0-trainval"
SCENE_NAME   = None        # e.g., "scene-0061"; None -> use SCENE_INDEX
SCENE_INDEX  = 0
MAX_FRAMES   = 500         # None = full scene; keep small while testing
SPAWN_VIEW   = False       # open the Rerun viewer
SAVE_RRD     = "trainval_multicam.rrd"       # e.g., "trainval_multicam.rrd" to persist, else None
# ───────────────────────────────────────────────────────────────────────────────

import os, sys
from typing import Optional
import numpy as np
from PIL import Image
import rerun as rr
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

CAM_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

join = os.path.join

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def ensure_symlink(dst: str, src: str):
    """Create dst -> src symlink if needed (non-destructive)."""
    if os.path.islink(dst):
        current = os.readlink(dst)
        cur_abs = os.path.abspath(join(os.path.dirname(dst), current))
        if cur_abs == os.path.abspath(src):
            print(f"[ok] link exists: {dst} -> {src}")
            return
        print(f"[warn] {dst} links elsewhere ({current}); leaving as-is.")
        return
    if os.path.exists(dst):
        print(f"[ok] {dst} exists (not a symlink); leaving as-is.")
        return
    if not os.path.exists(src):
        print(f"[error] missing source for link:\n  {src}")
        sys.exit(1)
    os.symlink(src, dst)
    print(f"[link] {dst} -> {src}")

def shallow_print(root: str, title: str):
    print(f"\n[dir] {title} (-L 1)")
    try:
        subdirs = sorted(next(os.walk(root))[1])
    except StopIteration:
        subdirs = []
    print(root)
    for d in subdirs:
        print(f"├── {d}")

def build_bridge():
    print(f"[setup] SRC_ROOT    = {SRC_ROOT}")
    print(f"[setup] BRIDGE_ROOT = {BRIDGE_ROOT}")
    ensure_dir(BRIDGE_ROOT)

    blobs = join(SRC_ROOT, BLOBS_DIR)
    meta  = join(SRC_ROOT, META_DIR)

    # create links inside BRIDGE_ROOT (on ext4) -> sources on external drive
    ensure_symlink(join(BRIDGE_ROOT, "samples"),       join(blobs, "samples"))
    ensure_symlink(join(BRIDGE_ROOT, "sweeps"),        join(blobs, "sweeps"))
    if os.path.isdir(join(meta, "maps")):
        ensure_symlink(join(BRIDGE_ROOT, "maps"),      join(meta, "maps"))
    elif os.path.isdir(join(blobs, "maps")):
        ensure_symlink(join(BRIDGE_ROOT, "maps"),      join(blobs, "maps"))
    else:
        os.makedirs(join(BRIDGE_ROOT, "maps"), exist_ok=True)
        print("[info] created empty maps/ (optional)")

    ensure_symlink(join(BRIDGE_ROOT, VERSION),         join(meta, VERSION))

    # validate required dirs/files
    for d in ("samples", "sweeps", VERSION):
        if not os.path.isdir(join(BRIDGE_ROOT, d)):
            print(f"[error] bridge missing: {d}")
            sys.exit(1)
    for jf in ("scene.json", "sample.json", "sample_data.json"):
        jfpath = join(BRIDGE_ROOT, VERSION, jf)
        if not os.path.isfile(jfpath):
            print(f"[error] missing metadata file: {jfpath}")
            sys.exit(1)

    shallow_print(SRC_ROOT,    "source")
    shallow_print(BRIDGE_ROOT, "bridge (dataroot)")

def pick_scene(nusc: NuScenes, name: Optional[str], idx: int):
    if name:
        for s in nusc.scene:
            if s["name"] == name:
                return s
        raise RuntimeError(f"Scene '{name}' not found.")
    if not (0 <= idx < len(nusc.scene)):
        raise RuntimeError(f"SCENE_INDEX {idx} out of range 0..{len(nusc.scene)-1}")
    return nusc.scene[idx]

def main():
    build_bridge()

    rr.init("nuScenes × Rerun (bridge, multicam)", spawn=SPAWN_VIEW)
    if SAVE_RRD:
        rr.save(SAVE_RRD)

    # Use the bridge as the dataroot so the devkit sees the expected topology
    nusc = NuScenes(version=VERSION, dataroot=BRIDGE_ROOT, verbose=True)
    scene = pick_scene(nusc, SCENE_NAME, SCENE_INDEX)
    print(f"[info] Using scene: {scene['name']}")
    sample = nusc.get("sample", scene["first_sample_token"])

    frame = 0
    done  = 0
    while True:
        if MAX_FRAMES is not None and done >= MAX_FRAMES:
            print(f"[stop] Reached MAX_FRAMES={MAX_FRAMES}")
            break

        rr.set_time_sequence("frame", frame)

        # --- LiDAR_TOP ---
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_path, boxes_lidar, _ = nusc.get_sample_data(lidar_token)
        pc = LidarPointCloud.from_file(lidar_path)    # (4, N): x,y,z,intensity
        pts = pc.points[:3].T
        colors = None
        if pc.points.shape[0] >= 4:
            intens = pc.points[3]
            intens = 255 * (intens - intens.min()) / (max(intens.ptp(), 1e-6))
            colors = np.stack([intens, intens, intens], axis=-1).astype(np.uint8)
        rr.log("world/LIDAR_TOP", rr.Points3D(pts, colors=colors))

        # --- ALL cameras (front/back/sides) ---
        for ch in CAM_CHANNELS:
            token = sample["data"].get(ch)
            if not token:
                continue
            img_path, _, _ = nusc.get_sample_data(token)
            img = np.asarray(Image.open(img_path))
            rr.log(f"world/{ch}", rr.Image(img))

        if not sample["next"]:
            print("[done] End of scene.")
            break
        sample = nusc.get("sample", sample["next"])
        frame += 1
        done  += 1

if __name__ == "__main__":
    main()
