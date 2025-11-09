#!/usr/bin/env python3
import os, json, shutil
from pathlib import Path
from collections import defaultdict

# =========================== CONFIG ===========================
# How many images you want in EACH camera folder (CAM_FRONT, CAM_BACK, etc.)
PER_CHANNEL_GOAL = 31000

# nuScenes has 6 cams, 1 lidar, 5 radars → scale targets by 6/1/5
# (Official suite: 6 cameras, 1 LiDAR, 5 RADAR; standard folders: samples/, sweeps/.) 
# Sources: nuScenes site & docs. :contentReference[oaicite:0]{index=0}
TARGETS = {
    "CAM":   6 * PER_CHANNEL_GOAL,   # total across all 6 CAM_* channels
    "LIDAR": 1 * PER_CHANNEL_GOAL,   # LIDAR_TOP
    "RADAR": 5 * PER_CHANNEL_GOAL,   # all RADAR_* channels combined
}

CONFIG = {
    # Required paths
    "DATAROOT": "/home/j_bindu/fyp-26-grp-38/Datasets/nuscenes/train",  # root with samples/, maps/, v1.0-*/
    "VERSION":  "v1.0-trainval",                                        # e.g., v1.0-trainval / v1.0-mini / v1.0-test
    "OUTROOT":  "/home/j_bindu/fyp-26-grp-38/Dataset_subset",           # where to create the subset

    # Targets (keyframes only, across all channels in each modality)
    "TARGETS": TARGETS,

    # Materialization behavior
    "INCLUDE_SWEEPS": False,   # also include non-keyframe sweeps in the subset
    "COPY_FILES": False,       # False = symlink, True = copy files

    # Force-drop these scene tokens first (optional)
    "ALWAYS_DROP_SCENES": [
        # "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    ],

    # External datasets (optional). JSON-array or JSONL accepted.
    "NUGROUNDING_PATH": "/home/j_bindu/fyp-26-grp-38/Datasets/LiDAR-LLM-Nu-Grounding/LiDAR-LLM-Nu-Grounding-train.json",  # e.g. "/data/sets/nuGrounding/nuGrounding.jsonl"
    "NUCAPTION_PATH":   "/home/j_bindu/fyp-26-grp-38/Datasets/LiDAR-LLM-Nu-Caption/train.json",  # e.g. "/data/sets/nuCaption/nuCaption.json"
    # Where filtered external files will go relative to OUTROOT
    "EXTERNAL_SUBDIR":  "external",
}

# =============================================================

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _dump_json(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

def _read_json_or_jsonl(path):
    if path is None or not os.path.exists(path):
        return [], None
    with open(path, "r") as f:
        head = f.read(2048)
        f.seek(0)
        if head.strip().startswith("["):
            return json.load(f), "json"
        return [json.loads(line) for line in f if line.strip()], "jsonl"

def _write_json_or_jsonl(path, records, fmt):
    if not path or not fmt:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
    else:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

def _by_token(rows):
    return {r["token"]: r for r in rows}

# ---------- Option A: derive channel from path ----------
def channel_from_filename(sd):
    """Extract channel from filename path segment after 'samples/' or 'sweeps/'."""
    fn = (sd.get("filename") or "").replace("\\", "/")
    parts = fn.split("/")
    for i, p in enumerate(parts):
        if p in ("samples", "sweeps") and i + 1 < len(parts):
            return parts[i + 1]  # e.g., CAM_FRONT, LIDAR_TOP, RADAR_FRONT
    return None

def modality_of(channel):
    """Map channel name to modality bucket."""
    if not channel:
        return None
    ch = str(channel).upper()
    if ch == "LIDAR_TOP":       return "LIDAR"
    if ch.startswith("CAM_"):   return "CAM"
    if ch.startswith("RADAR_"): return "RADAR"
    return None

def is_keyframe(sd):
    """Keyframe if flag says so; otherwise if path contains 'samples/' (per nuScenes layout)."""
    if "is_key_frame" in sd:
        return bool(sd["is_key_frame"])
    fn = (sd.get("filename") or "").replace("\\", "/")
    return "samples/" in fn
# --------------------------------------------------------

def main():
    DATAROOT = CONFIG["DATAROOT"]
    VERSION  = CONFIG["VERSION"]
    OUTROOT  = CONFIG["OUTROOT"]

    vpath = os.path.join(DATAROOT, VERSION)
    assert os.path.exists(vpath), f"Missing version folder: {vpath}"

    # ---- Load nuScenes base tables
    scene = _load_json(os.path.join(vpath, "scene.json"))
    sample = _load_json(os.path.join(vpath, "sample.json"))
    sample_data = _load_json(os.path.join(vpath, "sample_data.json"))
    sample_annotation_path = os.path.join(vpath, "sample_annotation.json")
    sample_annotation = _load_json(sample_annotation_path) if os.path.exists(sample_annotation_path) else []
    calibrated_sensor = _load_json(os.path.join(vpath, "calibrated_sensor.json"))
    ego_pose = _load_json(os.path.join(vpath, "ego_pose.json"))
    sensor = _load_json(os.path.join(vpath, "sensor.json"))
    logtbl = _load_json(os.path.join(vpath, "log.json"))
    map_path = os.path.join(vpath, "map.json")
    maptbl = _load_json(map_path) if os.path.exists(map_path) else []

    scene_by_tok  = _by_token(scene)
    sample_by_tok = _by_token(sample)
    sd_by_tok     = _by_token(sample_data)

    # ---- Map sample -> scene by walking scene sample chains
    sample_to_scene = {}
    for sc in scene:
        tok = sc["first_sample_token"]
        while tok:
            s = sample_by_tok[tok]
            sample_to_scene[s["token"]] = sc["token"]
            tok = s["next"] if s["next"] != "" else None

    # ---- Count keyframe sample_data per modality per scene (Option A channel resolver)
    scene_counts = defaultdict(lambda: {"CAM":0,"LIDAR":0,"RADAR":0})
    totals = {"CAM":0,"LIDAR":0,"RADAR":0}

    for sd in sample_data:
        if not is_keyframe(sd):
            continue  # only keyframes towards target
        ch_name = channel_from_filename(sd)  # <-- Option A
        mod = modality_of(ch_name)
        if mod is None:
            continue
        sc_tok = sample_to_scene.get(sd["sample_token"])
        if not sc_tok:
            continue
        scene_counts[sc_tok][mod] += 1
        totals[mod] += 1

    print("Initial keyframe totals (via sample_data + path channels):", totals)

    TARGETS = CONFIG["TARGETS"]
    keep_scenes = set(s["token"] for s in scene)
    drop_scenes = set()

    def over(mod):
        return max(0, totals[mod] - TARGETS[mod])

    def score(sc_tok):
        # Higher score if a scene reduces modalities that are currently over target
        return sum(scene_counts[sc_tok][m] for m in ("CAM","LIDAR","RADAR") if over(m) > 0)

    # 1) Force-drop blacklist
    for sc_tok in list(keep_scenes):
        if sc_tok in set(CONFIG["ALWAYS_DROP_SCENES"]):
            keep_scenes.remove(sc_tok); drop_scenes.add(sc_tok)
            for m in ("CAM","LIDAR","RADAR"):
                totals[m] -= scene_counts[sc_tok][m]

    # 2) Greedy drop until all modalities are within targets
    while any(totals[m] > TARGETS[m] for m in ("CAM","LIDAR","RADAR")):
        best, best_score = None, -1
        for sc_tok in keep_scenes:
            sc_ = score(sc_tok)
            if sc_ > best_score:
                best, best_score = sc_tok, sc_
        if best is None or best_score <= 0:
            print("No more helpful scenes to drop; stopping.")
            break
        keep_scenes.remove(best); drop_scenes.add(best)
        for m in ("CAM","LIDAR","RADAR"):
            totals[m] -= scene_counts[best][m]

    print("Planned keyframe totals:", totals)
    print(f"Keeping {len(keep_scenes)} scenes; Dropping {len(drop_scenes)} scenes.")

    # ---- Build kept tokens (samples, then sample_data)
    keep_samples = set()
    for sc_tok in keep_scenes:
        sc = scene_by_tok[sc_tok]
        tok = sc["first_sample_token"]
        while tok:
            s = sample_by_tok[tok]
            keep_samples.add(s["token"])
            tok = s["next"] if s["next"] != "" else None

    keep_sd = set()
    include_sweeps = CONFIG["INCLUDE_SWEEPS"]
    for sd in sample_data:
        if sd["sample_token"] in keep_samples:
            if is_keyframe(sd) or include_sweeps:
                keep_sd.add(sd["token"])

    # ---- Filter annotations and supporting tables
    keep_ann = [ann for ann in sample_annotation if ann["sample_token"] in keep_samples]
    sd_f     = [sd_by_tok[t] for t in keep_sd]
    scene_f  = [scene_by_tok[t] for t in keep_scenes]
    sample_f = [sample_by_tok[t] for t in keep_samples]

    keep_ego   = set(sd["ego_pose_token"] for sd in sd_f)
    keep_cal   = set(sd["calibrated_sensor_token"] for sd in sd_f)
    ego_pose_f = [e for e in ego_pose if e["token"] in keep_ego]
    cal_f      = [c for c in calibrated_sensor if c["token"] in keep_cal]
    keep_sensor = set(c["sensor_token"] for c in cal_f)
    sensor_f    = [s for s in sensor if s["token"] in keep_sensor]
    keep_logs   = set(sc["log_token"] for sc in scene_f)

    # Map selection: filter maps by kept logs if map.json has log_tokens; else keep all maps.
    def map_is_referenced(m):
        lt = m.get("log_tokens")
        return True if lt is None else any(t in keep_logs for t in lt)
    map_f = [m for m in maptbl if map_is_referenced(m)]

    # ---- Write filtered nuScenes JSONs
    out_ver = os.path.join(OUTROOT, VERSION)
    Path(out_ver).mkdir(parents=True, exist_ok=True)
    _dump_json(os.path.join(out_ver, "scene.json"), scene_f)
    _dump_json(os.path.join(out_ver, "sample.json"), sample_f)
    _dump_json(os.path.join(out_ver, "sample_data.json"), sd_f)
    _dump_json(os.path.join(out_ver, "sample_annotation.json"), keep_ann)
    _dump_json(os.path.join(out_ver, "calibrated_sensor.json"), cal_f)
    _dump_json(os.path.join(out_ver, "ego_pose.json"), ego_pose_f)
    _dump_json(os.path.join(out_ver, "sensor.json"), sensor_f)
    _dump_json(os.path.join(out_ver, "log.json"),
               list({l["token"]: l for l in (log for log in logtbl if log["token"] in keep_logs)}.values()))
    if maptbl:
        _dump_json(os.path.join(out_ver, "map.json"), map_f)
    print("Filtered nuScenes tables written to:", out_ver)

    # ---- Materialize files (keyframes, plus sweeps if requested)
    def materialize(sd_rows):
        for sd in sd_rows:
            rel = sd["filename"]  # e.g., "samples/CAM_FRONT/xxx.jpg" or "sweeps/LIDAR_TOP/xxx.pcd.bin"
            src = os.path.join(DATAROOT, rel)
            dst = os.path.join(OUTROOT, rel)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            if CONFIG["COPY_FILES"]:
                shutil.copy2(src, dst)
            else:
                try:
                    if os.path.lexists(dst): os.remove(dst)
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

    print("Materializing files ...")
    materialize(sd_f)
    print("Files materialized under:", OUTROOT)

    # ---- Filter nuGrounding / nuCaption by kept sample_token (optional)
    def token_of(rec):
        for k in ("sample_token","sampleToken","sample"):
            if k in rec: return rec[k]
        return None

    ng_path = CONFIG["NUGROUNDING_PATH"]
    nc_path = CONFIG["NUCAPTION_PATH"]
    ext_dir = os.path.join(OUTROOT, CONFIG["EXTERNAL_SUBDIR"])

    ng_records, ng_fmt = _read_json_or_jsonl(ng_path)
    nc_records, nc_fmt = _read_json_or_jsonl(nc_path)

    if ng_fmt:
        ng_keep = [r for r in ng_records if token_of(r) in keep_samples]
        Path(ext_dir).mkdir(parents=True, exist_ok=True)
        ng_out = os.path.join(ext_dir, f"nuGrounding.{ 'jsonl' if ng_fmt=='jsonl' else 'json'}")
        _write_json_or_jsonl(ng_out, ng_keep, ng_fmt)
        print(f"Filtered nuGrounding: kept {len(ng_keep)}/{len(ng_records)} → {ng_out}")

    if nc_fmt:
        nc_keep = [r for r in nc_records if token_of(r) in keep_samples]
        Path(ext_dir).mkdir(parents=True, exist_ok=True)
        nc_out = os.path.join(ext_dir, f"nuCaption.{ 'jsonl' if nc_fmt=='jsonl' else 'json'}")
        _write_json_or_jsonl(nc_out, nc_keep, nc_fmt)
        print(f"Filtered nuCaption: kept {len(nc_keep)}/{len(nc_records)} → {nc_out}")

    # ---- Save a manifest
    manifest = {
        "version": VERSION,
        "targets": CONFIG["TARGETS"],
        "kept_scenes": sorted(list(keep_scenes)),
        "dropped_scenes": sorted(list(drop_scenes)),
        "kept_samples_count": len(keep_samples),
        "final_keyframe_totals": totals,
        "include_sweeps": CONFIG["INCLUDE_SWEEPS"],
    }
    _dump_json(os.path.join(OUTROOT, "subset_manifest.json"), manifest)
    print("Wrote manifest:", os.path.join(OUTROOT, "subset_manifest.json"))

if __name__ == "__main__":
    main()
