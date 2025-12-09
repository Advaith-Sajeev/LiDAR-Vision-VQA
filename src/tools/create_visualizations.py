#!/usr/bin/env python3
"""Generate LiDAR + camera visualization panels for Modal inference samples.

Automatically pulls the latest run from the Modal volume (using "modal volume get")
unless --skip-download is provided.
"""

import argparse
import json
import shutil
import subprocess
import textwrap
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INFERENCE_ROOT = _REPO_ROOT / "modal_inference"
_DEFAULT_VOLUME_NAME = "lidar-llm"
_REMOTE_CHECKPOINTS_ROOT = PurePosixPath("/checkpoints")

# Question → camera view heuristics
_VIEW_KEYWORDS: List[Tuple[str, str]] = [
    ("front left", "CAM_FRONT_LEFT"),
    ("front right", "CAM_FRONT_RIGHT"),
    ("front view", "CAM_FRONT"),
    ("front", "CAM_FRONT"),
    ("back right", "CAM_BACK_RIGHT"),
    ("back left", "CAM_BACK_LEFT"),
    ("back view", "CAM_BACK"),
    ("back", "CAM_BACK"),
    ("left view", "CAM_BACK_LEFT"),
    ("right view", "CAM_BACK_RIGHT"),
]


def load_lidar_points(lidar_path: Path) -> np.ndarray:
    """Load nuScenes LIDAR_TOP .pcd.bin and return XYZ + intensity."""
    if not lidar_path or not lidar_path.exists():
        return np.empty((0, 4), dtype=np.float32)

    data = np.fromfile(lidar_path, dtype=np.float32)
    if data.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    feature_dim = 5  # nuScenes stores x, y, z, intensity, ring index
    usable = (data.size // feature_dim) * feature_dim
    if usable == 0:
        return np.empty((0, 4), dtype=np.float32)
    data = data[:usable].reshape(-1, feature_dim)
    return data[:, :4].copy()  # x, y, z, intensity


def detect_view(question: str) -> Optional[str]:
    """Return the camera view keyword that best matches the question."""
    question = (question or "").lower()
    for key, view in _VIEW_KEYWORDS:
        if key in question:
            return view
    return None


def find_image(files: List[str], target_view: str, sample_dir: Path) -> Optional[Path]:
    for name in files:
        if target_view in name:
            candidate = sample_dir / name
            if candidate.exists():
                return candidate
    return None


FIG_SIZE = (20.0, 6.0)
_GRID_WIDTH_RATIOS = [2.5, 2.5, 3.0]
TEXT_WRAP_WIDTH = 120
TEXT_FONT_SIZE = 24


def wrap_text(label: str, text: str, width: int = TEXT_WRAP_WIDTH) -> str:
    body = textwrap.fill(text.strip(), width=width)
    return f"{label}:\n{body}"


def render_visual(sample_data: Dict, lidar_points: np.ndarray, image_path: Path, out_path: Path) -> None:
    image = Image.open(image_path)
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 3, wspace=0.05, width_ratios=_GRID_WIDTH_RATIOS)

    # LiDAR BEV column
    ax_lidar = fig.add_subplot(gs[0, 0])
    if lidar_points.size:
        xs, ys = lidar_points[:, 0], lidar_points[:, 1]
        ax_lidar.scatter(xs, ys, c="#ffffff", s=2.5, alpha=1.0)
        ax_lidar.set_xlim(-10, 10)
        ax_lidar.set_ylim(-10, 10)
        ax_lidar.set_aspect("equal", adjustable="box")
    else:
        ax_lidar.text(
            0.5,
            0.5,
            "No LiDAR",
            ha="center",
            va="center",
            fontsize=22,
            color="#f2f2f2",
        )
        ax_lidar.set_xlim(-1, 1)
        ax_lidar.set_ylim(-1, 1)
    ax_lidar.set_facecolor("#000000")
    ax_lidar.set_title("LiDAR BEV", fontsize=24, pad=6)
    ax_lidar.set_xticks([])
    ax_lidar.set_yticks([])
    for spine in ax_lidar.spines.values():
        spine.set_visible(False)

    # Camera image column
    ax_img = fig.add_subplot(gs[0, 1])
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title("Camera View", fontsize=24, pad=6)

    # Text column
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")
    ax_txt.set_facecolor("#f8f9fb")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    question = wrap_text("Question", sample_data.get("question", ""))
    prediction = wrap_text("Prediction", sample_data.get("prediction", ""))
    ground_truth = wrap_text("Ground Truth", sample_data.get("ground_truth", ""))
    text_blocks = [question, "", prediction, "", ground_truth]
    ax_txt.text(
        0.02,
        0.98,
        "\n".join(text_blocks),
        fontsize=TEXT_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        family="DejaVu Sans",
        linespacing=1.2,
    )

    title = f"Sample {sample_data.get('sequence_id', '?'):>04}"
    fig.suptitle(title, fontsize=26, fontweight="bold")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def process_sample(sample_dir: Path, unmatched: List[Dict], out_dir: Path) -> None:
    sample_file = sample_dir / "sample.json"
    if not sample_file.exists():
        unmatched.append(
            {
                "sample_token": sample_dir.name,
                "reason": "missing sample.json",
            }
        )
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        sample = json.load(f)

    question = sample.get("question", "")
    target_view = detect_view(question)
    artifacts = sample.get("artifacts", {})
    image_files = artifacts.get("copied_image_files", [])
    lidar_file = artifacts.get("copied_lidar_file")

    if not target_view:
        unmatched.append(
            {
                "sample_token": sample.get("sample_token"),
                "question": question,
                "reason": "view keyword not detected",
            }
        )
        return

    image_path = find_image(image_files, target_view, sample_dir)
    if image_path is None:
        unmatched.append(
            {
                "sample_token": sample.get("sample_token"),
                "question": question,
                "target_view": target_view,
                "reason": "matching camera image not found",
            }
        )
        return

    lidar_path = sample_dir / lidar_file if lidar_file else None
    if lidar_path is None or not lidar_path.exists():
        unmatched.append(
            {
                "sample_token": sample.get("sample_token"),
                "reason": "LiDAR scan not found",
            }
        )
        return

    lidar_points = load_lidar_points(lidar_path)

    filename = f"{sample.get('sequence_id', 0):04d}_{sample.get('sample_token')}_{target_view}.png"
    out_path = out_dir / filename
    render_visual(sample, lidar_points, image_path, out_path)


def locate_latest_run(inference_root: Path) -> Path:
    if not inference_root.exists():
        raise FileNotFoundError(f"Inference root not found: {inference_root}")

    candidates: List[Tuple[float, Path]] = []
    for artifacts_dir in inference_root.rglob("artifacts"):
        if artifacts_dir.is_dir():
            run_dir = artifacts_dir.parent
            try:
                mtime = run_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            candidates.append((mtime, run_dir))

    if not candidates:
        raise FileNotFoundError(f"No downloaded runs with artifacts/ under {inference_root}")

    latest_mtime, latest_run = max(candidates, key=lambda pair: pair[0])
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_mtime))
    print(f"[viz] Auto-selected latest run at {latest_run} (modified {ts})")
    return latest_run


def _list_dir(volume_name: str, remote_path: PurePosixPath) -> List[Dict]:
    commands = [
        ["modal", "volume", "ls", volume_name, str(remote_path), "--json"],
        ["modal", "volume", "ls", volume_name, str(remote_path)],
    ]

    errors: List[str] = []
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Modal CLI not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            errors.append(exc.stderr.strip() or exc.stdout.strip() or str(exc))
            continue

        entries = _parse_listing_output(result.stdout)
        if entries is not None:
            return entries

        print("[viz] modal volume ls output could not be parsed. Command:")
        print("      $", " ".join(cmd))
        if result.stdout:
            print("[viz] stdout:\n" + result.stdout)
        if result.stderr:
            print("[viz] stderr:\n" + result.stderr)

    raise RuntimeError(
        "Failed to execute Modal volume listing commands. "
        + ("; ".join(errors) if errors else "")
    )


def _parse_listing_output(stdout: str) -> Optional[List[Dict]]:
    stripped = stdout.strip()
    if not stripped:
        return []

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        if isinstance(data, dict) and "files" in data:
            entries = data["files"]
        elif isinstance(data, list):
            entries = data
        else:
            return []
        for entry in entries:
            # Map 'Filename' (from modal ls --json) to 'path'/'name'
            if "Filename" in entry and "path" not in entry:
                entry["path"] = entry["Filename"]
            entry.setdefault("name", entry.get("path", ""))
        return entries

    lines = [line for line in stripped.splitlines() if line.strip()]
    data_rows: List[Dict] = []
    for line in lines:
        if "│" in line:
            cells = [c.strip() for c in line.split("│")[1:-1]]
            if len(cells) >= 2 and cells[0] and cells[0] != "Filename":
                name = cells[0]
                entry_type = cells[1]
                modified = cells[2] if len(cells) >= 3 else ""
                data_rows.append(
                    {
                        "name": name,
                        "path": name,
                        "type": entry_type,
                        "modified": modified,
                        "is_dir": entry_type.lower().startswith("dir"),
                    }
                )
    if data_rows:
        return data_rows

    # Fallback: plain text without table
    entries: List[Dict] = []
    for line in lines:
        if line.lower().startswith("directory listing"):
            continue
        tokens = line.split()
        if not tokens:
            continue
        name = tokens[-1]
        entries.append({"name": name, "path": name, "is_dir": True})
    return entries if entries else []


def _entry_is_dir(entry: Dict) -> bool:
    if entry.get("is_dir") is not None:
        return bool(entry["is_dir"])
    entry_type = entry.get("type")
    if entry_type:
        return entry_type.lower().startswith("dir")
    return False


def _entry_timestamp(entry: Dict, fallback_name: str) -> float:
    for key in ("mtime", "modified", "last_modified", "timestamp", "Created/Modified"):
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        ts = _parse_modified_string(str(value))
        if ts is not None:
            return ts
    try:
        return datetime.strptime(fallback_name, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        # Try to extract timestamp from run_YYYYMMDD_HHMMSS format if possible
        if "run_" in fallback_name:
            try:
                part = fallback_name.split("run_")[1]
                return datetime.strptime(part[:15], "%Y%m%d_%H%M%S").timestamp()
            except (ValueError, IndexError):
                pass
        return 0.0


def _parse_modified_string(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M %Z",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y%m%d_%H%M%S",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    # Attempt to drop trailing timezone words
    if " " in value:
        trimmed = value.rsplit(" ", 1)[0]
        try:
            dt = datetime.strptime(trimmed, "%Y-%m-%d %H:%M")
            return dt.timestamp()
        except ValueError:
            pass
    return None


def _derive_local_subpath(remote_posix: PurePosixPath) -> Path:
    parts = list(remote_posix.parts)
    for idx, part in enumerate(parts):
        if part.startswith("run_"):
            return Path(*parts[idx:])
    if "modal_inference" in parts:
        idx = parts.index("modal_inference")
        return Path(*parts[idx:])
    return Path(remote_posix.name)


def _drill_down_to_artifacts(volume_name: str, root: PurePosixPath) -> PurePosixPath:
    """
    Recursively search for a folder named 'artifacts' starting from 'root'.
    Prioritizes 'modal_inference' and recent timestamps.
    """
    print(f"[viz] Drilling down from {root} to find 'artifacts'...")
    queue = [(root, 0)]
    max_depth = 4
    visited = set()

    while queue:
        curr, depth = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)

        if depth > max_depth:
            continue

        try:
            entries = _list_dir(volume_name, curr)
        except Exception as e:
            print(f"[viz] Warning: Error listing {curr}: {e}")
            continue

        # Check if 'artifacts' is here
        for e in entries:
            if e.get('name') == 'artifacts' or e.get('path', '').endswith('/artifacts'):
                print(f"[viz] Found artifacts folder at {curr}")
                return curr

        # Collect subdirectories
        subdirs = []
        for e in entries:
            name = e.get('name')
            if not name:
                continue
            
            # Construct full path
            raw_path = e.get("path") or name
            if raw_path.startswith("/"):
                full_path = PurePosixPath(raw_path)
            else:
                # Handle relative paths carefully
                curr_rel = str(curr).lstrip("/")
                if curr_rel and raw_path.startswith(curr_rel + "/"):
                    full_path = PurePosixPath("/") / raw_path
                else:
                    full_path = curr / raw_path

            # Heuristic: if it looks like a file (has extension), skip unless we know it's a dir
            if not _entry_is_dir(e) and "." in full_path.name and not full_path.name.startswith("run_"):
                continue

            subdirs.append((full_path, _entry_timestamp(e, full_path.name)))

        # Sort: prioritize 'modal_inference', then by timestamp desc
        def sort_key(item):
            path, ts = item
            if path.name == 'modal_inference':
                return (2, ts)
            if path.name.startswith("20"): # Timestamp folder?
                return (1, ts)
            return (0, ts)

        subdirs.sort(key=sort_key, reverse=True)

        for path, _ in subdirs:
            queue.append((path, depth + 1))

    print(f"[viz] 'artifacts' folder not found under {root}. Using root.")
    return root


def _find_latest_run_path(
    volume_name: str,
    remote_root: PurePosixPath,
) -> Optional[PurePosixPath]:
    """
    Find the latest 'run_*' directory under remote_root.
    Performs a shallow BFS (depth=1) to find run directories even if they are nested.
    """
    print(f"[viz] Searching for runs in {remote_root} (volume: {volume_name})...")
    queue = [(PurePosixPath(remote_root), 0)]
    visited = set()
    candidates: List[Tuple[float, PurePosixPath]] = []

    # Safety limits
    max_depth = 1
    max_dirs_to_list = 10
    listed_count = 0

    while queue:
        current_path, depth = queue.pop(0)
        if current_path in visited:
            continue
        visited.add(current_path)

        try:
            entries = _list_dir(volume_name, current_path)
        except RuntimeError as exc:
            print(f"[viz] Warning: Unable to list {current_path}: {exc}")
            continue

        subdirs: List[PurePosixPath] = []

        for entry in entries:
            # If we can't determine if it's a dir, assume it is if it looks like a run
            is_dir = _entry_is_dir(entry)
            
            raw_path = entry.get("path") or entry.get("name") or ""
            if not raw_path:
                continue

            # Handle potential full paths returned by ls
            if raw_path.startswith("/"):
                full_path = PurePosixPath(raw_path)
            else:
                # Modal often returns paths relative to volume root (e.g. "checkpoints/run_...")
                # If current_path is "/checkpoints", we don't want "/checkpoints/checkpoints/run_..."
                # We check if raw_path seems to include the current path prefix.
                
                # Remove leading slash from current_path for comparison
                curr_rel = str(current_path).lstrip("/")
                if curr_rel and raw_path.startswith(curr_rel + "/"):
                    full_path = PurePosixPath("/") / raw_path
                else:
                    full_path = current_path / raw_path

            leaf = full_path.name

            if leaf.startswith("run_"):
                ts = _entry_timestamp(entry, leaf)
                candidates.append((ts, full_path))
            elif (is_dir or not leaf.startswith(".")) and depth < max_depth:
                # If it's a directory (or we're not sure but it's not hidden), check it
                subdirs.append(full_path)

        if depth < max_depth and listed_count < max_dirs_to_list:
            for sd in subdirs:
                queue.append((sd, depth + 1))
        
        listed_count += 1

    if not candidates:
        print(f"[viz] No 'run_*' directories found under {remote_root}")
        return None

    # Sort by timestamp descending
    candidates.sort(key=lambda item: item[0])
    latest_ts, latest_path = candidates[-1]

    ts_str = datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[viz] Found latest run: {latest_path} (modified {ts_str})")

    # Drill down to find artifacts
    return _drill_down_to_artifacts(volume_name, latest_path)


def _resolve_modal_inference_root(remote_root: PurePosixPath) -> PurePosixPath:
    parts = list(remote_root.parts)
    for idx, part in enumerate(parts):
        if part == "modal_inference":
            return PurePosixPath(*parts[: idx + 1])
    return remote_root / "modal_inference"


def download_modal_tree(
    *,
    volume_name: str,
    remote_root: PurePosixPath,
    inference_root: Path,
) -> Path:
    remote_modal_root = _resolve_modal_inference_root(remote_root)
    target_parent = inference_root.parent
    target_parent.mkdir(parents=True, exist_ok=True)

    print(f"[viz] Downloading entire modal_inference from {remote_modal_root} → {target_parent}")
    cmd = ["modal", "volume", "get", volume_name, str(remote_modal_root), str(target_parent), "--force"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"modal volume get failed for {remote_modal_root}. "
            "Set --remote-root to the run_* directory that contains modal_inference."
        ) from exc

    downloaded_dir = target_parent / remote_modal_root.name
    if not downloaded_dir.exists():
        raise FileNotFoundError(f"modal volume get did not produce {downloaded_dir}")

    if downloaded_dir != inference_root:
        if inference_root.exists():
            shutil.rmtree(inference_root)
        downloaded_dir.rename(inference_root)

    return locate_latest_run(inference_root)


def _download_remote_run(volume_name: str, remote_path: str, local_root: Path) -> Path:
    local_root.mkdir(parents=True, exist_ok=True)
    remote_posix = PurePosixPath(remote_path)

    print(f"[viz] Downloading {remote_path} → {local_root}")
    cmd = [
        "modal",
        "volume",
        "get",
        volume_name,
        remote_path,
        str(local_root),
        "--force",
    ]
    subprocess.run(cmd, check=True)
    leaf = remote_posix.name
    downloaded_dir = local_root / leaf
    target_rel = _derive_local_subpath(remote_posix)
    target_dir = local_root / target_rel

    if downloaded_dir.exists() and downloaded_dir != target_dir:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        downloaded_dir.rename(target_dir)
    elif not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    if (target_dir / "artifacts").exists():
        return target_dir

    try:
        latest = locate_latest_run(local_root)
        return latest
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Downloaded path {target_dir} does not contain artifacts/. Verify the remote path."
        )


def download_latest_modal_run(
    *,
    volume_name: str,
    remote_root: PurePosixPath,
    local_root: Path,
    remote_path_override: Optional[str] = None,
) -> Optional[Path]:
    candidate_paths: List[str] = []
    if remote_path_override:
        candidate_paths.append(remote_path_override)

    if not candidate_paths:
        latest_remote = _find_latest_run_path(volume_name, remote_root)
        if latest_remote is None:
            print(
                "[viz] Unable to discover run folders automatically. "
                "Pass --remote-path or use --download-modal-tree when the Modal CLI "
                "cannot list directories."
            )
        else:
            candidate_paths.append(str(latest_remote))

    if not candidate_paths:
        print("[viz] No modal_inference directories found on the volume.")
        return None

    for remote_path in candidate_paths:
        try:
            return _download_remote_run(volume_name, remote_path, local_root)
        except subprocess.CalledProcessError as exc:
            print(f"[viz] modal volume get failed for {remote_path} (exit {exc.returncode}).")
        except Exception as exc:
            print(f"[viz] Failed to download {remote_path}: {exc}")
    return None


def generate_visualizations(run_dir: Path) -> None:
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    unmatched: List[Dict] = []
    sample_dirs = sorted(p for p in artifacts_dir.iterdir() if p.is_dir())

    for sample_dir in sample_dirs:
        process_sample(sample_dir, unmatched, viz_dir)

    if unmatched:
        unmatched_path = viz_dir / "unmatched_samples.json"
        with open(unmatched_path, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, indent=2)
        print(f"[viz] {len(unmatched)} samples need manual inspection -> {unmatched_path}")
    else:
        print("[viz] All samples processed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visualization panels for inference samples")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Optional path to a specific run directory (defaults to latest downloaded run)",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=_DEFAULT_INFERENCE_ROOT,
        help=f"Root directory where modal runs are downloaded (default: {_DEFAULT_INFERENCE_ROOT})",
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default=_DEFAULT_VOLUME_NAME,
        help=f"Modal volume name to sync from (default: {_DEFAULT_VOLUME_NAME})",
    )
    parser.add_argument(
        "--remote-root",
        type=str,
        default=str(_REMOTE_CHECKPOINTS_ROOT),
        help="Remote root inside the volume that contains checkpoints (default: /checkpoints)",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="",
        help="Explicit remote modal_inference path to download (overrides auto-discovery)",
    )
    parser.add_argument(
        "--download-modal-tree",
        action="store_true",
        help="Download the entire modal_inference directory under --remote-root",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip auto-downloading the latest Modal run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_root = args.inference_root.expanduser()
    if not inference_root.is_absolute():
        inference_root = (Path.cwd() / inference_root).resolve()
    else:
        inference_root = inference_root.resolve()
    inference_root.mkdir(parents=True, exist_ok=True)

    remote_root = PurePosixPath(args.remote_root)
    downloaded_run: Optional[Path] = None
    modal_tree_synced = False
    if not args.skip_download:
        if args.download_modal_tree:
            try:
                downloaded_run = download_modal_tree(
                    volume_name=args.volume_name,
                    remote_root=remote_root,
                    inference_root=inference_root,
                )
                modal_tree_synced = True
            except Exception as exc:
                print(f"[viz] Failed to download modal_inference tree: {exc}")
                print(
                    "[viz] Ensure --remote-root points to the specific run_* directory "
                    "that owns modal_inference (e.g. /checkpoints/run_20251203_010659)."
                )
                raise SystemExit(1)
        else:
            downloaded_run = download_latest_modal_run(
                volume_name=args.volume_name,
                remote_root=remote_root,
                local_root=inference_root,
                remote_path_override=args.remote_path or None,
            )
            if downloaded_run is None and not args.remote_path:
                print(
                    "[viz] Falling back to downloading the entire modal_inference directory "
                    "because automatic run discovery failed."
                )
                try:
                    downloaded_run = download_modal_tree(
                        volume_name=args.volume_name,
                        remote_root=remote_root,
                        inference_root=inference_root,
                    )
                    modal_tree_synced = True
                except Exception as exc:
                    print(f"[viz] Fallback download failed: {exc}")
                    print(
                        "[viz] Provide --remote-root /checkpoints/run_YYYYMMDD_HHMMSS so the tool knows "
                        "which run folder to sync."
                    )
                    raise SystemExit(1)

    if args.run_dir:
        run_dir = args.run_dir.expanduser()
        if run_dir.is_absolute():
            candidates = [run_dir]
        else:
            candidates = [inference_root / run_dir, Path.cwd() / run_dir]

        for candidate in candidates:
            if candidate.exists():
                run_directory = candidate.resolve()
                break
        else:
            run_directory = candidates[0].resolve()
    elif downloaded_run is not None:
        run_directory = downloaded_run
    else:
        try:
            run_directory = locate_latest_run(inference_root)
        except FileNotFoundError as exc:
            print(f"[viz] {exc}")
            print(
                "[viz] Provide --remote-path with a full modal_inference/run_* folder or "
                "use --run-dir to point at an already-downloaded run."
            )
            print(
                f"[viz] Example: --remote-path {remote_root}/modal_inference/run_20251208_031041"
            )
            if not modal_tree_synced:
                print(
                    "[viz] Alternatively, pass --download-modal-tree to sync all remote runs at once."
                )
            raise SystemExit(1)

    generate_visualizations(run_directory)


if __name__ == "__main__":
    main()
