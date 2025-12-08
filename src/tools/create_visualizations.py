#!/usr/bin/env python3
"""Generate side-by-side panels for modal inference samples.

modal volume get lidar-llm /checkpoints/run_20251203_010659/modal_inference/XXXX ./modal_inference/XXXX

"""

import argparse
import json
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INFERENCE_ROOT = _REPO_ROOT / "modal_inference"

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


FIG_SIZE = (11.7, 5.8)  # A4 landscape width with reduced height
TEXT_WRAP_WIDTH = 90
TEXT_FONT_SIZE = 12


def wrap_text(label: str, text: str, width: int = TEXT_WRAP_WIDTH) -> str:
    body = textwrap.fill(text.strip(), width=width)
    return f"{label}:\n{body}"


def render_visual(sample_data: Dict, image_path: Path, out_path: Path) -> None:
    image = Image.open(image_path)
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2, wspace=0.04, width_ratios=[3, 2.5])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.axis("off")

    ax_txt = fig.add_subplot(gs[0, 1])
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
        linespacing=1.25,
    )

    footer = sample_data.get("metrics", {})
    footer_text = (
        f"F1: {footer.get('f1', 0):.3f}  ·  "
        f"Precision: {footer.get('precision', 0):.3f}  ·  "
        f"Recall: {footer.get('recall', 0):.3f}"
    )
    ax_txt.text(0.02, 0.07, footer_text, fontsize=11, fontweight="bold", color="#223", ha="left")

    title = f"Sample {sample_data.get('sequence_id', '?'):>04}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

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
    image_files = sample.get("artifacts", {}).get("copied_image_files", [])

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

    filename = f"{sample.get('sequence_id', 0):04d}_{sample.get('sample_token')}_{target_view}.png"
    out_path = out_dir / filename
    render_visual(sample, image_path, out_path)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_root = args.inference_root.expanduser()
    if not inference_root.is_absolute():
        inference_root = (Path.cwd() / inference_root).resolve()
    else:
        inference_root = inference_root.resolve()

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
    else:
        run_directory = locate_latest_run(inference_root)

    generate_visualizations(run_directory)


if __name__ == "__main__":
    main()
