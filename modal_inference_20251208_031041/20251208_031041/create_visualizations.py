import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
VIZ_DIR = ROOT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

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


def wrap_text(label: str, text: str, width: int = 120) -> str:
    body = textwrap.fill(text.strip(), width=width)
    return f"{label}:\n{body}"


def render_visual(sample_data: Dict, image_path: Path, out_path: Path) -> None:
    image = Image.open(image_path)
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.05, width_ratios=[3, 2])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.axis("off")

    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.axis("off")
    ax_txt.set_facecolor("#f8f9fb")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    question = wrap_text("Question", sample_data.get("question", ""), width=60)
    prediction = wrap_text("Prediction", sample_data.get("prediction", ""), width=60)
    ground_truth = wrap_text("Ground Truth", sample_data.get("ground_truth", ""), width=60)

    text_blocks = [question, "", prediction, "", ground_truth]
    ax_txt.text(
        0.02,
        0.98,
        "\n".join(text_blocks),
        fontsize=10,
        va="top",
        ha="left",
        family="DejaVu Sans",
    )

    footer = sample_data.get("metrics", {})
    footer_text = (
        f"F1: {footer.get('f1', 0):.3f}  ·  "
        f"Precision: {footer.get('precision', 0):.3f}  ·  "
        f"Recall: {footer.get('recall', 0):.3f}"
    )
    ax_txt.text(0.02, 0.05, footer_text, fontsize=9, color="#445", ha="left")

    title = f"Sample {sample_data.get('sequence_id', '?'):>04} · {sample_data.get('dataset_type', 'unknown')}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_sample(sample_dir: Path, unmatched: List[Dict]) -> None:
    sample_file = sample_dir / "sample.json"
    if not sample_file.exists():
        unmatched.append({
            "sample_token": sample_dir.name,
            "reason": "missing sample.json",
        })
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        sample = json.load(f)

    question = sample.get("question", "")
    target_view = detect_view(question)
    image_files = sample.get("artifacts", {}).get("copied_image_files", [])

    if not target_view:
        unmatched.append({
            "sample_token": sample.get("sample_token"),
            "question": question,
            "reason": "view keyword not detected",
        })
        return

    image_path = find_image(image_files, target_view, sample_dir)
    if image_path is None:
        unmatched.append({
            "sample_token": sample.get("sample_token"),
            "question": question,
            "target_view": target_view,
            "reason": "matching camera image not found",
        })
        return

    filename = f"{sample.get('sequence_id', 0):04d}_{sample.get('sample_token')}_{target_view}.png"
    out_path = VIZ_DIR / filename
    render_visual(sample, image_path, out_path)


def main() -> None:
    unmatched: List[Dict] = []
    sample_dirs = sorted(p for p in ARTIFACTS_DIR.iterdir() if p.is_dir())

    for sample_dir in sample_dirs:
        process_sample(sample_dir, unmatched)

    if unmatched:
        unmatched_path = VIZ_DIR / "unmatched_samples.json"
        with open(unmatched_path, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, indent=2)
        print(f"[viz] {len(unmatched)} samples need manual inspection -> {unmatched_path}")
    else:
        print("[viz] All samples processed successfully")


if __name__ == "__main__":
    main()
