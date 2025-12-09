"""
Utility functions for inference
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import torch

from configs.default_config import DEFAULT_VIEW_ORDER
from deepencoder.deepencoder_infer import resolve_cam_image_paths

if TYPE_CHECKING:
    from inference.inference_engine import InferenceEngine


def load_bev_feature(feature_path: Union[str, Path]) -> torch.Tensor:
    """
    Load BEV feature from .npy file.
    
    Args:
        feature_path: Path to .npy file
        
    Returns:
        BEV tensor [C, H, W]
    """
    bev = np.load(feature_path, allow_pickle=False)
    return torch.from_numpy(bev).float()


def format_prompt(
    question: str,
    use_vision: bool = True,
    use_lidar: bool = True,
    system_prompt: str = ""
) -> str:
    """
    Format a question into a prompt with special tokens.
    
    Args:
        question: User question
        use_vision: Whether to include vision tokens
        use_lidar: Whether to include LiDAR tokens
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    # Add system prompt if provided
    if system_prompt:
        parts.append(system_prompt)
        parts.append("\n\n")
    
    # Add vision tokens if enabled
    if use_vision:
        parts.append("<vision_start><vision_end>")
    
    # Add LiDAR tokens if enabled
    if use_lidar:
        parts.append("<lidar_start><lidar_end>")
    
    # Add question and answer prompt
    parts.append(question)
    parts.append("\nAnswer:")
    
    return "".join(parts)


def load_qa_pairs(json_path: Union[str, Path]) -> List[Dict]:
    """
    Load question-answer pairs from JSON/JSONL file.
    
    Args:
        json_path: Path to JSON or JSONL file
        
    Returns:
        List of dictionaries with QA pairs
    """
    json_path = Path(json_path)
    
    if json_path.suffix == ".jsonl":
        qa_pairs = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
        return qa_pairs
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                return [data]


def save_predictions(
    predictions: List[Dict],
    output_path: Union[str, Path],
    format: str = "json"
):
    """
    Save predictions to file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Output file path
        format: Output format ('json' or 'jsonl')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)
    
    print(f"[utils] Saved {len(predictions)} predictions to {output_path}")


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate basic metrics from predictions.
    
    Args:
        predictions: List of prediction dictionaries with 'prediction' and 'ground_truth' keys
        
    Returns:
        Dictionary of metrics
    """
    if not predictions:
        return {}
    
    # Calculate average token lengths
    pred_lengths = [len(p["prediction"].split()) for p in predictions]
    gt_lengths = [len(p.get("ground_truth", "").split()) for p in predictions if "ground_truth" in p]
    
    metrics = {
        "num_samples": len(predictions),
        "avg_prediction_length": sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0,
    }
    
    if gt_lengths:
        metrics["avg_ground_truth_length"] = sum(gt_lengths) / len(gt_lengths)
    
    return metrics


def format_output(
    question: str,
    prediction: str,
    ground_truth: str = None,
    sample_token: str = None,
    width: int = 80
) -> str:
    """
    Format prediction output for display.
    
    Args:
        question: Input question
        prediction: Model prediction
        ground_truth: Ground truth answer (optional)
        sample_token: Sample token (optional)
        width: Display width
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * width)
    
    if sample_token:
        lines.append(f"Sample: {sample_token}")
    
    lines.append(f"\nQuestion: {question}")
    lines.append(f"\nPrediction: {prediction}")
    
    if ground_truth:
        lines.append(f"\nGround Truth: {ground_truth}")
    
    lines.append("=" * width)
    
    return "\n".join(lines)


def _sanitize_folder_name(value: str) -> str:
    """Convert arbitrary strings to filesystem-friendly folder names."""
    if not value:
        return "sample"
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def save_inference_artifacts(
    *,
    artifact_dir: Optional[str],
    sample_payload: Dict[str, Any],
    bev_path: Optional[str],
    engine: "InferenceEngine",
    sequence_id: Optional[int] = None,
    vision_requested: bool = True,
) -> Optional[Path]:
    """Persist artifacts + per-sample JSON in a dedicated folder."""
    if not artifact_dir:
        return None

    dest_root = Path(artifact_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    sample_token = sample_payload.get("sample_token")
    seq_id = sequence_id or sample_payload.get("sequence_id") or 0

    slug = _sanitize_folder_name(sample_token or "sample")
    base_name = f"{seq_id:04d}_{slug}" if seq_id else slug
    target_dir = dest_root / base_name
    suffix = 1
    while target_dir.exists():
        suffix += 1
        target_dir = dest_root / f"{base_name}_{suffix:02d}"
    target_dir.mkdir(parents=True, exist_ok=False)

    copied_bev = None
    if bev_path:
        bev_src = Path(bev_path)
        if bev_src.exists():
            bev_dest = target_dir / f"bev_{bev_src.name}"
            shutil.copy2(bev_src, bev_dest)
            copied_bev = bev_dest.name
        else:
            print(f"[artifacts] Warning: BEV file not found: {bev_src}")

    copied_images: List[str] = []
    copy_images = (
        vision_requested
        and engine.use_vision
        and sample_token is not None
        and engine.nusc is not None
    )
    if copy_images:
        try:
            image_paths = resolve_cam_image_paths(engine.nusc, sample_token, view_order=DEFAULT_VIEW_ORDER)
            for idx, (view_name, img_path) in enumerate(zip(DEFAULT_VIEW_ORDER, image_paths)):
                if img_path is None:
                    continue
                img_src = Path(img_path)
                if not img_src.exists():
                    continue
                dest_name = f"{idx:02d}_{view_name}_{img_src.name}"
                img_dest = target_dir / dest_name
                shutil.copy2(img_src, img_dest)
                copied_images.append(dest_name)
        except Exception as exc:
            print(f"[artifacts] Warning: Failed to copy camera images: {exc}")
    elif vision_requested and sample_token and engine.nusc is None:
        print("[artifacts] Warning: nuScenes handle missing; skipping image copy")

    copied_lidar = None
    if sample_token and engine.nusc is not None:
        try:
            sample_rec = engine.nusc.get("sample", sample_token)
            lidar_token = (sample_rec.get("data") or {}).get("LIDAR_TOP")
            if lidar_token:
                sd_rec = engine.nusc.get("sample_data", lidar_token)
                lidar_rel = sd_rec.get("filename")
                if not lidar_rel:
                    print("[artifacts] Warning: LiDAR sample_data missing filename")
                else:
                    dataroot_val = getattr(engine.nusc, "dataroot", None)
                    if not dataroot_val:
                        print("[artifacts] Warning: nuScenes dataroot unset; skipping LiDAR copy")
                    else:
                        lidar_src = (Path(dataroot_val) / lidar_rel).resolve()
                        if lidar_src.exists():
                            lidar_dest = target_dir / f"lidar_{lidar_src.name}"
                            shutil.copy2(lidar_src, lidar_dest)
                            copied_lidar = lidar_dest.name
                        else:
                            print(f"[artifacts] Warning: LiDAR file missing → {lidar_src}")
            else:
                print("[artifacts] Warning: No LIDAR_TOP token on sample")
        except Exception as exc:
            print(f"[artifacts] Warning: Failed to copy LiDAR scan: {exc}")

    record = dict(sample_payload)
    record.setdefault("sequence_id", seq_id)
    record.setdefault("sample_token", sample_token)
    record.setdefault("created_at", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    record.setdefault("metrics", sample_payload.get("metrics", {}))
    record["artifacts"] = {
        "copied_bev_file": copied_bev,
        "copied_image_files": copied_images,
        "copied_lidar_file": copied_lidar,
    }

    with open(target_dir / "sample.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"[artifacts] Saved inference artifacts to {target_dir}")
    return target_dir


def select_random_samples(
    samples: List[Dict],
    *,
    count: int,
    split: str,
    seed: int,
    val_split: float,
) -> tuple[List[Dict], Dict[str, int]]:
    """Mirror training split logic and select a random subset without replacement."""
    total = len(samples)
    if total == 0 or count <= 0:
        return [] if count > 0 else samples, {
            "total": total,
            "train_size": 0,
            "val_size": 0,
            "available": total,
            "requested": count,
            "selected": 0,
        }

    val_fraction = max(0.0, min(1.0, float(val_split)))
    val_size = int(total * val_fraction)
    val_size = max(1, val_size) if total > 1 else min(1, total)
    val_size = min(val_size, total)
    train_size = max(0, total - val_size)

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(total, generator=generator).tolist()
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    if split == "val":
        pool = val_indices
    else:
        pool = train_indices

    available = len(pool)
    if available == 0:
        return [], {
            "total": total,
            "train_size": train_size,
            "val_size": val_size,
            "available": available,
            "requested": count,
            "selected": 0,
        }

    rng = random.Random(int(seed) + (1 if split == "val" else 0))
    rng.shuffle(pool)
    take = min(count, available)
    chosen = pool[:take]
    selected = [samples[i] for i in chosen]

    info = {
        "total": total,
        "train_size": train_size,
        "val_size": val_size,
        "available": available,
        "requested": count,
        "selected": len(selected),
    }
    return selected, info
