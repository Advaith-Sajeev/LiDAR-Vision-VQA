"""
Utility functions for inference
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Dict, List
import json


def load_bev_feature(feature_path: Union[str, Path]) -> torch.Tensor:
    """
    Load BEV feature from .npy file.
    
    Args:
        feature_path: Path to .npy file
        
    Returns:
        BEV tensor [C, H, W]
    """
    bev = np.load(feature_path)
    return torch.from_numpy(bev).float()


def format_prompt(
    question: str,
    use_vision: bool = True,
    system_prompt: str = ""
) -> str:
    """
    Format a question into a prompt with special tokens.
    
    Args:
        question: User question
        use_vision: Whether to include vision tokens
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted prompt string
    """
    if system_prompt:
        question = f"{system_prompt}\n\n{question}"
    
    if use_vision:
        return f"<vision_start><vision_end><lidar_start><lidar_end>{question}\nAnswer:"
    else:
        return f"<lidar_start><lidar_end>{question}\nAnswer:"


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
