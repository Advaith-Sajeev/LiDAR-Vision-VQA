"""Evaluation metrics for caption and grounding tasks"""

import re
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple, Optional


__all__ = [
    "calculate_caption_metrics",
    "calculate_grounding_metrics",
    "calculate_metrics_by_type",
    "calculate_sample_level_metrics",
]


def extract_bbox_from_text(text: str) -> Optional[List[float]]:
    """
    Extract 7D bounding box from grounding answer text.
    
    Format: [x_min, x_max, y_min, y_max, z_min, z_max, orientation]
    Example: "There is a car at the location [8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93]."
    
    For nested lists (multiple boxes), extracts the first valid 7D bbox.
    
    Returns:
        List of 7 floats or None if not found
    """
    # Match pattern: [...] with numbers
    pattern = r'\[([-\d.,\s]+)\]'
    matches = re.findall(pattern, text)
    
    if not matches:
        return None
    
    # Try each match to find a valid 7D bbox
    for coords_str in matches:
        try:
            # Parse comma-separated numbers
            coords = [float(x.strip()) for x in coords_str.split(',')]
            
            # Check if we have exactly 7 values
            if len(coords) == 7:
                return coords
            
            # If more than 7, might be nested list - try first 7
            if len(coords) > 7:
                # Check if first 7 values form a valid bbox
                first_bbox = coords[:7]
                return first_bbox
                
        except (ValueError, IndexError):
            continue
    
    return None


def calculate_bbox_iou_3d(box1: List[float], box2: List[float]) -> float:
    """
    Calculate 3D IoU between two bounding boxes.
    
    Box format: [x_min, x_max, y_min, y_max, z_min, z_max, orientation]
    
    Returns:
        IoU value between 0 and 1
    """
    # Extract coordinates
    x1_min, x1_max, y1_min, y1_max, z1_min, z1_max, _ = box1
    x2_min, x2_max, y2_min, y2_max, z2_min, z2_max, _ = box2
    
    # Calculate intersection in each dimension
    x_inter_min = max(x1_min, x2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_min = max(y1_min, y2_min)
    y_inter_max = min(y1_max, y2_max)
    z_inter_min = max(z1_min, z2_min)
    z_inter_max = min(z1_max, z2_max)
    
    # Check if there's overlap
    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min or z_inter_max <= z_inter_min:
        return 0.0
    
    # Calculate intersection volume
    inter_volume = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min) * (z_inter_max - z_inter_min)
    
    # Calculate volumes of each box
    vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    
    # Calculate union volume
    union_volume = vol1 + vol2 - inter_volume
    
    # Calculate IoU
    iou = inter_volume / union_volume if union_volume > 0 else 0.0
    
    return iou


def calculate_bev_iou_2d(box1: List[float], box2: List[float]) -> float:
    """
    Calculate 2D BEV IoU (ignoring z-dimension).
    
    Box format: [x_min, x_max, y_min, y_max, z_min, z_max, orientation]
    
    Returns:
        IoU value between 0 and 1
    """
    # Extract x, y coordinates only
    x1_min, x1_max, y1_min, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, x2_max, y2_min, y2_max = box2[0], box2[1], box2[2], box2[3]
    
    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_min = max(y1_min, y2_min)
    y_inter_max = min(y1_max, y2_max)
    
    # Check if there's overlap
    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0
    
    # Calculate intersection area
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Calculate areas of each box
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate union area
    union_area = area1 + area2 - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def extract_object_class(text: str) -> Optional[str]:
    """
    Extract object class from grounding answer.
    
    Example: "There is a car at the location..." -> "car"
    """
    text = text.lower()
    
    # Common object classes in nuScenes
    objects = [
        "car", "truck", "bus", "trailer", "construction_vehicle",
        "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier"
    ]
    
    for obj in objects:
        if obj.replace("_", " ") in text or obj in text:
            return obj
    
    return None


def calculate_caption_metrics(predictions: List[str], references: List[str], config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Calculate caption evaluation metrics: BLEU-4, CIDEr, SPICE, BERTScore
    
    Only computes metrics that are enabled in config to avoid expensive operations
    (e.g., SPICE requires downloading Stanford CoreNLP).
    
    Args:
        predictions: List of predicted captions
        references: List of ground truth captions
        config: Optional config dict with metric toggles (eval_caption_*, eval_det_area_*, eval_det_object_*)
        
    Returns:
        Dictionary with metric scores
    """
    # Default: compute all metrics if no config provided
    if config is None:
        config = {}
    
    # Check which metrics are actually needed by ANY dashboard
    # This avoids computing expensive metrics when disabled
    needs_bleu = (
        config.get("eval_caption_bleu4", True) or
        config.get("eval_det_area_bleu4", True) or
        config.get("eval_det_object_bleu4", True)
    )
    needs_cider = (
        config.get("eval_caption_cider", True) or
        config.get("eval_det_area_cider", True) or
        config.get("eval_det_object_cider", True)
    )
    needs_spice = (
        config.get("eval_caption_spice", False) or
        config.get("eval_det_area_spice", False) or
        config.get("eval_det_object_spice", False)
    )
    needs_bertscore = (
        config.get("eval_caption_bertscore", False) or
        config.get("eval_det_area_bertscore", False) or
        config.get("eval_det_object_bertscore", False)
    )
    
    results = {"bleu4": 0.0, "cider": 0.0, "spice": 0.0, "bertscore_f1": 0.0}
    
    # Only import and compute metrics that are needed
    if needs_bleu or needs_cider or needs_spice:
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.cider.cider import Cider
            if needs_spice:
                from pycocoevalcap.spice.spice import Spice
        except ImportError:
            print("[metrics] Warning: pycocoevalcap not installed. Install with: pip install pycocoevalcap")
            return results
    
    if needs_bertscore:
        try:
            from bert_score import score as bert_score
        except ImportError:
            print("[metrics] Warning: bert-score not installed. Install with: pip install bert-score")
            bert_score = None
    else:
        bert_score = None
    
    # Filter out empty predictions/references to avoid metric calculation crashes
    # pycocoevalcap and BERTScore can fail or produce undefined results with empty strings
    valid_pairs = [
        (i, pred, ref) for i, (pred, ref) in enumerate(zip(predictions, references))
        if pred.strip() and ref.strip()  # Both must be non-empty after stripping whitespace
    ]
    
    if not valid_pairs:
        print("[metrics] Warning: All predictions or references are empty. Returning zero scores.")
        return results
    
    # Report if any pairs were filtered
    num_filtered = len(predictions) - len(valid_pairs)
    if num_filtered > 0:
        print(f"[metrics] Filtered {num_filtered}/{len(predictions)} empty prediction/reference pairs")
    
    # Format for pycocoevalcap (expects dict format)
    # Use filtered pairs with re-indexed keys
    gts = {i: [triplet[2]] for i, triplet in enumerate(valid_pairs)}  # references
    res = {i: [triplet[1]] for i, triplet in enumerate(valid_pairs)}  # predictions
    
    # BLEU-4
    if needs_bleu:
        try:
            bleu_scorer = Bleu(4)
            bleu_score, _ = bleu_scorer.compute_score(gts, res)
            results["bleu4"] = bleu_score[3]  # BLEU-4 is the 4th element (index 3)
        except Exception as e:
            print(f"[metrics] BLEU-4 calculation failed: {e}")
    
    # CIDEr
    if needs_cider:
        try:
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            results["cider"] = cider_score
        except Exception as e:
            print(f"[metrics] CIDEr calculation failed: {e}")
    
    # SPICE (only if explicitly enabled - requires Java + Stanford CoreNLP)
    if needs_spice:
        try:
            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(gts, res)
            results["spice"] = spice_score
        except Exception as e:
            print(f"[metrics] SPICE calculation failed: {e}")
    
    # BERTScore (only if explicitly enabled - downloads RoBERTa model)
    # Use filtered predictions/references to avoid empty string issues
    if needs_bertscore and bert_score is not None:
        try:
            filtered_preds = [triplet[1] for triplet in valid_pairs]
            filtered_refs = [triplet[2] for triplet in valid_pairs]
            P, R, F1 = bert_score(filtered_preds, filtered_refs, lang="en", verbose=False)
            results["bertscore_f1"] = F1.mean().item()
        except Exception as e:
            print(f"[metrics] BERTScore calculation failed: {e}")
    
    return results


def calculate_grounding_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate grounding metrics: Top-1 Accuracy + BEV IoU
    
    Extracts 7D bounding boxes from text answers and calculates:
    - Top-1 Accuracy: Correct object class identification
    - BEV IoU: 2D Intersection over Union in Bird's Eye View
    - 3D IoU: Full 3D bounding box IoU (bonus metric)
    
    Box format: [x_min, x_max, y_min, y_max, z_min, z_max, orientation]
    
    Args:
        predictions: List of predicted grounding answers (text with bboxes)
        references: List of ground truth grounding answers (text with bboxes)
    
    Returns:
        Dictionary with:
        - top1_accuracy: Percentage of correctly identified object classes
        - bev_iou: Average 2D BEV IoU across all predictions
        - iou_3d: Average 3D IoU (bonus metric)
        - valid_samples: Number of samples with valid bbox parsing
        - total_samples: Total number of samples
    """
    if not predictions or not references:
        return {"top1_accuracy": 0.0, "bev_iou": 0.0, "iou_3d": 0.0, "valid_samples": 0, "total_samples": 0}
    
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    correct_class = 0
    bev_ious = []
    iou_3ds = []
    valid_samples = 0
    
    for pred, ref in zip(predictions, references):
        # Extract object classes
        pred_class = extract_object_class(pred)
        ref_class = extract_object_class(ref)
        
        # Extract bounding boxes
        pred_bbox = extract_bbox_from_text(pred)
        ref_bbox = extract_bbox_from_text(ref)
        
        # Skip if parsing failed
        if pred_bbox is None or ref_bbox is None:
            continue
        
        valid_samples += 1
        
        # Calculate Top-1 Accuracy (class match)
        if pred_class == ref_class and pred_class is not None:
            correct_class += 1
        
        # Calculate BEV IoU (2D)
        bev_iou = calculate_bev_iou_2d(pred_bbox, ref_bbox)
        bev_ious.append(bev_iou)
        
        # Calculate 3D IoU (bonus)
        iou_3d = calculate_bbox_iou_3d(pred_bbox, ref_bbox)
        iou_3ds.append(iou_3d)
    
    # Avoid division by zero
    if valid_samples == 0:
        return {
            "top1_accuracy": 0.0,
            "bev_iou": 0.0,
            "iou_3d": 0.0,
            "valid_samples": 0,
            "total_samples": len(predictions)
        }
    
    metrics = {
        "top1_accuracy": (correct_class / valid_samples) * 100.0,
        "bev_iou": np.mean(bev_ious) if bev_ious else 0.0,
        "iou_3d": np.mean(iou_3ds) if iou_3ds else 0.0,
        "valid_samples": valid_samples,
        "total_samples": len(predictions)
    }
    
    return metrics



def calculate_metrics_by_type(results: List[Dict], config: Optional[Dict] = None) -> Dict:
    """
    Calculate metrics grouped by dataset type.
    
    Dynamically calculates metrics based on:
    1. Which dataset types are present in results
    2. The dataset_mode config setting
    3. Individual metric toggle settings
    
    Args:
        results: List of result dictionaries with keys:
                 - prediction
                 - ground_truth
                 - dataset_type ("caption", "grounding_det_area", or "grounding_det_object")
        config: Optional config dict with metric toggles and dataset_mode
    
    Returns:
        Dictionary with metrics for each type (only includes dashboards with samples)
    """
    # Default config: enable all metrics
    if config is None:
        config = {}
    
    # Get dataset mode for dynamic metric calculation
    dataset_mode = config.get("dataset_mode", "both")
    
    caption_results = [r for r in results if r.get("dataset_type") == "caption"]
    det_area_results = [r for r in results if r.get("dataset_type") == "grounding_det_area"]
    det_object_results = [r for r in results if r.get("dataset_type") == "grounding_det_object"]
    
    metrics = {}
    
    # Caption metrics (text quality only) - only calculate if we have caption samples
    if caption_results:
        cap_preds = [r["prediction"] for r in caption_results]
        cap_refs = [r["ground_truth"] for r in caption_results]
        text_metrics = calculate_caption_metrics(cap_preds, cap_refs, config)
        
        # Filter metrics based on config toggles
        metrics["caption_dashboard"] = {}
        if config.get("eval_caption_bleu4", True):
            metrics["caption_dashboard"]["bleu4"] = text_metrics["bleu4"]
        if config.get("eval_caption_cider", True):
            metrics["caption_dashboard"]["cider"] = text_metrics["cider"]
        if config.get("eval_caption_spice", True):
            metrics["caption_dashboard"]["spice"] = text_metrics["spice"]
        if config.get("eval_caption_bertscore", True):
            metrics["caption_dashboard"]["bertscore_f1"] = text_metrics["bertscore_f1"]
        
        metrics["caption_dashboard"]["num_samples"] = len(caption_results)
    elif dataset_mode in ("caption", "both"):
        # Only show empty dashboard if mode expects caption samples
        metrics["caption_dashboard"] = {"num_samples": 0, "note": "No caption samples available"}
    
    # Grounding det_area metrics (text quality + bbox accuracy)
    # Only calculate if we have det_area samples AND mode includes grounding
    if det_area_results:
        area_preds = [r["prediction"] for r in det_area_results]
        area_refs = [r["ground_truth"] for r in det_area_results]
        
        # Text quality metrics
        text_metrics = calculate_caption_metrics(area_preds, area_refs, config)
        
        # Bbox accuracy metrics
        bbox_metrics = calculate_grounding_metrics(area_preds, area_refs)
        
        # Filter metrics based on config toggles
        metrics["grounding_det_area_dashboard"] = {}
        if config.get("eval_det_area_bleu4", True):
            metrics["grounding_det_area_dashboard"]["bleu4"] = text_metrics["bleu4"]
        if config.get("eval_det_area_cider", True):
            metrics["grounding_det_area_dashboard"]["cider"] = text_metrics["cider"]
        if config.get("eval_det_area_spice", True):
            metrics["grounding_det_area_dashboard"]["spice"] = text_metrics["spice"]
        if config.get("eval_det_area_bertscore", True):
            metrics["grounding_det_area_dashboard"]["bertscore_f1"] = text_metrics["bertscore_f1"]
        if config.get("eval_det_area_top1_acc", True):
            metrics["grounding_det_area_dashboard"]["top1_accuracy"] = bbox_metrics["top1_accuracy"]
        if config.get("eval_det_area_bev_iou", True):
            metrics["grounding_det_area_dashboard"]["bev_iou"] = bbox_metrics["bev_iou"]
        
        metrics["grounding_det_area_dashboard"]["bbox_valid_samples"] = bbox_metrics["valid_samples"]
        metrics["grounding_det_area_dashboard"]["num_samples"] = len(det_area_results)
        metrics["grounding_det_area_dashboard"]["note"] = "Text quality + bbox accuracy"
    elif dataset_mode in ("grounding", "both"):
        # Only show empty dashboard if mode expects grounding samples
        metrics["grounding_det_area_dashboard"] = {
            "bbox_valid_samples": 0,
            "num_samples": 0, 
            "note": "No det_area samples available"
        }
    # If dataset_mode is "caption", don't include grounding dashboards at all
    
    # Grounding det_object metrics (text quality only)
    # Only calculate if we have det_object samples AND mode includes grounding
    if det_object_results:
        obj_preds = [r["prediction"] for r in det_object_results]
        obj_refs = [r["ground_truth"] for r in det_object_results]
        
        text_metrics = calculate_caption_metrics(obj_preds, obj_refs, config)
        
        # Filter metrics based on config toggles
        metrics["grounding_det_object_dashboard"] = {}
        if config.get("eval_det_object_bleu4", True):
            metrics["grounding_det_object_dashboard"]["bleu4"] = text_metrics["bleu4"]
        if config.get("eval_det_object_cider", True):
            metrics["grounding_det_object_dashboard"]["cider"] = text_metrics["cider"]
        if config.get("eval_det_object_spice", True):
            metrics["grounding_det_object_dashboard"]["spice"] = text_metrics["spice"]
        if config.get("eval_det_object_bertscore", True):
            metrics["grounding_det_object_dashboard"]["bertscore_f1"] = text_metrics["bertscore_f1"]
        
        metrics["grounding_det_object_dashboard"]["num_samples"] = len(det_object_results)
        metrics["grounding_det_object_dashboard"]["note"] = "Text quality only (coords in question)"
    elif dataset_mode in ("grounding", "both"):
        # Only show empty dashboard if mode expects grounding samples
        metrics["grounding_det_object_dashboard"] = {
            "num_samples": 0, 
            "note": "No det_object samples available"
        }
    # If dataset_mode is "caption", don't include grounding dashboards at all
    
    return metrics


def _tokenize_for_metrics(text: str) -> List[str]:
    return [tok for tok in text.strip().lower().split() if tok]


def _basic_text_overlap_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    pred_tokens = _tokenize_for_metrics(prediction)
    gt_tokens = _tokenize_for_metrics(ground_truth)

    if not pred_tokens and not gt_tokens:
        return {
            "exact_match": True,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "prediction_len": 0,
            "ground_truth_len": 0,
            "token_overlap": 0,
        }

    if not gt_tokens:
        return {
            "exact_match": False,
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "prediction_len": len(pred_tokens),
            "ground_truth_len": 0,
            "token_overlap": 0,
        }

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    overlap = pred_counter & gt_counter
    overlap_count = sum(overlap.values())

    precision = overlap_count / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap_count / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "exact_match": bool(ground_truth.strip()) and prediction.strip().lower() == ground_truth.strip().lower(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prediction_len": len(pred_tokens),
        "ground_truth_len": len(gt_tokens),
        "token_overlap": overlap_count,
    }


def calculate_sample_level_metrics(result: Dict, config: Optional[Dict] = None) -> Dict[str, float]:
    """Compute lightweight per-sample metrics for logging and artifact export."""

    prediction = result.get("prediction", "") or ""
    ground_truth = result.get("ground_truth", "") or ""
    dataset_type = result.get("dataset_type", "caption")

    metrics = _basic_text_overlap_metrics(prediction, ground_truth)

    if dataset_type == "grounding_det_area" and ground_truth:
        grounding = calculate_grounding_metrics([prediction], [ground_truth])
        metrics.update(
            {
                "top1_accuracy": grounding.get("top1_accuracy", 0.0),
                "bev_iou": grounding.get("bev_iou", 0.0),
                "bbox_valid": bool(grounding.get("valid_samples", 0)),
            }
        )

    return metrics
