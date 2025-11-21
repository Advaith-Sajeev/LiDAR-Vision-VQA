"""Evaluation metrics for caption and grounding tasks"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional


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
    
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
    except ImportError:
        print("[metrics] Warning: pycocoevalcap not installed. Install with: pip install pycocoevalcap")
        return {"bleu4": 0.0, "cider": 0.0, "spice": 0.0, "bertscore_f1": 0.0}
    
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("[metrics] Warning: bert-score not installed. Install with: pip install bert-score")
        bert_score = None
    
    # Format for pycocoevalcap (expects dict format)
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    
    results = {}
    
    # BLEU-4
    try:
        bleu_scorer = Bleu(4)
        bleu_score, _ = bleu_scorer.compute_score(gts, res)
        results["bleu4"] = bleu_score[3]  # BLEU-4 is the 4th element (index 3)
    except Exception as e:
        print(f"[metrics] BLEU-4 calculation failed: {e}")
        results["bleu4"] = 0.0
    
    # CIDEr
    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        results["cider"] = cider_score
    except Exception as e:
        print(f"[metrics] CIDEr calculation failed: {e}")
        results["cider"] = 0.0
    
    # SPICE
    try:
        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(gts, res)
        results["spice"] = spice_score
    except Exception as e:
        print(f"[metrics] SPICE calculation failed: {e}")
        results["spice"] = 0.0
    
    # BERTScore
    if bert_score is not None:
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            results["bertscore_f1"] = F1.mean().item()
        except Exception as e:
            print(f"[metrics] BERTScore calculation failed: {e}")
            results["bertscore_f1"] = 0.0
    else:
        results["bertscore_f1"] = 0.0
    
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
    
    Args:
        results: List of result dictionaries with keys:
                 - prediction
                 - ground_truth
                 - dataset_type ("caption", "grounding_det_area", or "grounding_det_object")
        config: Optional config dict with metric toggles
    
    Returns:
        Dictionary with metrics for each type
    """
    # Default config: enable all metrics
    if config is None:
        config = {}
    
    caption_results = [r for r in results if r.get("dataset_type") == "caption"]
    det_area_results = [r for r in results if r.get("dataset_type") == "grounding_det_area"]
    det_object_results = [r for r in results if r.get("dataset_type") == "grounding_det_object"]
    
    metrics = {}
    
    # Caption metrics (text quality only)
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
    else:
        metrics["caption_dashboard"] = {"num_samples": 0}
    
    # Grounding det_area metrics (text quality + bbox accuracy)
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
    else:
        metrics["grounding_det_area_dashboard"] = {
            "bbox_valid_samples": 0,
            "num_samples": 0, 
            "note": "No det_area samples"
        }
    
    # Grounding det_object metrics (text quality only)
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
    else:
        metrics["grounding_det_object_dashboard"] = {
            "num_samples": 0, 
            "note": "No det_object samples"
        }
    
    return metrics
