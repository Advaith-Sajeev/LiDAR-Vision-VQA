"""Evaluation metrics for caption and grounding tasks"""

import re
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple, Optional


__all__ = [
    "calculate_caption_metrics",
    "calculate_metrics_by_type",
    "calculate_sample_level_metrics",
]





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
    
    needs_rouge = (
        config.get("eval_caption_rougel", False) or
        config.get("eval_det_area_rougel", False) or
        config.get("eval_det_object_rougel", False)
    )
    needs_meteor = (
        config.get("eval_caption_meteor", False) or
        config.get("eval_det_area_meteor", False) or
        config.get("eval_det_object_meteor", False)
    )

    results = {
        "bleu4": 0.0,
        "cider": 0.0,
        "spice": 0.0,
        "bertscore_f1": 0.0,
        "rouge_l": 0.0,
        "meteor": 0.0,
    }
    
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

    if needs_rouge:
        try:
            from rouge_score import rouge_scorer
            rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            print("[metrics] Warning: rouge-score not installed. Install with: pip install rouge-score")
            rouge_scorer = None
    else:
        rouge_scorer = None

    if needs_meteor:
        try:
            from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
        except ImportError:
            print("[metrics] Warning: nltk meteor_score not available. Install nltk>=3.8.1")
            meteor_fn = None
        else:
            meteor_fn = nltk_meteor_score
    else:
        meteor_fn = None
    
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

    if needs_rouge and rouge_scorer is not None:
        try:
            rouge_scores = []
            for _, pred, ref in valid_pairs:
                score = rouge_scorer.score(ref, pred)
                rouge_scores.append(score["rougeL"].fmeasure)
            results["rouge_l"] = float(np.mean(rouge_scores)) if rouge_scores else 0.0
        except Exception as e:
            print(f"[metrics] ROUGE-L calculation failed: {e}")

    if needs_meteor and meteor_fn is not None:
        try:
            meteor_scores = [
                meteor_fn([ref.split()], pred.split())
                for _, pred, ref in valid_pairs
            ]
            results["meteor"] = float(np.mean(meteor_scores)) if meteor_scores else 0.0
        except Exception as e:
            print(f"[metrics] METEOR calculation failed: {e}")
    
    return results






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
    
    caption_results = [r for r in results if r.get("dataset_type") == "caption"]
    
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
        if config.get("eval_caption_rougel", False):
            metrics["caption_dashboard"]["rouge_l"] = text_metrics["rouge_l"]
        if config.get("eval_caption_meteor", False):
            metrics["caption_dashboard"]["meteor"] = text_metrics["meteor"]
        if config.get("eval_caption_spice", True):
            metrics["caption_dashboard"]["spice"] = text_metrics["spice"]
        if config.get("eval_caption_bertscore", True):
            metrics["caption_dashboard"]["bertscore_f1"] = text_metrics["bertscore_f1"]
        
        metrics["caption_dashboard"]["num_samples"] = len(caption_results)
    else:
        # Only show empty dashboard if no caption samples
        metrics["caption_dashboard"] = {"num_samples": 0, "note": "No caption samples available"}
    
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
    return metrics
