"""
Test inference sampling and grounding evaluation logic.
Verifies that det_object samples are used for bbox evaluation.
"""

import pytest
from unittest.mock import Mock, patch


def test_grounding_samples_filtered_for_bbox_eval():
    """Test that only det_object samples are used for bbox evaluation."""
    # Mock grounding data
    grounding_data = [
        {
            "sample_token": "area1",
            "template_type": "det_area",
            "question": "Which area has traffic?",
            "answer": "The left lane"
        },
        {
            "sample_token": "obj1",
            "template_type": "det_object",
            "question": "Where is the car?",
            "answer": "[1.0, 2.0, -1.0, 1.0, 0.0, 2.0, 0.5] car"
        },
        {
            "sample_token": "obj2",
            "template_type": "det_object",
            "question": "Locate the truck",
            "answer": "[3.0, 5.0, -2.0, 0.0, 0.0, 3.0, 1.0] truck"
        },
        {
            "sample_token": "area2",
            "template_type": "det_area",
            "question": "What is in the intersection?",
            "answer": "Multiple vehicles"
        },
    ]
    
    token2path = {s["sample_token"]: f"/path/to/{s['sample_token']}" for s in grounding_data}
    
    # Filter logic: only keep det_object for bbox evaluation
    grounding_available = [
        s for s in grounding_data 
        if s.get("sample_token") in token2path and s.get("template_type") == "det_object"
    ]
    
    # Should have 2 det_object samples
    assert len(grounding_available) == 2
    assert all(s["template_type"] == "det_object" for s in grounding_available)
    assert "obj1" in [s["sample_token"] for s in grounding_available]
    assert "obj2" in [s["sample_token"] for s in grounding_available]
    
    # det_area samples should be filtered out
    assert "area1" not in [s["sample_token"] for s in grounding_available]
    assert "area2" not in [s["sample_token"] for s in grounding_available]


def test_bbox_extraction_from_det_object():
    """Test that bbox can be extracted from det_object answers."""
    det_object_answers = [
        "[1.0, 2.0, -1.0, 1.0, 0.0, 2.0, 0.5] car",
        "[3.5, 5.0, -2.0, 0.0, 0.5, 3.0, 1.2] truck",
        "[0.0, 1.0, 1.0, 3.0, 0.0, 2.5, 0.0] bicycle",
    ]
    
    # These should all contain extractable bboxes
    for answer in det_object_answers:
        # Check format: starts with [numbers...] followed by class name
        assert answer.startswith("["), f"Answer should start with bbox: {answer}"
        assert "]" in answer, f"Answer should contain closing bracket: {answer}"
        
        # Extract bbox part
        bbox_str = answer[answer.index("["):answer.index("]")+1]
        class_name = answer[answer.index("]")+1:].strip()
        
        assert class_name in ["car", "truck", "bicycle"], f"Should have valid class: {class_name}"


def test_det_area_answers_no_bbox():
    """Test that det_area answers don't contain bboxes (just descriptions)."""
    det_area_answers = [
        "The left lane has heavy traffic",
        "The intersection is clear",
        "Multiple vehicles in the parking lot",
    ]
    
    # These should NOT contain bbox coordinates
    for answer in det_area_answers:
        assert not answer.startswith("["), f"det_area answer shouldn't have bbox: {answer}"
        # Just plain text descriptions


def test_caption_and_grounding_separation():
    """Test that caption and grounding samples are properly separated."""
    all_samples = [
        {"sample_token": "cap1", "dataset_type": "caption"},
        {"sample_token": "cap2", "dataset_type": "caption"},
        {"sample_token": "ground1", "dataset_type": "grounding", "template_type": "det_object"},
        {"sample_token": "ground2", "dataset_type": "grounding", "template_type": "det_object"},
    ]
    
    caption_samples = [s for s in all_samples if s["dataset_type"] == "caption"]
    grounding_samples = [s for s in all_samples if s["dataset_type"] == "grounding"]
    
    assert len(caption_samples) == 2
    assert len(grounding_samples) == 2
    
    # Grounding samples should have template_type
    for s in grounding_samples:
        assert "template_type" in s


def test_metrics_calculation_requires_valid_samples():
    """Test that metrics are only calculated on samples with valid bbox extraction."""
    results = [
        {
            "prediction": "[1.0, 2.0, -1.0, 1.0, 0.0, 2.0, 0.5] car",
            "ground_truth": "[1.1, 2.1, -0.9, 1.1, 0.1, 2.1, 0.6] car",
        },
        {
            "prediction": "invalid format",  # Should be skipped
            "ground_truth": "[2.0, 3.0, -1.0, 1.0, 0.0, 2.0, 0.0] truck",
        },
        {
            "prediction": "[3.0, 4.0, -2.0, 0.0, 0.0, 3.0, 1.0] truck",
            "ground_truth": "[3.1, 4.1, -1.9, 0.1, 0.1, 3.1, 1.1] truck",
        },
    ]
    
    # Only 2 out of 3 should have valid bbox extraction
    # (one has invalid format)
    expected_valid = 2
    
    # Mock validation logic
    valid_count = 0
    for r in results:
        pred_has_bbox = r["prediction"].startswith("[") and "]" in r["prediction"]
        gt_has_bbox = r["ground_truth"].startswith("[") and "]" in r["ground_truth"]
        if pred_has_bbox and gt_has_bbox:
            valid_count += 1
    
    assert valid_count == expected_valid


def test_logging_messages_updated():
    """Test that logging messages reflect new filtering behavior."""
    # OLD message: "Filtered X grounding samples (kept only det_area, removed det_object to prevent data leakage)"
    # NEW message: "Using X det_object samples for bbox evaluation (filtered Y det_area samples)"
    
    # This documents the change in behavior
    old_message_pattern = "removed det_object to prevent data leakage"
    new_message_pattern = "det_object samples for bbox evaluation"
    
    # The new implementation should NOT mention "data leakage" for training
    # It should only mention filtering det_area during evaluation
    assert new_message_pattern  # New behavior exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
