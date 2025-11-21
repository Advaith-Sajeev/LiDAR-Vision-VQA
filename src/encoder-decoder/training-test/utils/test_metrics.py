"""Tests for metrics utilities"""

import pytest
from training.utils.metrics import (
    extract_bbox_from_text,
    calculate_bbox_iou_3d,
    calculate_bev_iou_2d,
    extract_object_class,
    calculate_caption_metrics,
    calculate_grounding_metrics,
    calculate_metrics_by_type,
)


class TestBBoxExtraction:
    """Tests for bounding box extraction from text"""
    
    def test_extract_bbox_valid_format(self):
        """Test extraction of valid bbox format"""
        text = "The object is at [1.5, 2.3, 0.5, 4.0, 3.0, 1.8, 0.0]"
        bbox = extract_bbox_from_text(text)
        assert bbox is not None
        assert len(bbox) == 7
        assert bbox == [1.5, 2.3, 0.5, 4.0, 3.0, 1.8, 0.0]
    
    def test_extract_bbox_negative_values(self):
        """Test extraction with negative coordinates"""
        text = "[-1.5, -2.3, 0.5, 4.0, 3.0, 1.8, -0.5]"
        bbox = extract_bbox_from_text(text)
        assert bbox == [-1.5, -2.3, 0.5, 4.0, 3.0, 1.8, -0.5]
    
    def test_extract_bbox_no_bbox(self):
        """Test when no bbox present"""
        text = "This is just a description without coordinates"
        bbox = extract_bbox_from_text(text)
        assert bbox is None
    
    def test_extract_bbox_wrong_length(self):
        """Test with wrong number of coordinates"""
        text = "[1.0, 2.0, 3.0]"  # Only 3 values instead of 7
        bbox = extract_bbox_from_text(text)
        assert bbox is None
    
    def test_extract_bbox_first_occurrence(self):
        """Test that it extracts first bbox when multiple present"""
        text = "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] and [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]"
        bbox = extract_bbox_from_text(text)
        assert bbox == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


class TestBBox3DIOU:
    """Tests for 3D bounding box IOU calculation"""
    
    def test_identical_boxes(self):
        """Test IOU of identical boxes should be 1.0"""
        box1 = [0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0]  # [x_min, x_max, y_min, y_max, z_min, z_max, yaw]
        box2 = [0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0]
        iou = calculate_bbox_iou_3d(box1, box2)
        assert abs(iou - 1.0) < 1e-6
    
    def test_no_overlap(self):
        """Test IOU of non-overlapping boxes should be 0.0"""
        box1 = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
        box2 = [10.0, 12.0, 10.0, 12.0, 10.0, 12.0, 0.0]
        iou = calculate_bbox_iou_3d(box1, box2)
        assert iou == 0.0
    
    def test_partial_overlap(self):
        """Test IOU of partially overlapping boxes"""
        box1 = [0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0]
        box2 = [2.0, 6.0, 2.0, 6.0, 2.0, 6.0, 0.0]
        iou = calculate_bbox_iou_3d(box1, box2)
        # Intersection: 2x2x2 = 8, Union: 64 + 64 - 8 = 120, IOU = 8/120
        expected_iou = 8.0 / 120.0
        assert abs(iou - expected_iou) < 1e-6
    
    def test_one_inside_other(self):
        """Test IOU when one box is inside another"""
        box1 = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0]  # Large box
        box2 = [2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.0]      # Small box inside
        iou = calculate_bbox_iou_3d(box1, box2)
        # Intersection: 2*2*2=8, Union: 1000 + 8 - 8 = 1000, IOU = 8/1000
        expected_iou = 8.0 / 1000.0
        assert abs(iou - expected_iou) < 1e-6


class TestBEV2DIOU:
    """Tests for 2D BEV IOU calculation"""
    
    def test_identical_boxes_2d(self):
        """Test IOU of identical boxes in BEV"""
        box1 = [0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0]  # [x_min, x_max, y_min, y_max, z_min, z_max, yaw]
        box2 = [0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0]
        iou = calculate_bev_iou_2d(box1, box2)
        assert abs(iou - 1.0) < 1e-6
    
    def test_no_overlap_2d(self):
        """Test IOU of non-overlapping boxes in BEV"""
        box1 = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
        box2 = [10.0, 12.0, 10.0, 12.0, 0.0, 2.0, 0.0]
        iou = calculate_bev_iou_2d(box1, box2)
        assert iou == 0.0
    
    def test_different_heights_same_footprint(self):
        """Test that height doesn't affect BEV IOU"""
        box1 = [0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0]
        box2 = [0.0, 4.0, 0.0, 3.0, 5.0, 10.0, 0.0]  # Different z and height
        iou = calculate_bev_iou_2d(box1, box2)
        assert abs(iou - 1.0) < 1e-6  # Should be 1.0 since x,y footprint is same


class TestObjectClassExtraction:
    """Tests for object class extraction"""
    
    def test_extract_car(self):
        """Test extracting car class"""
        text = "This is a car in the scene"
        obj_class = extract_object_class(text)
        assert obj_class == "car"
    
    def test_extract_pedestrian(self):
        """Test extracting pedestrian class"""
        text = "A pedestrian is walking"
        obj_class = extract_object_class(text)
        assert obj_class == "pedestrian"
    
    def test_extract_truck(self):
        """Test extracting truck class"""
        text = "There is a truck ahead"
        obj_class = extract_object_class(text)
        assert obj_class == "truck"
    
    def test_no_class_found(self):
        """Test when no class keyword present"""
        text = "This is just a random description"
        obj_class = extract_object_class(text)
        assert obj_class is None
    
    def test_case_insensitive(self):
        """Test case insensitive matching"""
        text = "The BICYCLE is parked"
        obj_class = extract_object_class(text)
        assert obj_class == "bicycle"


class TestCaptionMetrics:
    """Tests for caption metrics calculation"""
    
    def test_identical_captions(self):
        """Test metrics for identical captions"""
        preds = ["The car is red", "A truck is moving"]
        refs = ["The car is red", "A truck is moving"]
        metrics = calculate_caption_metrics(preds, refs)
        
        assert "bleu4" in metrics
        assert "cider" in metrics
        assert "spice" in metrics
        assert "bertscore_f1" in metrics
        # Note: Returns 0.0 if pycocoevalcap not installed
    
    def test_completely_different(self):
        """Test metrics for completely different captions"""
        preds = ["xyz abc def"]
        refs = ["qwe rty uio"]
        metrics = calculate_caption_metrics(preds, refs)
        
        assert "bleu4" in metrics
        assert metrics["bleu4"] == 0.0
    
    def test_partial_match(self):
        """Test metrics for partially matching captions"""
        preds = ["The red car"]
        refs = ["The blue car"]
        metrics = calculate_caption_metrics(preds, refs)
        
        assert "bleu4" in metrics
        assert "cider" in metrics


class TestGroundingMetrics:
    """Tests for grounding metrics calculation"""
    
    def test_perfect_grounding(self):
        """Test grounding with perfect predictions"""
        # Predictions and references should be text with embedded bboxes
        pred_texts = ["There is a car at the location [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 0.0]."]
        gt_texts = ["There is a car at the location [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 0.0]."]
        
        metrics = calculate_grounding_metrics(pred_texts, gt_texts)
        
        assert metrics["iou_3d"] == 1.0
        assert metrics["bev_iou"] == 1.0
        assert metrics["top1_accuracy"] == 100.0
    
    def test_no_predictions(self):
        """Test grounding when no predictions made"""
        pred_texts = []
        gt_texts = []
        
        metrics = calculate_grounding_metrics(pred_texts, gt_texts)
        
        assert metrics["iou_3d"] == 0.0
        assert metrics["top1_accuracy"] == 0.0
        assert metrics["total_samples"] == 0
    
    def test_multiple_boxes(self):
        """Test grounding with multiple boxes"""
        pred_texts = [
            "There is a car at [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 0.0]",
            "A truck is at [10.0, 12.0, 20.0, 24.0, 30.0, 35.0, 0.0]"
        ]
        gt_texts = [
            "There is a car at [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 0.0]",
            "A truck is at [10.0, 12.0, 20.0, 24.0, 30.0, 35.0, 0.0]"
        ]
        
        metrics = calculate_grounding_metrics(pred_texts, gt_texts)
        
        assert metrics["iou_3d"] == 1.0
        assert metrics["bev_iou"] == 1.0
        assert metrics["top1_accuracy"] == 100.0


class TestMetricsByType:
    """Tests for metrics calculation by task type"""
    
    def test_mixed_results(self):
        """Test calculating metrics for mixed task types"""
        results = [
            {
                "dataset_type": "caption",
                "prediction": "A red car",
                "ground_truth": "A red car"
            },
            {
                "dataset_type": "grounding",
                "prediction": "Car at [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 0.0]",
                "ground_truth": "Car at [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 0.0]"
            }
        ]
        
        metrics = calculate_metrics_by_type(results)
        
        assert "caption_dashboard" in metrics
        assert "grounding_dashboard" in metrics
        assert metrics["caption_dashboard"]["num_samples"] == 1
        assert metrics["grounding_dashboard"]["num_samples"] == 1
    
    def test_empty_results(self):
        """Test with empty results list"""
        results = []
        metrics = calculate_metrics_by_type(results)
        
        # Should return dashboards with 0 samples
        assert "caption_dashboard" in metrics
        assert "grounding_dashboard" in metrics
        assert metrics["caption_dashboard"]["num_samples"] == 0
        assert metrics["grounding_dashboard"]["num_samples"] == 0
    
    def test_caption_only(self):
        """Test with only caption tasks"""
        results = [
            {"dataset_type": "caption", "prediction": "car", "ground_truth": "car"},
            {"dataset_type": "caption", "prediction": "truck", "ground_truth": "vehicle"}
        ]
        
        metrics = calculate_metrics_by_type(results)
        
        assert "caption_dashboard" in metrics
        assert "grounding_dashboard" in metrics
        assert metrics["caption_dashboard"]["num_samples"] == 2
        assert metrics["grounding_dashboard"]["num_samples"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
