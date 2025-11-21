"""Test three-dashboard metrics calculation system

Tests the comprehensive evaluation strategy:
1. Caption dashboard: Text quality metrics
2. Grounding det_area dashboard: Text quality + Bbox accuracy
3. Grounding det_object dashboard: Text quality only (skip bbox to avoid coordinate copying)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from relative path
sys.path.insert(0, str(Path(__file__).parent))

from metrics import (
    calculate_metrics_by_type,
    calculate_caption_metrics,
    calculate_grounding_metrics,
    extract_bbox_from_text,
    extract_object_class,
)


def test_dataset_type_filtering():
    """Test that samples are correctly filtered by dataset_type"""
    
    print("=" * 80)
    print("TEST 1: Dataset Type Filtering")
    print("=" * 80)
    
    # Mock results with all three types
    results = [
        # Caption samples
        {
            "dataset_type": "caption",
            "prediction": "A busy urban intersection with multiple vehicles and pedestrians.",
            "ground_truth": "An urban scene with cars and people crossing the street.",
        },
        {
            "dataset_type": "caption",
            "prediction": "Highway scene with traffic moving at high speed.",
            "ground_truth": "Fast-moving traffic on a multi-lane highway.",
        },
        
        # Grounding det_area samples (descriptive Q, bbox A)
        {
            "dataset_type": "grounding_det_area",
            "question": "Where is the pedestrian in front of you?",
            "prediction": "There is a pedestrian at the location [8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93].",
            "ground_truth": "There is a pedestrian at the location [8.5,10.1,-7.8,-3.5,-1.9,-0.4,-1.95].",
        },
        {
            "dataset_type": "grounding_det_area",
            "question": "What vehicle is to your left?",
            "prediction": "There is a car at the location [-5.2,-2.8,10.5,15.3,-1.2,0.5,1.57].",
            "ground_truth": "There is a car at the location [-5.1,-2.9,10.4,15.2,-1.1,0.4,1.58].",
        },
        
        # Grounding det_object samples (coords in Q, bbox A)
        {
            "dataset_type": "grounding_det_object",
            "question": "What is at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82]?",
            "prediction": "There is a truck at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82].",
            "ground_truth": "There is a truck at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82].",
        },
        {
            "dataset_type": "grounding_det_object",
            "question": "What object is at [-10.5,-8.2,5.3,8.7,-1.5,0.2,0.0]?",
            "prediction": "There is a barrier at the location [-10.5,-8.2,5.3,8.7,-1.5,0.2,0.0].",
            "ground_truth": "There is a barrier at the location [-10.5,-8.2,5.3,8.7,-1.5,0.2,0.0].",
        },
    ]
    
    metrics = calculate_metrics_by_type(results)
    
    # Verify all three dashboards exist
    assert "caption_dashboard" in metrics, "Missing caption_dashboard"
    assert "grounding_det_area_dashboard" in metrics, "Missing grounding_det_area_dashboard"
    assert "grounding_det_object_dashboard" in metrics, "Missing grounding_det_object_dashboard"
    
    # Verify sample counts
    assert metrics["caption_dashboard"]["num_samples"] == 2, f"Expected 2 caption samples, got {metrics['caption_dashboard']['num_samples']}"
    assert metrics["grounding_det_area_dashboard"]["num_samples"] == 2, f"Expected 2 det_area samples, got {metrics['grounding_det_area_dashboard']['num_samples']}"
    assert metrics["grounding_det_object_dashboard"]["num_samples"] == 2, f"Expected 2 det_object samples, got {metrics['grounding_det_object_dashboard']['num_samples']}"
    
    print("\n✓ All three dashboards created successfully")
    print(f"  Caption: {metrics['caption_dashboard']['num_samples']} samples")
    print(f"  Grounding det_area: {metrics['grounding_det_area_dashboard']['num_samples']} samples")
    print(f"  Grounding det_object: {metrics['grounding_det_object_dashboard']['num_samples']} samples")
    
    return True


def test_caption_dashboard_metrics():
    """Test caption dashboard only has text quality metrics"""
    
    print("\n" + "=" * 80)
    print("TEST 2: Caption Dashboard Metrics")
    print("=" * 80)
    
    results = [
        {
            "dataset_type": "caption",
            "prediction": "A busy intersection with cars and pedestrians.",
            "ground_truth": "An urban scene with vehicles and people.",
        },
    ]
    
    metrics = calculate_metrics_by_type(results)
    cap = metrics["caption_dashboard"]
    
    # Should have text quality metrics
    assert "bleu4" in cap, "Missing BLEU-4"
    assert "cider" in cap, "Missing CIDEr"
    assert "spice" in cap, "Missing SPICE"
    assert "bertscore_f1" in cap, "Missing BERTScore"
    
    # Should NOT have bbox metrics
    assert "top1_accuracy" not in cap, "Caption should not have top1_accuracy"
    assert "bev_iou" not in cap, "Caption should not have bev_iou"
    
    print("\n✓ Caption dashboard has correct metrics:")
    print(f"  Text Quality: BLEU-4={cap['bleu4']:.4f}, CIDEr={cap['cider']:.4f}, SPICE={cap['spice']:.4f}, BERTScore={cap['bertscore_f1']:.4f}")
    print(f"  Bbox Metrics: None (as expected)")
    
    return True


def test_det_area_dashboard_metrics():
    """Test det_area dashboard has BOTH text quality AND bbox metrics"""
    
    print("\n" + "=" * 80)
    print("TEST 3: Grounding det_area Dashboard Metrics")
    print("=" * 80)
    
    results = [
        {
            "dataset_type": "grounding_det_area",
            "question": "Where is the car ahead?",
            "prediction": "There is a car at the location [8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93].",
            "ground_truth": "There is a car at the location [8.5,10.1,-7.8,-3.5,-1.9,-0.4,-1.95].",
        },
        {
            "dataset_type": "grounding_det_area",
            "question": "What is to the left?",
            "prediction": "There is a pedestrian at the location [-5.2,-2.8,10.5,15.3,-1.2,0.5,1.57].",
            "ground_truth": "There is a pedestrian at the location [-5.1,-2.9,10.4,15.2,-1.1,0.4,1.58].",
        },
    ]
    
    metrics = calculate_metrics_by_type(results)
    det_area = metrics["grounding_det_area_dashboard"]
    
    # Should have text quality metrics
    assert "bleu4" in det_area, "Missing BLEU-4"
    assert "cider" in det_area, "Missing CIDEr"
    assert "spice" in det_area, "Missing SPICE"
    assert "bertscore_f1" in det_area, "Missing BERTScore"
    
    # Should ALSO have bbox metrics
    assert "top1_accuracy" in det_area, "Missing top1_accuracy"
    assert "bev_iou" in det_area, "Missing bev_iou"
    assert "bbox_valid_samples" in det_area, "Missing bbox_valid_samples"
    
    print("\n✓ Grounding det_area dashboard has BOTH metric types:")
    print(f"  Text Quality: BLEU-4={det_area['bleu4']:.4f}, CIDEr={det_area['cider']:.4f}, SPICE={det_area['spice']:.4f}, BERTScore={det_area['bertscore_f1']:.4f}")
    print(f"  Bbox Accuracy: Top-1={det_area['top1_accuracy']:.2f}%, BEV IoU={det_area['bev_iou']:.4f}")
    print(f"  Valid bbox parses: {det_area['bbox_valid_samples']}/{det_area['num_samples']}")
    
    return True


def test_det_object_dashboard_metrics():
    """Test det_object dashboard has ONLY text quality (no bbox to avoid coordinate copying)"""
    
    print("\n" + "=" * 80)
    print("TEST 4: Grounding det_object Dashboard Metrics")
    print("=" * 80)
    
    results = [
        {
            "dataset_type": "grounding_det_object",
            "question": "What is at [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82]?",
            "prediction": "There is a truck at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82].",
            "ground_truth": "There is a truck at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82].",
        },
    ]
    
    metrics = calculate_metrics_by_type(results)
    det_obj = metrics["grounding_det_object_dashboard"]
    
    # Should have text quality metrics
    assert "bleu4" in det_obj, "Missing BLEU-4"
    assert "cider" in det_obj, "Missing CIDEr"
    assert "spice" in det_obj, "Missing SPICE"
    assert "bertscore_f1" in det_obj, "Missing BERTScore"
    
    # Should NOT have bbox metrics (to avoid coordinate copying)
    assert "top1_accuracy" not in det_obj, "det_object should not have top1_accuracy"
    assert "bev_iou" not in det_obj, "det_object should not have bev_iou"
    
    # Should have explanatory note
    assert "note" in det_obj, "Missing explanatory note"
    assert "coords in question" in det_obj["note"].lower(), "Note should mention coordinates in question"
    
    print("\n✓ Grounding det_object dashboard has correct metrics:")
    print(f"  Text Quality: BLEU-4={det_obj['bleu4']:.4f}, CIDEr={det_obj['cider']:.4f}, SPICE={det_obj['spice']:.4f}, BERTScore={det_obj['bertscore_f1']:.4f}")
    print(f"  Bbox Metrics: None (prevented to avoid coordinate copying)")
    print(f"  Note: {det_obj['note']}")
    
    return True


def test_bbox_extraction():
    """Test bbox extraction from various text formats"""
    
    print("\n" + "=" * 80)
    print("TEST 5: Bbox Extraction Robustness")
    print("=" * 80)
    
    test_cases = [
        # Standard format
        {
            "text": "There is a car at the location [8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93].",
            "expected_count": 7,
            "description": "Standard format with 7 values"
        },
        # No spaces
        {
            "text": "Object at [1.0,2.0,3.0,4.0,5.0,6.0,7.0]",
            "expected_count": 7,
            "description": "Compact format without spaces"
        },
        # With spaces
        {
            "text": "Location [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ]",
            "expected_count": 7,
            "description": "Format with spaces"
        },
        # Negative values
        {
            "text": "Position [-5.2,-3.8,10.5,15.3,-1.2,0.5,1.57]",
            "expected_count": 7,
            "description": "With negative values"
        },
        # Invalid: too few values
        {
            "text": "Incomplete [1.0, 2.0, 3.0]",
            "expected_count": None,
            "description": "Incomplete bbox (only 3 values)"
        },
        # Invalid: no bbox
        {
            "text": "There is a car ahead",
            "expected_count": None,
            "description": "No bbox coordinates"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        bbox = extract_bbox_from_text(case["text"])
        
        if case["expected_count"] is None:
            # Should fail to extract
            if bbox is None:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
                print(f"\n{status} Test {i}: {case['description']}")
                print(f"  Expected: None")
                print(f"  Got: {bbox}")
        else:
            # Should succeed
            if bbox is not None and len(bbox) == case["expected_count"]:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
                print(f"\n{status} Test {i}: {case['description']}")
                print(f"  Expected: {case['expected_count']} values")
                print(f"  Got: {bbox}")
    
    print(f"\n✓ Bbox extraction: {passed}/{len(test_cases)} tests passed")
    
    return failed == 0


def test_object_class_extraction():
    """Test object class extraction from grounding answers"""
    
    print("\n" + "=" * 80)
    print("TEST 6: Object Class Extraction")
    print("=" * 80)
    
    test_cases = [
        ("There is a car at the location [...]", "car"),
        ("A truck is ahead at [...]", "truck"),
        ("There is a pedestrian crossing [...]", "pedestrian"),
        ("A bicycle is located at [...]", "bicycle"),
        ("Traffic cone at position [...]", "traffic_cone"),
        ("Construction vehicle nearby [...]", "construction_vehicle"),
        ("Some object at [...]", None),  # Unknown object
    ]
    
    passed = 0
    failed = 0
    
    for text, expected in test_cases:
        result = extract_object_class(text)
        
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
            print(f"\n{status}: '{text[:50]}...'")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
    
    print(f"\n✓ Object class extraction: {passed}/{len(test_cases)} tests passed")
    
    return failed == 0


def test_comprehensive_evaluation():
    """Test complete evaluation pipeline with all three dashboards"""
    
    print("\n" + "=" * 80)
    print("TEST 7: Comprehensive Three-Dashboard Evaluation")
    print("=" * 80)
    
    # Realistic mixed dataset
    results = [
        # Caption samples (2)
        {
            "dataset_type": "caption",
            "prediction": "A busy urban intersection with multiple vehicles.",
            "ground_truth": "An urban scene with cars and traffic.",
        },
        {
            "dataset_type": "caption",
            "prediction": "Highway with fast-moving traffic.",
            "ground_truth": "Multi-lane highway scene with vehicles.",
        },
        
        # Grounding det_area (2) - descriptive questions
        {
            "dataset_type": "grounding_det_area",
            "question": "Where is the pedestrian?",
            "prediction": "There is a pedestrian at the location [8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93].",
            "ground_truth": "There is a pedestrian at the location [8.5,10.1,-7.8,-3.5,-1.9,-0.4,-1.95].",
        },
        {
            "dataset_type": "grounding_det_area",
            "question": "What vehicle is ahead?",
            "prediction": "There is a car at the location [20.0,25.0,0.0,5.0,-1.0,1.0,0.0].",
            "ground_truth": "There is a car at the location [20.5,25.5,0.5,5.5,-1.0,1.0,0.0].",
        },
        
        # Grounding det_object (2) - coordinate questions
        {
            "dataset_type": "grounding_det_object",
            "question": "What is at [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82]?",
            "prediction": "There is a truck at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82].",
            "ground_truth": "There is a truck at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82].",
        },
        {
            "dataset_type": "grounding_det_object",
            "question": "Describe object at [-10.5,-8.2,5.3,8.7,-1.5,0.2,0.0]?",
            "prediction": "There is a barrier at the location [-10.5,-8.2,5.3,8.7,-1.5,0.2,0.0].",
            "ground_truth": "There is a barrier at the location [-10.5,-8.2,5.3,8.7,-1.5,0.2,0.0].",
        },
    ]
    
    metrics = calculate_metrics_by_type(results)
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    
    # Caption Dashboard
    print(f"\n1. Caption Dashboard ({metrics['caption_dashboard']['num_samples']} samples):")
    print(f"   Text Quality:")
    print(f"     BLEU-4:       {metrics['caption_dashboard']['bleu4']:.4f}")
    print(f"     CIDEr:        {metrics['caption_dashboard']['cider']:.4f}")
    print(f"     SPICE:        {metrics['caption_dashboard']['spice']:.4f}")
    print(f"     BERTScore-F1: {metrics['caption_dashboard']['bertscore_f1']:.4f}")
    
    # Grounding det_area Dashboard
    det_area = metrics['grounding_det_area_dashboard']
    print(f"\n2. Grounding det_area Dashboard ({det_area['num_samples']} samples):")
    print(f"   Text Quality:")
    print(f"     BLEU-4:       {det_area['bleu4']:.4f}")
    print(f"     CIDEr:        {det_area['cider']:.4f}")
    print(f"     SPICE:        {det_area['spice']:.4f}")
    print(f"     BERTScore-F1: {det_area['bertscore_f1']:.4f}")
    print(f"   Bbox Accuracy ({det_area['bbox_valid_samples']} valid parses):")
    print(f"     Top-1 Acc:    {det_area['top1_accuracy']:.2f}%")
    print(f"     BEV IoU:      {det_area['bev_iou']:.4f}")
    
    # Grounding det_object Dashboard
    det_obj = metrics['grounding_det_object_dashboard']
    print(f"\n3. Grounding det_object Dashboard ({det_obj['num_samples']} samples):")
    print(f"   Text Quality:")
    print(f"     BLEU-4:       {det_obj['bleu4']:.4f}")
    print(f"     CIDEr:        {det_obj['cider']:.4f}")
    print(f"     SPICE:        {det_obj['spice']:.4f}")
    print(f"     BERTScore-F1: {det_obj['bertscore_f1']:.4f}")
    print(f"   Note: {det_obj['note']}")
    
    print("\n" + "=" * 80)
    print("✓ Three-dashboard evaluation completed successfully!")
    print("\nEvaluation Strategy Summary:")
    print("  - Caption: Text quality metrics (BLEU, CIDEr, SPICE, BERTScore)")
    print("  - det_area: Text quality + Bbox accuracy (Top-1 Acc, BEV IoU)")
    print("  - det_object: Text quality only (prevents coordinate copying)")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("THREE-DASHBOARD METRICS SYSTEM TEST SUITE")
    print("=" * 80)
    print("\nTesting comprehensive evaluation with:")
    print("  1. Caption dashboard: Text quality")
    print("  2. Grounding det_area: Text quality + Bbox accuracy")
    print("  3. Grounding det_object: Text quality only")
    print("\n" + "=" * 80)
    
    tests = [
        ("Dataset Type Filtering", test_dataset_type_filtering),
        ("Caption Dashboard Metrics", test_caption_dashboard_metrics),
        ("det_area Dashboard Metrics", test_det_area_dashboard_metrics),
        ("det_object Dashboard Metrics", test_det_object_dashboard_metrics),
        ("Bbox Extraction Robustness", test_bbox_extraction),
        ("Object Class Extraction", test_object_class_extraction),
        ("Comprehensive Evaluation", test_comprehensive_evaluation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "CRASH"))
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, status in results if status == "PASS")
    failed = sum(1 for _, status in results if status == "FAIL")
    crashed = sum(1 for _, status in results if status == "CRASH")
    
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed, {crashed} crashed")
    print("=" * 80)
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Three-dashboard system is working correctly.")
        print("\nKey Features Validated:")
        print("  ✓ Caption samples → Text quality metrics only")
        print("  ✓ det_area samples → Text quality + Bbox accuracy (both evaluations)")
        print("  ✓ det_object samples → Text quality only (bbox skipped)")
        print("  ✓ Prevents coordinate copying from questions")
        print("  ✓ Comprehensive spatial understanding evaluation")
    else:
        print(f"\n⚠️  {failed + crashed} test(s) failed. Review the errors above.")
    
    print("\n" + "=" * 80 + "\n")
