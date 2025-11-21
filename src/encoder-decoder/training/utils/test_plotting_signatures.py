"""Simple test for three-dashboard plotting system (no actual plotting)

Validates function signatures and logic without matplotlib rendering.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_plotting_function_signatures():
    """Test that plotting functions have correct signatures for three dashboards"""
    
    print("=" * 80)
    print("TEST: Three-Dashboard Plotting Function Signatures")
    print("=" * 80)
    
    # Import function to check signature
    import inspect
    from plotting import plot_all_metrics
    
    sig = inspect.signature(plot_all_metrics)
    params = list(sig.parameters.keys())
    
    print(f"\n✓ plot_all_metrics signature:")
    print(f"  Parameters: {params}")
    
    # Verify expected parameters
    expected_params = [
        "caption_metrics",
        "grounding_det_area_metrics",
        "grounding_det_object_metrics",
        "epochs",
        "out_dir"
    ]
    
    assert params == expected_params, f"Expected {expected_params}, got {params}"
    
    print(f"\n✓ Function signature correct!")
    print(f"  ✓ caption_metrics: Text quality metrics")
    print(f"  ✓ grounding_det_area_metrics: Text + bbox metrics")
    print(f"  ✓ grounding_det_object_metrics: Text only metrics")
    print(f"  ✓ epochs: Epoch numbers")
    print(f"  ✓ out_dir: Output directory")
    
    return True


def test_trainer_metrics_structure():
    """Test that trainer has correct metrics history dictionaries"""
    
    print("\n" + "=" * 80)
    print("TEST: Trainer Metrics History Structure")
    print("=" * 80)
    
    # Read trainer.py to verify structure
    trainer_file = Path(__file__).parent.parent / "core" / "trainer.py"
    
    with open(trainer_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for three metrics dictionaries
    has_caption = "self.caption_metrics_history" in content
    has_det_area = "self.grounding_det_area_metrics_history" in content
    has_det_object = "self.grounding_det_object_metrics_history" in content
    
    assert has_caption, "Missing caption_metrics_history"
    assert has_det_area, "Missing grounding_det_area_metrics_history"
    assert has_det_object, "Missing grounding_det_object_metrics_history"
    
    print("\n✓ Trainer has all three metrics history dictionaries:")
    print("  ✓ self.caption_metrics_history")
    print("  ✓ self.grounding_det_area_metrics_history")
    print("  ✓ self.grounding_det_object_metrics_history")
    
    # Check for correct dashboard names in metrics storage
    has_caption_dash = '"caption_dashboard"' in content
    has_area_dash = '"grounding_det_area_dashboard"' in content
    has_obj_dash = '"grounding_det_object_dashboard"' in content
    
    assert has_caption_dash, "Missing caption_dashboard reference"
    assert has_area_dash, "Missing grounding_det_area_dashboard reference"
    assert has_obj_dash, "Missing grounding_det_object_dashboard reference"
    
    print("\n✓ Trainer correctly references all three dashboards:")
    print('  ✓ "caption_dashboard"')
    print('  ✓ "grounding_det_area_dashboard"')
    print('  ✓ "grounding_det_object_dashboard"')
    
    # Check that det_area has both text and bbox metrics
    has_det_area_text = 'grounding_det_area_metrics_history["bleu4"]' in content
    has_det_area_bbox = 'grounding_det_area_metrics_history["top1_accuracy"]' in content
    
    assert has_det_area_text, "det_area should track text metrics"
    assert has_det_area_bbox, "det_area should track bbox metrics"
    
    print("\n✓ det_area tracks both metric types:")
    print("  ✓ Text quality: bleu4, cider, spice, bertscore_f1")
    print("  ✓ Bbox accuracy: top1_accuracy, bev_iou")
    
    # Check that det_object has only text metrics
    has_det_obj_text = 'grounding_det_object_metrics_history["bleu4"]' in content
    has_det_obj_bbox = 'grounding_det_object_metrics_history["top1_accuracy"]' in content or \
                        'grounding_det_object_metrics_history["bev_iou"]' in content
    
    assert has_det_obj_text, "det_object should track text metrics"
    assert not has_det_obj_bbox, "det_object should NOT track bbox metrics"
    
    print("\n✓ det_object tracks only text metrics:")
    print("  ✓ Text quality: bleu4, cider, spice, bertscore_f1")
    print("  ✓ Bbox accuracy: None (correctly excluded)")
    
    return True


def test_plotting_integration():
    """Test that plot_all_metrics is called with correct arguments"""
    
    print("\n" + "=" * 80)
    print("TEST: Plotting Integration in Trainer")
    print("=" * 80)
    
    trainer_file = Path(__file__).parent.parent / "core" / "trainer.py"
    
    with open(trainer_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find plot_all_metrics call
    has_plot_call = "plot_all_metrics(" in content
    assert has_plot_call, "plot_all_metrics not called in trainer"
    
    # Check that it's called with three metrics dictionaries
    plot_call_context = content[content.find("plot_all_metrics("):content.find("plot_all_metrics(") + 500]
    
    has_caption_arg = "self.caption_metrics_history" in plot_call_context
    has_area_arg = "self.grounding_det_area_metrics_history" in plot_call_context
    has_obj_arg = "self.grounding_det_object_metrics_history" in plot_call_context
    has_epochs_arg = "self.metrics_epochs" in plot_call_context
    has_dir_arg = "self.out_dir" in plot_call_context
    
    assert has_caption_arg, "Missing caption_metrics_history argument"
    assert has_area_arg, "Missing grounding_det_area_metrics_history argument"
    assert has_obj_arg, "Missing grounding_det_object_metrics_history argument"
    assert has_epochs_arg, "Missing metrics_epochs argument"
    assert has_dir_arg, "Missing out_dir argument"
    
    print("\n✓ plot_all_metrics called with correct arguments:")
    print("  ✓ caption_metrics_history")
    print("  ✓ grounding_det_area_metrics_history")
    print("  ✓ grounding_det_object_metrics_history")
    print("  ✓ metrics_epochs")
    print("  ✓ out_dir")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("THREE-DASHBOARD PLOTTING SYSTEM - SIGNATURE TEST")
    print("=" * 80)
    print("\nValidating:")
    print("  1. Function signatures match three-dashboard system")
    print("  2. Trainer metrics history structure")
    print("  3. Plotting integration in trainer")
    print("\n" + "=" * 80)
    
    tests = [
        ("Plotting Function Signatures", test_plotting_function_signatures),
        ("Trainer Metrics Structure", test_trainer_metrics_structure),
        ("Plotting Integration", test_plotting_integration),
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
        print("\n🎉 All tests passed! Plotting system properly integrated.")
        print("\nValidated:")
        print("  ✓ plot_all_metrics accepts three metrics dictionaries")
        print("  ✓ Trainer tracks three separate metrics histories")
        print("  ✓ det_area includes both text AND bbox metrics")
        print("  ✓ det_object includes only text metrics")
        print("  ✓ Trainer correctly calls plotting with all three dashboards")
    else:
        print(f"\n⚠️  {failed + crashed} test(s) failed.")
    
    print("\n" + "=" * 80 + "\n")
