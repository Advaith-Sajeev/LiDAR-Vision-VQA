"""Test plotting system with three dashboards

Verifies that plot_all_metrics correctly handles:
1. Caption dashboard plots
2. Grounding det_area dashboard plots (text + bbox)
3. Grounding det_object dashboard plots (text only)
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from plotting import plot_all_metrics, plot_metric_curves


def test_three_dashboard_plotting():
    """Test that all three dashboards generate plots correctly"""
    
    print("=" * 80)
    print("TEST: Three-Dashboard Plotting System")
    print("=" * 80)
    
    # Create temporary directory for test plots
    test_dir = Path(tempfile.mkdtemp(prefix="test_plots_"))
    
    try:
        # Mock metric histories
        epochs = [1, 2, 3, 4, 5]
        
        caption_metrics = {
            "bleu4": [0.15, 0.18, 0.21, 0.23, 0.25],
            "cider": [0.45, 0.52, 0.58, 0.63, 0.67],
            "spice": [0.12, 0.15, 0.17, 0.19, 0.21],
            "bertscore_f1": [0.72, 0.75, 0.77, 0.79, 0.81],
        }
        
        grounding_det_area_metrics = {
            "bleu4": [0.10, 0.13, 0.16, 0.18, 0.20],
            "cider": [0.35, 0.42, 0.48, 0.53, 0.57],
            "spice": [0.08, 0.11, 0.13, 0.15, 0.17],
            "bertscore_f1": [0.65, 0.68, 0.71, 0.73, 0.75],
            "top1_accuracy": [45.0, 52.0, 58.0, 63.0, 67.0],
            "bev_iou": [0.35, 0.42, 0.48, 0.53, 0.57],
        }
        
        grounding_det_object_metrics = {
            "bleu4": [0.22, 0.25, 0.28, 0.30, 0.32],
            "cider": [0.55, 0.61, 0.66, 0.70, 0.74],
            "spice": [0.18, 0.21, 0.23, 0.25, 0.27],
            "bertscore_f1": [0.78, 0.80, 0.82, 0.84, 0.85],
        }
        
        # Generate plots
        print("\n✓ Generating plots for three dashboards...")
        plot_all_metrics(
            caption_metrics,
            grounding_det_area_metrics,
            grounding_det_object_metrics,
            epochs,
            test_dir
        )
        
        # Verify plots were created
        metrics_dir = test_dir / "metrics"
        assert metrics_dir.exists(), "Metrics directory not created"
        
        plot_files = list(metrics_dir.glob("*.png"))
        print(f"\n✓ Generated {len(plot_files)} plot files:")
        
        # Expected plots
        expected_plots = {
            "caption": ["bleu4", "cider", "spice", "bertscore_f1", "metrics_combined"],
            "grounding_det_area": ["bleu4", "cider", "spice", "bertscore_f1", "top1_accuracy", "bev_iou", "metrics_combined"],
            "grounding_det_object": ["bleu4", "cider", "spice", "bertscore_f1", "metrics_combined"],
        }
        
        # Count plots by dashboard
        caption_plots = [f for f in plot_files if f.name.startswith("caption_")]
        det_area_plots = [f for f in plot_files if f.name.startswith("grounding_det_area_")]
        det_object_plots = [f for f in plot_files if f.name.startswith("grounding_det_object_")]
        
        print(f"\n  Caption plots: {len(caption_plots)}")
        for p in sorted(caption_plots):
            print(f"    - {p.name}")
        
        print(f"\n  Grounding det_area plots: {len(det_area_plots)}")
        for p in sorted(det_area_plots):
            print(f"    - {p.name}")
        
        print(f"\n  Grounding det_object plots: {len(det_object_plots)}")
        for p in sorted(det_object_plots):
            print(f"    - {p.name}")
        
        # Verify expected counts
        assert len(caption_plots) == len(expected_plots["caption"]), \
            f"Expected {len(expected_plots['caption'])} caption plots, got {len(caption_plots)}"
        
        assert len(det_area_plots) == len(expected_plots["grounding_det_area"]), \
            f"Expected {len(expected_plots['grounding_det_area'])} det_area plots, got {len(det_area_plots)}"
        
        assert len(det_object_plots) == len(expected_plots["grounding_det_object"]), \
            f"Expected {len(expected_plots['grounding_det_object'])} det_object plots, got {len(det_object_plots)}"
        
        print("\n✓ All expected plots generated successfully!")
        
        # Verify plot naming
        print("\n✓ Verifying plot naming conventions:")
        print("  ✓ Caption plots use prefix 'caption_'")
        print("  ✓ det_area plots use prefix 'grounding_det_area_'")
        print("  ✓ det_object plots use prefix 'grounding_det_object_'")
        
        # Verify det_area has both text and bbox metrics
        print("\n✓ Verifying metric types:")
        det_area_metric_names = [p.stem.replace("grounding_det_area_", "") for p in det_area_plots]
        has_text_metrics = any(m in det_area_metric_names for m in ["bleu4", "cider", "spice", "bertscore_f1"])
        has_bbox_metrics = any(m in det_area_metric_names for m in ["top1_accuracy", "bev_iou"])
        
        assert has_text_metrics, "det_area should have text quality metrics"
        assert has_bbox_metrics, "det_area should have bbox accuracy metrics"
        print("  ✓ det_area has both text quality AND bbox accuracy plots")
        
        # Verify det_object has only text metrics
        det_object_metric_names = [p.stem.replace("grounding_det_object_", "") for p in det_object_plots]
        has_only_text = all(m in ["bleu4", "cider", "spice", "bertscore_f1", "metrics_combined"] 
                           for m in det_object_metric_names)
        
        assert has_only_text, "det_object should have only text quality metrics"
        print("  ✓ det_object has only text quality plots (no bbox)")
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        
        return True
        
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\n✓ Cleaned up test directory: {test_dir}")


def test_metric_curve_formats():
    """Test that individual and combined plots are generated correctly"""
    
    print("\n" + "=" * 80)
    print("TEST: Individual and Combined Metric Plots")
    print("=" * 80)
    
    test_dir = Path(tempfile.mkdtemp(prefix="test_curves_"))
    
    try:
        epochs = [1, 2, 3]
        
        # Test with multiple metrics (should create individual + combined)
        multiple_metrics = {
            "bleu4": [0.1, 0.2, 0.3],
            "cider": [0.4, 0.5, 0.6],
        }
        
        print("\n✓ Testing with multiple metrics...")
        plot_metric_curves(multiple_metrics, epochs, test_dir, "test_multi")
        
        plots = list(test_dir.glob("test_multi_*.png"))
        print(f"  Generated {len(plots)} plots:")
        for p in plots:
            print(f"    - {p.name}")
        
        # Should have: bleu4, cider, and combined
        assert len(plots) == 3, f"Expected 3 plots (2 individual + 1 combined), got {len(plots)}"
        
        has_combined = any("combined" in p.name for p in plots)
        assert has_combined, "Should have combined plot"
        print("  ✓ Individual plots + combined plot generated")
        
        # Clean up for next test
        for p in plots:
            p.unlink()
        
        # Test with single metric (should create only individual plot, no combined)
        single_metric = {
            "bleu4": [0.1, 0.2, 0.3],
        }
        
        print("\n✓ Testing with single metric...")
        plot_metric_curves(single_metric, epochs, test_dir, "test_single")
        
        plots = list(test_dir.glob("test_single_*.png"))
        print(f"  Generated {len(plots)} plots:")
        for p in plots:
            print(f"    - {p.name}")
        
        # Should have only 1 plot (no combined needed for single metric)
        assert len(plots) == 1, f"Expected 1 plot for single metric, got {len(plots)}"
        print("  ✓ Only individual plot generated (no combined for single metric)")
        
        print("\n" + "=" * 80)
        print("✓ METRIC CURVE TESTS PASSED")
        print("=" * 80)
        
        return True
        
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("THREE-DASHBOARD PLOTTING SYSTEM TEST SUITE")
    print("=" * 80)
    print("\nTesting:")
    print("  1. Three separate dashboard plots (caption, det_area, det_object)")
    print("  2. Correct metric types per dashboard")
    print("  3. Individual and combined plot generation")
    print("\n" + "=" * 80)
    
    tests = [
        ("Three-Dashboard Plotting", test_three_dashboard_plotting),
        ("Metric Curve Formats", test_metric_curve_formats),
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
        print("\n🎉 All tests passed! Plotting system supports three dashboards.")
        print("\nKey Features:")
        print("  ✓ Separate plots for caption, det_area, and det_object")
        print("  ✓ det_area includes both text quality AND bbox accuracy plots")
        print("  ✓ det_object includes only text quality plots")
        print("  ✓ Individual metric plots + combined plots generated")
        print("  ✓ Proper naming conventions for all dashboards")
    else:
        print(f"\n⚠️  {failed + crashed} test(s) failed.")
    
    print("\n" + "=" * 80 + "\n")
