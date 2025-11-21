"""Test validation filtering for grounding questions with coordinates"""

import re
import json
from pathlib import Path


def has_coordinates_in_question(sample):
    """Check if question contains coordinate arrays like [1.2, 3.4, 5.6]"""
    question = sample.get("question", "")
    # Match patterns like [25.67,28.0,32.6,...] in the question
    coord_pattern = r'\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*'
    return bool(re.search(coord_pattern, question))


def load_validation_jsons():
    """Try to load actual validation JSONs if available"""
    
    # Common paths for validation data
    possible_paths = [
        "data/nuscenes/json/nuScenes_grounding_val_v1.0_mini.json",
        "data/nuscenes/json/nuScenes_grounding_val.json",
        "../../../data/nuscenes/json/nuScenes_grounding_val_v1.0_mini.json",
    ]
    
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            print(f"📂 Found validation JSON: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   Loaded {len(data)} samples")
                return data, str(path)
            except Exception as e:
                print(f"   Error loading: {e}")
    
    return None, None


def test_coordinate_filtering():
    """Test that coordinate detection works correctly"""
    
    # Test cases
    test_samples = [
        # Should be FILTERED OUT (contains coordinates)
        {
            "question": "What is at the location [25.67,28.0,32.6,38.06,-0.11,1.91,-1.82]?",
            "template_type": "det_object",
            "expected_filtered": True,
            "reason": "Contains full coordinate array in question"
        },
        {
            "question": "Describe the object at [10.5, 15.2, 20.3]?",
            "template_type": "det_object",
            "expected_filtered": True,
            "reason": "Contains coordinate array with spaces"
        },
        {
            "question": "What vehicle is at [-5.2, 10.8, 3.4, 15.6]?",
            "template_type": "det_object",
            "expected_filtered": True,
            "reason": "Contains negative coordinates"
        },
        
        # Should be KEPT (no coordinates in question)
        {
            "question": "What vehicle is in front of the ego car?",
            "template_type": "det_object",
            "expected_filtered": False,
            "reason": "No coordinates, requires spatial reasoning"
        },
        {
            "question": "Describe the object ahead.",
            "template_type": "det_object",
            "expected_filtered": False,
            "reason": "No coordinates, directional question"
        },
        {
            "question": "What is to the left of the vehicle?",
            "template_type": "det_object",
            "expected_filtered": False,
            "reason": "No coordinates, spatial relationship question"
        },
        {
            "question": "What is the closest object?",
            "template_type": "det_object",
            "expected_filtered": False,
            "reason": "No coordinates, distance-based question"
        },
        
        # Edge cases
        {
            "question": "How many cars are there? I see 3 ahead.",
            "template_type": "det_object",
            "expected_filtered": False,
            "reason": "Contains single number, not coordinates"
        },
        {
            "question": "The car is 10 meters away",
            "template_type": "det_object",
            "expected_filtered": False,
            "reason": "Contains distance, but not coordinate array"
        },
    ]
    
    print("=" * 80)
    print("VALIDATION FILTERING TEST")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, sample in enumerate(test_samples, 1):
        has_coords = has_coordinates_in_question(sample)
        should_filter = sample["expected_filtered"]
        
        test_passed = (has_coords == should_filter)
        
        status = "✓ PASS" if test_passed else "✗ FAIL"
        action = "FILTERED" if has_coords else "KEPT"
        expected_action = "FILTERED" if should_filter else "KEPT"
        
        print(f"\nTest {i}: {status}")
        print(f"  Question: {sample['question']}")
        print(f"  Result: {action} (expected: {expected_action})")
        print(f"  Reason: {sample['reason']}")
        
        if test_passed:
            passed += 1
        else:
            failed += 1
            print(f"  ⚠️  ERROR: Expected {expected_action} but got {action}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(test_samples)} passed, {failed}/{len(test_samples)} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✓ All tests passed! Coordinate filtering is working correctly.")
        return True
    else:
        print("✗ Some tests failed! Check the coordinate detection logic.")
        return False


def test_sample_selection():
    """Test that sample selection picks only valid det_object samples"""
    
    print("\n" + "=" * 80)
    print("SAMPLE SELECTION TEST")
    print("=" * 80)
    
    # Try to load actual validation JSON
    actual_data, data_path = load_validation_jsons()
    
    if actual_data:
        print(f"\n✓ Using ACTUAL validation data from: {data_path}")
        grounding_data = actual_data
        
        # Show sample structure for debugging
        if grounding_data:
            print("\n📋 Sample JSON structure (first sample):")
            sample = grounding_data[0]
            print(json.dumps(sample, indent=2))
            
            print("\n📊 Available fields:")
            for key in sample.keys():
                value = sample[key]
                value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"  - {key}: {type(value).__name__} = {value_preview}")
        
        # Simulate token2path (assume all tokens have BEV features for testing)
        token2path = {s["sample_token"]: f"/path/{s['sample_token']}.npy" for s in grounding_data}
    else:
        print("\n⚠️  No actual validation data found, using FALLBACK synthetic data")
        # Simulate grounding data
        grounding_data = [
            {"sample_token": "tok1", "question": "What is at [1,2,3]?", "template_type": "det_object"},
            {"sample_token": "tok2", "question": "What vehicle is ahead?", "template_type": "det_object"},
            {"sample_token": "tok3", "question": "Describe the car.", "template_type": "det_area"},
            {"sample_token": "tok4", "question": "What is to the left?", "template_type": "det_object"},
            {"sample_token": "tok5", "question": "Object at [5,10,15]?", "template_type": "det_object"},
            {"sample_token": "tok6", "question": "What is the closest object?", "template_type": "det_object"},
        ]
        
        # Simulate token2path (all tokens have BEV features)
        token2path = {s["sample_token"]: f"/path/{s['sample_token']}.npy" for s in grounding_data}
    
    # Apply filtering logic
    grounding_available = [
        s for s in grounding_data 
        if s.get("sample_token") in token2path 
        and s.get("template_type") == "det_object"
        and not has_coordinates_in_question(s)
    ]
    
    print(f"\nOriginal grounding samples: {len(grounding_data)}")
    print(f"After filtering: {len(grounding_available)}")
    
    print("\n📋 Samples KEPT for evaluation (first 5):")
    for i, s in enumerate(grounding_available[:5], 1):
        print(f"  {i}. [{s['sample_token']}] {s['question'][:80]}{'...' if len(s['question']) > 80 else ''}")
    
    if len(grounding_available) > 5:
        print(f"  ... and {len(grounding_available) - 5} more samples")
    
    # Count breakdown
    det_object_all = [s for s in grounding_data if s.get("template_type") == "det_object"]
    det_object_with_coords = [s for s in det_object_all if has_coordinates_in_question(s)]
    det_area_all = [s for s in grounding_data if s.get("template_type") == "det_area"]
    
    print("\n📊 Filtering breakdown:")
    print(f"  Total samples: {len(grounding_data)}")
    print(f"  det_object: {len(det_object_all)}")
    print(f"    - with coordinates in question: {len(det_object_with_coords)} (FILTERED)")
    print(f"    - without coordinates: {len(grounding_available)} (KEPT)")
    print(f"  det_area: {len(det_area_all)} (FILTERED - no bbox for eval)")
    
    # Show examples of filtered samples
    if det_object_with_coords:
        print(f"\n🚫 Examples of FILTERED det_object samples (coordinates in question):")
        for i, s in enumerate(det_object_with_coords[:3], 1):
            print(f"  {i}. {s['question'][:100]}{'...' if len(s['question']) > 100 else ''}")
    
    if actual_data:
        # For real data, just verify we got some samples
        if len(grounding_available) > 0:
            print(f"\n✓ Successfully filtered {len(grounding_available)} valid samples from actual data!")
            return True
        else:
            print(f"\n✗ Warning: No valid samples after filtering!")
            return False
    else:
        # For synthetic data, verify exact matches
        expected_kept = ["tok2", "tok4", "tok6"]  # det_object without coordinates
        actual_kept = [s["sample_token"] for s in grounding_available]
        
        print("\n🔍 Detailed breakdown (synthetic data):")
        for s in grounding_data:
            token = s["sample_token"]
            question = s["question"]
            template = s["template_type"]
            has_coords = has_coordinates_in_question(s)
            
            if template != "det_object":
                reason = f"❌ Wrong template ({template})"
            elif has_coords:
                reason = "❌ Has coordinates in question"
            else:
                reason = "✓ Valid for evaluation"
            
            print(f"  {token}: {reason}")
            print(f"    Q: {question}")
        
        # Verify
        if set(actual_kept) == set(expected_kept):
            print(f"\n✓ Filtering correct! Kept {len(actual_kept)} valid samples.")
            return True
        else:
            print(f"\n✗ Filtering error!")
            print(f"  Expected: {expected_kept}")
            print(f"  Got: {actual_kept}")
            return False


if __name__ == "__main__":
    print("\n🧪 Running validation filtering tests...\n")
    
    test1_pass = test_coordinate_filtering()
    test2_pass = test_sample_selection()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if test1_pass and test2_pass:
        print("✓ All tests passed! The filtering logic is correct.")
        print("\nSummary:")
        print("  - Questions with coordinates [x,y,z,...] are FILTERED OUT")
        print("  - Only det_object questions without coordinates are KEPT")
        print("  - This ensures bbox evaluation tests actual spatial reasoning")
    else:
        print("✗ Some tests failed. Review the filtering logic.")
    
    print("=" * 80)
