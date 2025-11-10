"""Quick test to verify bounding box extraction from grounding answers"""

from metrics import (
    extract_bbox_from_text, 
    extract_object_class,
    calculate_bev_iou_2d,
    calculate_bbox_iou_3d,
    calculate_grounding_metrics
)

# Test data from user's examples
test_samples = [
    {
        "text": "There is a traffic_cone at the location [-16.33,-16.03,-1.32,-1.03,-1.61,-0.87,1.36].",
        "expected_bbox": [-16.33, -16.03, -1.32, -1.03, -1.61, -0.87, 1.36],
        "expected_class": "traffic_cone"
    },
    {
        "text": "There is a truck at the location [-11.38,-9.07,15.7,23.22,-1.51,1.58,-0.96].",
        "expected_bbox": [-11.38, -9.07, 15.7, 23.22, -1.51, 1.58, -0.96],
        "expected_class": "truck"
    },
    {
        "text": "The 2 cars are located at [[8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93],[-0.55,1.32,-15.47,-10.99,-1.94,-0.49,-2.66]].",
        "expected_bbox": [8.4, 10.03, -7.7, -3.45, -1.8, -0.36, -1.93],  # Should extract first
        "expected_class": "car"
    }
]

def test_bbox_extraction():
    """Test bounding box extraction"""
    print("=" * 60)
    print("TESTING BOUNDING BOX EXTRACTION")
    print("=" * 60)
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest {i}:")
        print(f"Text: {sample['text'][:80]}...")
        
        bbox = extract_bbox_from_text(sample['text'])
        obj_class = extract_object_class(sample['text'])
        
        print(f"Extracted bbox: {bbox}")
        print(f"Expected bbox:  {sample['expected_bbox']}")
        print(f"Match: {bbox == sample['expected_bbox']}")
        
        print(f"Extracted class: {obj_class}")
        print(f"Expected class:  {sample['expected_class']}")
        print(f"Match: {obj_class == sample['expected_class']}")


def test_iou_calculation():
    """Test IoU calculations"""
    print("\n" + "=" * 60)
    print("TESTING IOU CALCULATIONS")
    print("=" * 60)
    
    # Perfect match
    box1 = [8.4, 10.03, -7.7, -3.45, -1.8, -0.36, -1.93]
    box2 = [8.4, 10.03, -7.7, -3.45, -1.8, -0.36, -1.93]
    
    bev_iou = calculate_bev_iou_2d(box1, box2)
    iou_3d = calculate_bbox_iou_3d(box1, box2)
    
    print(f"\nPerfect Match Test:")
    print(f"Box1: {box1}")
    print(f"Box2: {box2}")
    print(f"BEV IoU: {bev_iou:.4f} (expected: 1.0000)")
    print(f"3D IoU:  {iou_3d:.4f} (expected: 1.0000)")
    
    # No overlap
    box3 = [0, 2, 0, 2, 0, 2, 0]
    box4 = [10, 12, 10, 12, 10, 12, 0]
    
    bev_iou = calculate_bev_iou_2d(box3, box4)
    iou_3d = calculate_bbox_iou_3d(box3, box4)
    
    print(f"\nNo Overlap Test:")
    print(f"Box3: {box3}")
    print(f"Box4: {box4}")
    print(f"BEV IoU: {bev_iou:.4f} (expected: 0.0000)")
    print(f"3D IoU:  {iou_3d:.4f} (expected: 0.0000)")
    
    # Partial overlap
    box5 = [0, 4, 0, 4, 0, 2, 0]
    box6 = [2, 6, 2, 6, 0, 2, 0]
    
    bev_iou = calculate_bev_iou_2d(box5, box6)
    iou_3d = calculate_bbox_iou_3d(box5, box6)
    
    print(f"\nPartial Overlap Test:")
    print(f"Box5: {box5}")
    print(f"Box6: {box6}")
    print(f"BEV IoU: {bev_iou:.4f}")
    print(f"3D IoU:  {iou_3d:.4f}")
    
    # Calculate expected IoU manually
    # Box5: 4x4 area = 16, Box6: 4x4 area = 16
    # Intersection: 2x2 = 4
    # Union: 16 + 16 - 4 = 28
    # IoU = 4/28 = 0.1429
    print(f"Expected BEV IoU: {4/28:.4f}")


def test_grounding_metrics():
    """Test full grounding metrics pipeline"""
    print("\n" + "=" * 60)
    print("TESTING GROUNDING METRICS PIPELINE")
    print("=" * 60)
    
    predictions = [
        "There is a car at the location [8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93].",
        "There is a truck at the location [-11.38,-9.07,15.7,23.22,-1.51,1.58,-0.96].",
        "There is a pedestrian at the location [-16.1,-15.36,-1.1,-0.53,-1.55,0.16,1.36]."
    ]
    
    references = [
        "There is a car at the location [8.4,10.03,-7.7,-3.45,-1.8,-0.36,-1.93].",
        "There is a car at the location [-11.38,-9.07,15.7,23.22,-1.51,1.58,-0.96].",  # Wrong class
        "There is a pedestrian at the location [-16.0,-15.0,-1.0,-0.5,-1.5,0.2,1.4]."   # Close but not exact
    ]
    
    metrics = calculate_grounding_metrics(predictions, references)
    
    print(f"\nMetrics:")
    print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"  BEV IoU:        {metrics['bev_iou']:.4f}")
    print(f"  3D IoU:         {metrics['iou_3d']:.4f}")
    print(f"  Valid samples:  {metrics['valid_samples']}/{metrics['total_samples']}")
    
    print(f"\nExpected:")
    print(f"  Top-1 Accuracy: 66.67% (2 out of 3: car matches, truck doesn't, pedestrian matches)")
    print(f"  BEV IoU:        ~0.67 (1.0 + 1.0 + ~0.0) / 3")


if __name__ == "__main__":
    test_bbox_extraction()
    test_iou_calculation()
    test_grounding_metrics()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
