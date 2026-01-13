"""
Test hệ thống hoàn chỉnh với ảnh thật
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.detector import create_detector, CLASS_NAMES
from src.tracker import ObjectTracker
from src.violation_logic import ViolationDetector


def test_with_real_image():
    """Test detection với ảnh thật từ dataset"""
    print("="*60)
    print("🚦 TEST HỆ THỐNG VỚI ẢNH THẬT")
    print("="*60)
    
    # Config
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Force YOLOv11 (đang hoạt động)
    config['model']['type'] = 'yolov11'
    
    print(f"\n📁 Config: {config_path}")
    print(f"🤖 Model: {config['model']['type']}")
    
    # Find test images
    test_images_dir = Path(__file__).parent.parent / "data" / "red_light_violation_dataset" / "test" / "images"
    
    if not test_images_dir.exists():
        # Try alternative path
        test_images_dir = Path(__file__).parent.parent / "data" / "test" / "images"
    
    if not test_images_dir.exists():
        print(f"❌ Test images not found at {test_images_dir}")
        print("💡 Tạo ảnh test giả để demo...")
        
        # Create dummy test
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 200), (200, 350), (255, 255, 255), -1)  # Fake car
        cv2.circle(test_img, (500, 50), 20, (0, 0, 255), -1)  # Fake red light
        cv2.line(test_img, (0, 400), (640, 400), (0, 255, 0), 3)  # Fake stop line
        
        test_images = [test_img]
        print("✅ Created dummy test image")
    else:
        test_images_files = list(test_images_dir.glob("*.jpg"))[:5]
        if not test_images_files:
            print(f"❌ No .jpg files found in {test_images_dir}")
            return
        
        print(f"✅ Found {len(test_images_files)} test images")
        test_images = [cv2.imread(str(f)) for f in test_images_files]
    
    # Initialize components
    print("\n⏳ Initializing detector...")
    
    try:
        detector = create_detector(config)
        print("✅ Detector created")
    except Exception as e:
        print(f"❌ Failed to create detector: {e}")
        return
    
    # Initialize tracker
    tracker = ObjectTracker(config)
    print("✅ Tracker created")
    
    # Initialize violation detector
    violation_detector = ViolationDetector(config)
    violation_detector.set_stop_line_manual(400)
    print("✅ Violation detector created")
    
    # Test each image
    print("\n" + "="*60)
    print("🔍 RUNNING DETECTION")
    print("="*60)
    
    total_detections = 0
    detection_counts = {cls: 0 for cls in CLASS_NAMES.values()}
    
    for i, img in enumerate(test_images[:5]):
        if img is None:
            continue
        
        print(f"\n📷 Image {i+1}:")
        print(f"   Shape: {img.shape}")
        
        # Detect
        detections = detector.detect(img)
        print(f"   Detections: {len(detections)}")
        
        for det in detections:
            print(f"   - {det.class_name}: {det.confidence:.2%} at {det.bbox}")
            detection_counts[det.class_name] = detection_counts.get(det.class_name, 0) + 1
            total_detections += 1
        
        # Draw and save result
        if len(detections) > 0:
            annotated = detector.draw_detections(img, detections)
            output_path = Path(__file__).parent.parent / f"test_output_{i+1}.jpg"
            cv2.imwrite(str(output_path), annotated)
            print(f"   💾 Saved: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"\nTotal detections: {total_detections}")
    print("\nBy class:")
    for cls, count in detection_counts.items():
        if count > 0:
            print(f"   • {cls}: {count}")
    
    print("\n✅ TEST COMPLETE!")
    print("="*60)


def test_full_pipeline_simulation():
    """Test full pipeline với simulation"""
    print("\n" + "="*60)
    print("🎬 TEST FULL PIPELINE (SIMULATION)")
    print("="*60)
    
    # Config
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    config['model']['type'] = 'yolov11'
    
    # Initialize
    detector = create_detector(config)
    tracker = ObjectTracker(config)
    violation_detector = ViolationDetector(config)
    violation_detector.set_stop_line_manual(400)
    
    print("\n✅ All components initialized")
    print("✅ Stop line set at Y=400")
    
    # Simulate 100 frames
    print("\n⏳ Simulating 100 frames...")
    
    for frame_num in range(100):
        # Create dummy frame
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Detect (will return empty for blank frame)
        detections = detector.detect(frame)
        
        # Track
        tracked = tracker.update(detections)
        
        # Check violations
        violations = violation_detector.update(
            tracked, detections, frame, frame_num, datetime.now()
        )
    
    # Stats
    stats = violation_detector.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Frames processed: {stats['frames_processed']}")
    print(f"   Violations: {stats['total_violations']}")
    
    print("\n✅ FULL PIPELINE TEST COMPLETE!")


if __name__ == "__main__":
    test_with_real_image()
    test_full_pipeline_simulation()
