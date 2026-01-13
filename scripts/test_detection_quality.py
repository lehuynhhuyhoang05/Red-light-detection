"""
Test model detection quality trên test set
Kiểm tra xem model có detect được red_light, stop_line không
"""

import os
from pathlib import Path
from roboflow import Roboflow
from collections import defaultdict

# Roboflow config
API_KEY = "1lyMl95ObWTqCvgFpxxt"
WORKSPACE = "huyhoang"
PROJECT = "red-light-violation-detect-hecrg"
VERSION = 1

# Test set path
TEST_IMAGES_DIR = r"data\red_light_violation_dataset\test\images"

# Initialize model
print("📡 Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)
model = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).model

# Get random test images
test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))[:20]  # Test 20 ảnh đầu
print(f"🧪 Testing on {len(test_images)} images")

# Statistics
class_counts = defaultdict(int)
total_detections = 0
images_with_red_light = 0
images_with_stop_line = 0
images_with_vehicles = 0

print("\n" + "="*80)
print("TESTING MODEL DETECTION QUALITY")
print("="*80)

for i, img_path in enumerate(test_images, 1):
    print(f"\n[{i}/{len(test_images)}] Testing: {img_path.name}")
    
    # Run prediction
    result = model.predict(str(img_path), confidence=40, overlap=30).json()
    predictions = result.get('predictions', [])
    
    if predictions:
        print(f"  ✅ Detected {len(predictions)} objects:")
        
        # Count classes
        has_red_light = False
        has_stop_line = False
        has_vehicle = False
        
        for pred in predictions:
            class_name = pred['class']
            confidence = pred['confidence']
            class_counts[class_name] += 1
            total_detections += 1
            
            print(f"    - {class_name}: {confidence:.1%}")
            
            if class_name == 'red_light':
                has_red_light = True
            elif class_name == 'stop_line':
                has_stop_line = True
            elif class_name in ['car', 'motobike']:
                has_vehicle = True
        
        if has_red_light:
            images_with_red_light += 1
        if has_stop_line:
            images_with_stop_line += 1
        if has_vehicle:
            images_with_vehicles += 1
    else:
        print(f"  ❌ No detections")

# Print summary
print("\n" + "="*80)
print("📊 DETECTION SUMMARY")
print("="*80)
print(f"Total images tested: {len(test_images)}")
print(f"Total detections: {total_detections}")
print(f"\n🚗 Images with vehicles: {images_with_vehicles}/{len(test_images)} ({images_with_vehicles/len(test_images)*100:.1f}%)")
print(f"🔴 Images with red_light: {images_with_red_light}/{len(test_images)} ({images_with_red_light/len(test_images)*100:.1f}%)")
print(f"🛑 Images with stop_line: {images_with_stop_line}/{len(test_images)} ({images_with_stop_line/len(test_images)*100:.1f}%)")

print(f"\n📈 Detections by class:")
for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {count}")

# Check if red_light detection is working
if images_with_red_light == 0:
    print("\n⚠️  WARNING: Model không detect được red_light trong 20 ảnh test!")
    print("   Có thể nguyên nhân:")
    print("   1. Test set không có đèn đỏ trong 20 ảnh đầu")
    print("   2. Confidence threshold 40% quá cao")
    print("   3. Model cần train thêm")
elif images_with_red_light < 5:
    print(f"\n⚠️  WARNING: Chỉ detect được red_light trong {images_with_red_light} ảnh")
    print("   Model có thể cần cải thiện")
else:
    print(f"\n✅ Model hoạt động tốt! Detect red_light trong {images_with_red_light} ảnh")

print("\n💡 Suggestion: Xem lại ảnh training có đủ red_light không?")
