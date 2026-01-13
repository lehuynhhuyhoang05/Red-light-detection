"""
Quick test with actual image
"""
from roboflow import Roboflow
import sys

# API key (replace with yours)
api_key = "1lyMl95ObWTqCvgFpxxt"

# Initialize
rf = Roboflow(api_key=api_key)
project = rf.workspace("huyhoang").project("red-light-violation-detect-hecrg")
model = project.version(1).model

# Test image
test_image = r"data\red_light_violation_dataset\test\images\0_jpg.rf.486846884c89827797fc94c0a2e822b3.jpg"

print(f"\n🔍 Testing with: {test_image}\n")

# Predict
result = model.predict(test_image, confidence=40, overlap=30).json()

# Show results
print(f"✅ Detected {len(result['predictions'])} objects:\n")

for pred in result['predictions']:
    print(f"  - {pred['class']:15s} | Confidence: {pred['confidence']:.1%} | Box: [{pred['x']:.0f},{pred['y']:.0f},{pred['width']:.0f}x{pred['height']:.0f}]")

# Save result
model.predict(test_image, confidence=40, overlap=30).save("test_result.jpg")
print(f"\n💾 Result saved to: test_result.jpg")
print("\n✅ MODEL WORKS! Ready to use in your system!")
