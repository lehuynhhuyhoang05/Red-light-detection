"""
Test cả 2 models local (YOLOv11 + RT-DETR)
Không cần API, chạy inference trực tiếp
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config


def test_yolov11(config):
    """Test YOLOv11 model"""
    print("\n" + "="*60)
    print("🔵 TESTING YOLOv11")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        
        weights = config['model']['yolov11']['weights']
        weights_path = Path(__file__).parent.parent / weights
        
        if not weights_path.exists():
            print(f"❌ Weights not found: {weights_path}")
            return None
        
        print(f"📂 Loading weights: {weights_path}")
        print(f"📊 File size: {weights_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Load model
        start = time.time()
        model = YOLO(str(weights_path))
        load_time = time.time() - start
        print(f"⏱️  Load time: {load_time:.2f}s")
        
        # Get model info
        print(f"\n📋 Model Info:")
        print(f"   - Task: {model.task}")
        print(f"   - Names: {model.names}")
        
        # Create test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Inference test
        print(f"\n🔍 Running inference...")
        start = time.time()
        results = model(test_img, verbose=False)
        inference_time = (time.time() - start) * 1000
        print(f"⏱️  Inference time: {inference_time:.2f}ms")
        
        # Check classes
        print(f"\n✅ YOLOv11 ready!")
        print(f"   Classes: {list(model.names.values())}")
        
        return {
            'status': 'OK',
            'load_time': load_time,
            'inference_time': inference_time,
            'classes': list(model.names.values()),
            'model_size_mb': weights_path.stat().st_size / 1024 / 1024
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'status': 'ERROR', 'error': str(e)}


def test_rtdetr(config):
    """Test RT-DETR model"""
    print("\n" + "="*60)
    print("🟢 TESTING RT-DETR")
    print("="*60)
    
    try:
        from ultralytics import RTDETR
        
        weights = config['model']['rt_detr']['weights']
        weights_path = Path(__file__).parent.parent / weights
        
        if not weights_path.exists():
            print(f"❌ Weights not found: {weights_path}")
            return None
        
        print(f"📂 Loading weights: {weights_path}")
        print(f"📊 File size: {weights_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Load model
        start = time.time()
        model = RTDETR(str(weights_path))
        load_time = time.time() - start
        print(f"⏱️  Load time: {load_time:.2f}s")
        
        # Get model info
        print(f"\n📋 Model Info:")
        print(f"   - Task: {model.task}")
        print(f"   - Names: {model.names}")
        
        # Create test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Inference test
        print(f"\n🔍 Running inference...")
        start = time.time()
        results = model(test_img, verbose=False)
        inference_time = (time.time() - start) * 1000
        print(f"⏱️  Inference time: {inference_time:.2f}ms")
        
        # Check classes
        print(f"\n✅ RT-DETR ready!")
        print(f"   Classes: {list(model.names.values())}")
        
        return {
            'status': 'OK',
            'load_time': load_time,
            'inference_time': inference_time,
            'classes': list(model.names.values()),
            'model_size_mb': weights_path.stat().st_size / 1024 / 1024
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'status': 'ERROR', 'error': str(e)}


def main():
    print("="*60)
    print("🚦 RED LIGHT VIOLATION DETECTION - MODEL TEST")
    print("="*60)
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    
    print(f"\n📁 Config: {config_path}")
    print(f"📁 Models folder: {Path(__file__).parent.parent / 'models'}")
    
    # Test both models
    results = {}
    
    results['yolov11'] = test_yolov11(config)
    results['rtdetr'] = test_rtdetr(config)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<15} {'Status':<10} {'Size (MB)':<12} {'Load (s)':<10} {'Inference (ms)':<15}")
    print("-" * 65)
    
    for name, info in results.items():
        if info and info.get('status') == 'OK':
            print(f"{name:<15} {'✅ OK':<10} {info['model_size_mb']:<12.2f} {info['load_time']:<10.2f} {info['inference_time']:<15.2f}")
        else:
            error = info.get('error', 'Unknown') if info else 'Not found'
            print(f"{name:<15} {'❌ FAIL':<10} {'-':<12} {'-':<10} {error:<15}")
    
    # Check if both models work
    both_ok = all(r and r.get('status') == 'OK' for r in results.values())
    
    print("\n" + "="*60)
    if both_ok:
        print("✅ CẢ HAI MODELS ĐỀU SẴN SÀNG!")
        print("   Bạn có thể tiến hành xây dựng GUI và chạy so sánh.")
    else:
        print("⚠️  MỘT HOẶC CẢ HAI MODELS CÓ VẤN ĐỀ!")
        print("   Kiểm tra lại file weights và dependencies.")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
