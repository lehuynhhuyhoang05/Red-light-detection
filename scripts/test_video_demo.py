"""
Test model trên video và visualize kết quả
Để kiểm tra xem model có detect red_light trong video demo không
"""

import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from roboflow import Roboflow
from src.detector import RoboflowDetector, Detection

# Roboflow config
API_KEY = "1lyMl95ObWTqCvgFpxxt"
WORKSPACE = "huyhoang"
PROJECT = "red-light-violation-detect-hecrg"
VERSION = 1

# Video path - thay đổi path này
VIDEO_PATH = r"data\videos\demo_video.mp4"  # Bạn cần download video demo vào đây

def test_video(video_path, confidence=40, max_frames=100):
    """Test model trên video"""
    
    # Check video exists
    if not Path(video_path).exists():
        print(f"❌ Video không tồn tại: {video_path}")
        print("💡 Bạn cần download video demo vào thư mục data/videos/")
        return
    
    # Initialize detector
    print("📡 Loading Roboflow model...")
    detector = RoboflowDetector(API_KEY, WORKSPACE, PROJECT, VERSION)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Testing first {min(max_frames, total_frames)} frames")
    
    # Statistics
    from collections import defaultdict
    class_counts = defaultdict(int)
    frames_with_red_light = 0
    frames_with_vehicles = 0
    total_detections = 0
    
    # Setup video writer
    output_path = "data/videos/demo_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n🎬 Processing video... (Confidence: {confidence}%)")
    print("="*80)
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Detect every 10 frames to save API quota
        if frame_idx % 10 == 0:
            print(f"\n[Frame {frame_idx}/{min(max_frames, total_frames)}]", end=" ")
            
            # Run detection
            detections = detector.detect(frame, confidence=confidence)
            total_detections += len(detections)
            
            if detections:
                print(f"✅ {len(detections)} objects:", end=" ")
                
                has_red = False
                has_vehicle = False
                
                for det in detections:
                    class_counts[det.class_name] += 1
                    print(f"{det.class_name}({det.confidence:.0%})", end=" ")
                    
                    if det.class_name == 'red_light':
                        has_red = True
                    if det.class_name in ['car', 'motobike']:
                        has_vehicle = True
                
                if has_red:
                    frames_with_red_light += 1
                if has_vehicle:
                    frames_with_vehicles += 1
                
                # Draw on frame
                frame = detector.draw_detections(frame, detections)
            else:
                print("❌ No detections")
        
        # Write frame
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 VIDEO TEST SUMMARY")
    print("="*80)
    print(f"Frames processed: {frame_idx}")
    print(f"Frames tested: {frame_idx // 10}")
    print(f"Total detections: {total_detections}")
    print(f"\n🔴 Frames with red_light: {frames_with_red_light} ({frames_with_red_light/(frame_idx//10)*100:.1f}%)")
    print(f"🚗 Frames with vehicles: {frames_with_vehicles} ({frames_with_vehicles/(frame_idx//10)*100:.1f}%)")
    
    print(f"\n📈 Detections by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    
    print(f"\n💾 Output video saved: {output_path}")
    
    # Diagnosis
    if frames_with_red_light == 0:
        print("\n⚠️  WARNING: Không detect được red_light trong video!")
        print("   Thử:")
        print("   1. Giảm confidence threshold (hiện tại: 40%)")
        print("   2. Kiểm tra video có đèn đỏ không")
        print("   3. Dùng video khác từ YouTube")
    else:
        print(f"\n✅ Model detect được red_light! ({frames_with_red_light} frames)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=VIDEO_PATH, help='Path to video file')
    parser.add_argument('--confidence', type=int, default=40, help='Confidence threshold (0-100)')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames to test')
    args = parser.parse_args()
    
    test_video(args.video, args.confidence, args.max_frames)
