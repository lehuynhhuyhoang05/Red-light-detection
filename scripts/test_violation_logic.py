"""
Test script for Violation Detection Logic
Kiểm tra các trường hợp vi phạm và không vi phạm
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, List
from collections import deque

# Mock classes for testing (không cần import thật)
@dataclass
class MockDetection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int] = (0, 0)
    class_id: int = 0
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))


@dataclass 
class MockTrackedObject:
    track_id: int
    detection: MockDetection
    trajectory: List[tuple] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = [self.detection.center]


def test_violation_logic():
    """
    Test các scenarios vi phạm và không vi phạm
    """
    dummy_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
    print("=" * 60)
    print("TEST VIOLATION DETECTION LOGIC")
    print("=" * 60)
    
    # Import after adding to path
    from src.violation_logic import ViolationDetector, VEHICLE_CLASSES
    
    # Config
    config = {
        'violation': {
            'grace_period': 1.5,
            'min_frames': 3,
            'stop_line_threshold': 10
        },
        'location': {
            'intersection': 'Test Intersection',
            'camera_id': 'TEST_CAM'
        }
    }
    
    detector = ViolationDetector(config)
    
    # Stop line at Y=400
    STOP_LINE_Y = 400
    detector.set_stop_line_manual(STOP_LINE_Y)
    
    print(f"\n📍 Stop line set at Y={STOP_LINE_Y}")
    print(f"⏱️  Grace period: {detector.grace_period}s")
    print(f"🎯 Min frames: {detector.min_frames}")
    print()
    
    # ========================================
    # SCENARIO 1: VI PHẠM THỰC SỰ
    # ========================================
    print("-" * 40)
    print("SCENARIO 1: Vi phạm thực sự")
    print("  - Xe ở trước vạch khi đèn đỏ")
    print("  - Xe vượt qua vạch sau grace period")
    print("-" * 40)
    
    # Reset
    detector.reset()
    base_time = datetime.now()
    
    # Frame 0-20: Đèn xanh, xe đang tiến gần vạch
    frame = 0
    timestamp = base_time
    vehicle_y = 350  # Trước vạch (Y < 400)
    
    vehicle_det = MockDetection('car', 0.9, (100, 300, 200, vehicle_y))
    vehicle = MockTrackedObject(track_id=1, detection=vehicle_det)
    light_det = MockDetection('green_light', 0.95, (500, 50, 530, 80))
    
    for i in range(10):
        violations = detector.update(
            tracked_vehicles=[vehicle],
            detections=[light_det, vehicle_det],
            frame=dummy_frame,
            frame_number=i,
            timestamp=timestamp + timedelta(milliseconds=33 * i)
        )
    print(f"  Frame 0-9: Đèn XANH, xe Y={vehicle_y}, violations={len(violations)}")
    
    # Frame 10-30: Đèn chuyển đỏ, xe vẫn trước vạch (Y=370)
    # Phải feed nhiều frames với red_light để voting mechanism confirm RED
    frame = 10
    timestamp = base_time + timedelta(seconds=0.5)
    vehicle_y = 370  # Vẫn trước vạch
    
    light_det = MockDetection('red_light', 0.95, (500, 50, 530, 80))
    vehicle_det = MockDetection('car', 0.9, (100, 320, 200, vehicle_y))
    vehicle = MockTrackedObject(track_id=1, detection=vehicle_det)
    vehicle.trajectory = [(150, 350), (150, 360), (150, 365), (150, 368), (150, 370)]
    
    for i in range(20):
        violations = detector.update(
            tracked_vehicles=[vehicle],
            detections=[light_det, vehicle_det],
            frame=dummy_frame,
            frame_number=10 + i,
            timestamp=timestamp + timedelta(milliseconds=33 * i)
        )
    print(f"  Frame 10-29: Đèn ĐỎ bắt đầu, xe Y={vehicle_y} (trước vạch), violations={len(violations)}")
    assert len(violations) == 0, "No violation yet - still before line"
    
    # Frame 90+: Sau grace period (3 giây), xe vượt qua vạch
    frame = 90
    timestamp = base_time + timedelta(seconds=3)
    vehicle_y = 450  # Đã vượt vạch
    
    light_det = MockDetection('red_light', 0.95, (500, 50, 530, 80))
    vehicle_det = MockDetection('car', 0.9, (100, 400, 200, vehicle_y))
    vehicle = MockTrackedObject(track_id=1, detection=vehicle_det)
    # Trajectory rõ ràng: từ trước vạch (370) đến sau vạch (450)
    vehicle.trajectory = [(150, 370), (150, 390), (150, 410), (150, 430), (150, 450)]
    
    # Feed nhiều frames để confirm violation
    all_violations = []
    for i in range(10):
        violations_this_frame = detector.update(
            tracked_vehicles=[vehicle],
            detections=[light_det, vehicle_det],
            frame=dummy_frame,
            frame_number=frame + i,
            timestamp=timestamp + timedelta(milliseconds=33 * i)
        )
        if violations_this_frame:
            all_violations.extend(violations_this_frame)
    
    print(f"  Frame {frame}-{frame+9}: Xe vượt vạch Y={vehicle_y} sau grace period")
    print(f"  Violations detected: {len(all_violations)}")
    
    if len(all_violations) > 0:
        print(f"  ✅ VI PHẠM ĐƯỢC GHI NHẬN!")
        print(f"     - Violation ID: {all_violations[0].violation_id}")
        print(f"     - Track ID: {all_violations[0].track_id}")
        print(f"     - Evidence frames: {len(all_violations[0].evidence_frames)}")
    else:
        print(f"  ❌ KHÔNG GHI NHẬN VI PHẠM trong return list")
        print(f"     Debug: Total violations stored: {len(detector.violations)}")
        if detector.violations:
            for vid, v in list(detector.violations.items())[:1]:
                print(f"     Stored violation: {v.violation_id} (Track {vid})")
    
    print()
    
    # ========================================
    # SCENARIO 2: XE ĐÃ QUA VẠCH KHI ĐÈN ĐỎ - KHÔNG VI PHẠM
    # ========================================
    print("-" * 40)
    print("SCENARIO 2: Xe đã qua vạch khi đèn đỏ (KHÔNG vi phạm)")
    print("  - Xe đã ở sau vạch khi đèn chuyển đỏ")
    print("  - Đang đi qua hợp lệ")
    print("-" * 40)
    
    detector.reset()
    base_time = datetime.now()
    
    # Frame 0: Đèn xanh, xe ĐÃ ở sau vạch (Y=450)
    frame = 0
    timestamp = base_time
    
    vehicle_det = MockDetection('car', 0.9, (100, 400, 200, 450))  # Y=450 > 400
    vehicle = MockTrackedObject(track_id=2, detection=vehicle_det)
    light_det = MockDetection('green_light', 0.95, (500, 50, 530, 80))
    
    violations = detector.update(
        tracked_vehicles=[vehicle],
        detections=[light_det, vehicle_det],
        frame=dummy_frame,
        frame_number=frame,
        timestamp=timestamp
    )
    print(f"  Frame {frame}: Đèn XANH, xe Y=450 (đã qua vạch), violations={len(violations)}")
    
    # Frame 30: Đèn chuyển đỏ
    frame = 30
    timestamp = base_time + timedelta(seconds=1)
    
    light_det = MockDetection('red_light', 0.95, (500, 50, 530, 80))
    vehicle_det = MockDetection('car', 0.9, (100, 450, 200, 500))  # Tiếp tục đi
    vehicle = MockTrackedObject(track_id=2, detection=vehicle_det)
    
    violations = detector.update(
        tracked_vehicles=[vehicle],
        detections=[light_det, vehicle_det],
        frame=dummy_frame,
        frame_number=frame,
        timestamp=timestamp
    )
    print(f"  Frame {frame}: Đèn ĐỎ, xe tiếp tục đi Y=500, violations={len(violations)}")
    
    assert len(violations) == 0, "Should NOT have violation - car was already past line"
    print(f"  ✅ ĐÚNG: Không ghi nhận vi phạm (xe đang đi hợp lệ)")
    print()
    
    # ========================================
    # SCENARIO 3: GRACE PERIOD - KHÔNG VI PHẠM
    # ========================================
    print("-" * 40)
    print("SCENARIO 3: Xe vượt trong Grace Period (KHÔNG vi phạm)")
    print("  - Xe vượt ngay sau khi đèn đỏ")
    print("  - Trong 1.5 giây grace period")
    print("-" * 40)
    
    detector.reset()
    base_time = datetime.now()
    
    # Frame 0: Đèn chuyển đỏ
    frame = 0
    timestamp = base_time
    
    vehicle_det = MockDetection('car', 0.9, (100, 350, 200, 390))  # Trước vạch
    vehicle = MockTrackedObject(track_id=3, detection=vehicle_det)
    light_det = MockDetection('red_light', 0.95, (500, 50, 530, 80))
    
    violations = detector.update(
        tracked_vehicles=[vehicle],
        detections=[light_det, vehicle_det],
        frame=dummy_frame,
        frame_number=frame,
        timestamp=timestamp
    )
    print(f"  Frame {frame}: Đèn ĐỎ bắt đầu, xe Y=390, violations={len(violations)}")
    
    # Frame 15: 0.5 giây sau (trong grace period), xe vượt vạch
    frame = 15
    timestamp = base_time + timedelta(seconds=0.5)  # Trong grace period
    
    vehicle_det = MockDetection('car', 0.9, (100, 400, 200, 450))  # Vượt vạch
    vehicle = MockTrackedObject(track_id=3, detection=vehicle_det)
    vehicle.trajectory = [(150, 390), (150, 400), (150, 420), (150, 440), (150, 450)]
    
    for i in range(5):
        violations = detector.update(
            tracked_vehicles=[vehicle],
            detections=[light_det, vehicle_det],
            frame=dummy_frame,
            frame_number=frame + i,
            timestamp=timestamp + timedelta(milliseconds=33 * i)
        )
    
    print(f"  Frame {frame}: Xe vượt vạch TRONG grace period, violations={len(violations)}")
    
    if len(violations) == 0:
        print(f"  ✅ ĐÚNG: Không ghi nhận vi phạm (trong grace period)")
    else:
        print(f"  ❌ SAI: Ghi nhận vi phạm khi đang trong grace period!")
    
    print()
    
    # ========================================
    # SUMMARY
    # ========================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total violations recorded: {len(detector.violations)}")
    print(f"Total frames processed: {detector.total_frames_processed}")
    print()
    
    stats = detector.get_statistics()
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    test_violation_logic()
