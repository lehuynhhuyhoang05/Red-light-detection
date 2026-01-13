"""
Violation Detection Logic - IMPROVED VERSION
Detects red light violations based on traffic light state and vehicle behavior

=============================================================================
CORE LOGIC VI PHẠM VƯỢT ĐÈN ĐỎ - CẢI TIẾN
=============================================================================

Điều kiện XÁC NHẬN vi phạm (TẤT CẢ phải đúng):
1. Đèn đang ĐỎ
2. Xe ở TRƯỚC vạch dừng khi đèn chuyển đỏ  
3. Xe VƯỢT QUA vạch sau khi đèn đỏ (có chuyển động - crossing motion)
4. Không phải trong grace period (1-2 giây sau khi đèn đỏ)
5. Chưa ghi nhận trước đó (deduplication)
6. Đủ số frame xác nhận (tránh noise)

Điều kiện KHÔNG VI PHẠM:
- Xe đã ở SAU vạch khi đèn chuyển đỏ (đang đi qua hợp lệ)
- Xe đứng yên sau vạch (không có crossing motion)
- Trong grace period
- Xe ưu tiên (ambulance, police, etc.) - future
=============================================================================
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple, Deque
from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger

from .tracker import TrackedObject, TrajectoryAnalyzer
from .detector import Detection


# ============================================================================
# CONSTANTS - Các hằng số cấu hình
# ============================================================================

# Vehicle classes có thể vi phạm (phải khớp với model classes)
VEHICLE_CLASSES = {'car', 'motobike'}

# Traffic light classes
LIGHT_CLASSES = {'red_light', 'yellow_light', 'green_light'}

# Grace period sau khi đèn chuyển đỏ (giây)
# Không phạt trong thời gian này để tránh oan
DEFAULT_GRACE_PERIOD = 1.5

# Số frame tối thiểu để xác nhận vi phạm (tránh detection noise)
DEFAULT_MIN_FRAMES = 3

# Ngưỡng vượt qua stop line (pixels)
# Xe phải qua vạch ít nhất X pixels mới tính là vi phạm
DEFAULT_STOP_LINE_THRESHOLD = 10

# Độ dài trajectory tối thiểu để detect crossing motion
TRAJECTORY_MIN_LENGTH = 5

# Số frame lưu history cho voting traffic light
LIGHT_STATE_HISTORY_SIZE = 5


# ============================================================================
# DATA CLASSES - Cấu trúc dữ liệu
# ============================================================================

@dataclass
class TrafficLightState:
    """
    Traffic light state với history cho voting mechanism
    Voting giúp tránh flicker khi detection không ổn định
    """
    current_state: str = "UNKNOWN"  # RED, YELLOW, GREEN, UNKNOWN
    confidence: float = 0.0
    state_history: Deque = field(default_factory=lambda: deque(maxlen=LIGHT_STATE_HISTORY_SIZE))
    last_change_time: Optional[datetime] = None
    red_start_time: Optional[datetime] = None
    red_start_frame: Optional[int] = None


@dataclass 
class StopLine:
    """
    Stop line information
    Có thể từ detection hoặc set manual
    """
    detection: Optional[Detection] = None
    y_position: Optional[int] = None  # Manual position if not detected
    
    @property
    def line_y(self) -> Optional[int]:
        """
        Get Y coordinate của stop line
        QUAN TRỌNG: Dùng CENTER Y, không phải top Y
        """
        if self.detection:
            # Use CENTER Y của bounding box
            y1 = self.detection.bbox[1]
            y2 = self.detection.bbox[3]
            return (y1 + y2) // 2
        elif self.y_position:
            return self.y_position
        return None
    
    @property
    def is_valid(self) -> bool:
        return self.line_y is not None


@dataclass
class VehicleState:
    """
    Track state của mỗi xe cho violation detection
    
    QUAN TRỌNG: 
    - was_before_line_when_red: Xe có ở TRƯỚC vạch khi đèn đỏ không?
    - Chỉ những xe ở TRƯỚC vạch mới có thể vi phạm
    - Xe đã qua vạch khi đèn đỏ = đang đi hợp lệ, KHÔNG PHẠT
    """
    track_id: int
    
    # ========== VỊ TRÍ KHI ĐÈN CHUYỂN ĐỎ ==========
    # Đây là điểm quan trọng nhất - cần lưu vị trí NGAY KHI đèn đỏ
    position_when_red_started: Optional[int] = None  # Y coordinate
    was_before_line_when_red: bool = False  # True = có thể vi phạm
    
    # ========== CROSSING DETECTION ==========
    has_crossed: bool = False
    crossing_frame: Optional[int] = None
    crossing_time: Optional[datetime] = None
    
    # ========== VIOLATION CONFIRMATION ==========
    violation_confirmed: bool = False
    violation_frames_count: int = 0  # Đếm số frame vi phạm liên tiếp
    
    # ========== EXEMPTIONS ==========
    yellow_exempt: bool = False  # Miễn vì quá gần khi đèn vàng
    
    # ========== TRAJECTORY ==========
    # Lưu history Y positions để detect crossing motion
    y_positions: Deque = field(default_factory=lambda: deque(maxlen=10))
    # Lưu history X positions để detect xe đi ngang
    x_positions: Deque = field(default_factory=lambda: deque(maxlen=10))
    
    def update_position(self, y: int, x: int = None):
        """Update position history"""
        self.y_positions.append(y)
        if x is not None:
            self.x_positions.append(x)


@dataclass
class Violation:
    """
    Violation record với đầy đủ thông tin bằng chứng
    """
    violation_id: str
    track_id: int
    timestamp: datetime
    frame_number: int
    
    # Vehicle info
    vehicle_class: str
    vehicle_bbox: Tuple[int, int, int, int]
    vehicle_confidence: float
    
    # Traffic light info
    light_state: str
    red_light_duration: float  # Đèn đỏ đã bao lâu khi vi phạm
    
    # Stop line info
    stop_line_y: int
    crossing_distance: float  # Đã vượt qua vạch bao xa (pixels)
    
    # Evidence
    evidence_frames: List[np.ndarray] = field(default_factory=list)
    evidence_paths: List[str] = field(default_factory=list)
    
    # Metadata
    location: str = ""
    camera_id: str = ""
    model_used: str = "YOLOv11"
    status: str = "Chưa xử lý"  # Chưa xử lý, Đã xử lý, Đã hủy
    license_plate: Optional[str] = None
    officer_note: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export"""
        return {
            'violation_id': self.violation_id,
            'track_id': self.track_id,
            'timestamp': self.timestamp.isoformat(),
            'frame_number': self.frame_number,
            'vehicle': {
                'class': self.vehicle_class,
                'bbox': list(self.vehicle_bbox),
                'confidence': self.vehicle_confidence
            },
            'traffic_light': {
                'state': self.light_state,
                'red_duration_seconds': self.red_light_duration
            },
            'stop_line': {
                'y_position': self.stop_line_y,
                'crossing_distance_pixels': self.crossing_distance
            },
            'evidence_paths': self.evidence_paths,
            'location': self.location,
            'camera_id': self.camera_id,
            'model_used': self.model_used,
            'status': self.status,
            'license_plate': self.license_plate,
            'officer_note': self.officer_note
        }


# ============================================================================
# MAIN VIOLATION DETECTOR CLASS
# ============================================================================

class ViolationDetector:
    """
    Main violation detection logic - IMPROVED VERSION
    
    ==========================================================================
    CORE LOGIC FLOW:
    ==========================================================================
    
    1. TRACK TRAFFIC LIGHT với voting mechanism
       - Dùng history 5 frames để voting
       - Tránh flicker từ detection không ổn định
       
    2. KHI ĐÈN CHUYỂN ĐỎ: Snapshot vị trí TẤT CẢ xe
       - Lưu Y position của mỗi xe
       - Đánh dấu xe nào ở TRƯỚC/SAU vạch
       - CHỈ xe ở TRƯỚC vạch mới có thể vi phạm
       
    3. THEO DÕI CROSSING MOTION:
       - Không chỉ check "xe ở sau vạch"
       - Phải có CHUYỂN ĐỘNG từ trước vạch → sau vạch
       - Tránh phạt xe đứng yên
       
    4. GRACE PERIOD:
       - Không phạt trong 1-2 giây đầu sau khi đèn đỏ
       - Tránh oan xe đang đi và không kịp dừng
       
    5. MULTI-FRAME CONFIRMATION:
       - Cần ít nhất 3 frames vi phạm liên tiếp
       - Tránh detection noise
       
    6. GHI NHẬN BẰNG CHỨNG:
       - 3 ảnh: trước, trong, sau vi phạm
       - Annotated với bounding boxes
       - Metadata đầy đủ
    ==========================================================================
    """
    
    def __init__(self, config: dict):
        self.config = config
        violation_config = config.get('violation', {})
        location_config = config.get('location', {})
        
        # ========== CONFIGURATION ==========
        self.grace_period = violation_config.get('grace_period', DEFAULT_GRACE_PERIOD)
        self.min_frames = violation_config.get('min_frames', DEFAULT_MIN_FRAMES)
        self.stop_line_threshold = violation_config.get('stop_line_threshold', DEFAULT_STOP_LINE_THRESHOLD)
        
        # ROI config
        roi_config = violation_config.get('roi', {})
        self.roi_enabled = roi_config.get('enabled', False)
        self.roi_x_min = roi_config.get('x_min', 0.0)
        self.roi_x_max = roi_config.get('x_max', 1.0)
        self.roi_y_min = roi_config.get('y_min', 0.0)
        self.roi_y_max = roi_config.get('y_max', 1.0)
        
        # Minimum vehicle confidence để tính vi phạm
        self.min_vehicle_confidence = violation_config.get('min_vehicle_confidence', 0.5)
        
        # Location info
        self.location = location_config.get('intersection', 'Unknown')
        self.camera_id = location_config.get('camera_id', 'CAM_001')
        
        # ========== STATE ==========
        # Traffic light state với voting
        self.traffic_light = TrafficLightState()
        
        # Stop line
        self.stop_line: Optional[StopLine] = None
        
        # Vehicle states: track_id -> VehicleState
        self.vehicle_states: Dict[int, VehicleState] = {}
        
        # Recorded violations: track_id -> Violation
        self.violations: Dict[int, Violation] = {}
        
        # Frame buffer cho evidence collection (~5 giây at 30fps)
        self.frame_buffer: Deque = deque(maxlen=150)
        
        # Store current detections for evidence
        self.current_detections: List[Detection] = []
        
        # ========== STATISTICS ==========
        self.total_frames_processed = 0
        self.total_vehicles_tracked = 0
        
        # Track vị trí đèn đỏ để xác định lane
        self.red_light_bbox = None  # (x1, y1, x2, y2) của đèn đỏ
        self.red_light_center_x = None  # Tâm x của đèn đỏ (0-1)
        
        logger.info(f"✅ ViolationDetector initialized")
        logger.info(f"   - Grace period: {self.grace_period}s")
        logger.info(f"   - Min frames: {self.min_frames}")
        logger.info(f"   - Stop line threshold: {self.stop_line_threshold}px")
        if self.roi_enabled:
            logger.info(f"   - ROI: x=[{self.roi_x_min:.0%}-{self.roi_x_max:.0%}], y=[{self.roi_y_min:.0%}-{self.roi_y_max:.0%}]")
    
    @property
    def current_light_state(self) -> str:
        """Trả về trạng thái đèn hiện tại"""
        return self.traffic_light.current_state
    
    def _is_in_roi(self, vehicle: 'TrackedObject', frame_shape: tuple) -> bool:
        """
        Check if vehicle is in the lane controlled by the detected red light
        
        Logic:
        - Nếu đèn đỏ ở bên PHẢI (x > 0.5): chỉ bắt xe ở lane GIỮA và PHẢI
        - Nếu đèn đỏ ở bên TRÁI (x < 0.5): chỉ bắt xe ở lane GIỮA và TRÁI  
        - Xe ở lane đối diện (ngược lại) = KHÔNG bắt
        """
        if not self.roi_enabled:
            return True
        
        x1, y1, x2, y2 = vehicle.detection.bbox
        h, w = frame_shape[:2]
        self._frame_width = w  # Lưu để tính red_light_center_x
        
        # Get center of vehicle (normalized 0-1)
        vehicle_cx = (x1 + x2) / 2 / w
        vehicle_cy = (y1 + y2) / 2 / h
        
        # Check Y trong range
        if not (self.roi_y_min <= vehicle_cy <= self.roi_y_max):
            return False
        
        # Nếu có vị trí đèn đỏ, dùng nó để xác định lane
        if self.red_light_center_x is not None:
            # Đèn đỏ ở bên phải (> 0.5) -> chỉ bắt xe ở phần giữa-phải của frame
            # Đèn đỏ ở bên trái (< 0.5) -> chỉ bắt xe ở phần giữa-trái của frame
            if self.red_light_center_x > 0.5:
                # Đèn ở phải -> xe phải ở vùng center đến phải (0.3 - 0.85)
                # Không bắt xe ở mép trái (lane ngược chiều)
                if vehicle_cx < 0.25:
                    return False  # Xe ở lane bên trái (ngược chiều)
            else:
                # Đèn ở trái -> xe phải ở vùng center đến trái (0.15 - 0.7)
                if vehicle_cx > 0.75:
                    return False  # Xe ở lane bên phải (ngược chiều)
        
        # Fallback: dùng ROI config
        return (self.roi_x_min <= vehicle_cx <= self.roi_x_max)
    
    # ========================================================================
    # PUBLIC API - Interface chính
    # ========================================================================
    
    def update(self,
               tracked_vehicles: List[TrackedObject],
               detections: List[Detection],
               frame: np.ndarray,
               frame_number: int,
               timestamp: datetime) -> List[Violation]:
        """
        Main update function - GỌI MỖI FRAME
        
        Args:
            tracked_vehicles: List tracked vehicles từ ByteTrack
            detections: Tất cả detections từ model
            frame: Frame image hiện tại
            frame_number: Số frame
            timestamp: Thời gian hiện tại
            
        Returns:
            List violations MỚI phát hiện trong frame này
        """
        self.total_frames_processed += 1
        new_violations = []
        
        # Lưu detections hiện tại để vẽ lên evidence
        self.current_detections = detections
        
        # Store frame vào buffer cho evidence (kèm detections)
        self.frame_buffer.append({
            'frame': frame.copy(),
            'frame_number': frame_number,
            'timestamp': timestamp,
            'detections': detections  # Lưu detections để annotate evidence
        })
        
        # 1. Update traffic light state (với voting)
        self._update_traffic_light_state(detections, timestamp, frame_number)
        
        # 2. Update stop line position
        self._update_stop_line(detections)
        
        # 2b. Fallback: nếu chưa có stop_line, dùng default
        # Logic mới: KHÔNG CẦN stop_line cũng có thể detect vi phạm
        # Nếu đèn đỏ + xe di chuyển ra xa (y tăng) = vi phạm
        if self.stop_line is None or not self.stop_line.is_valid:
            if frame is not None:
                # Stop line ở khoảng 25% từ trên xuống (vùng trên của camera)
                default_y = int(frame.shape[0] * 0.25)
                self.stop_line = StopLine(y_position=default_y)
                logger.info(f"📍 Using default stop line at y={default_y}")
        
        # 3. Skip CHỈ khi không có đèn giao thông
        if self.traffic_light.current_state == "UNKNOWN":
            logger.debug(f"Cannot detect: no traffic light detected")
            return new_violations
        
        # Nếu không có stop_line, vẫn tiếp tục với default
        
        stop_line_y = self.stop_line.line_y
        
        # 4. Handle light state changes (QUAN TRỌNG)
        self._handle_light_state_change(tracked_vehicles, stop_line_y, timestamp, frame_number)
        
        # 5. Check từng xe cho violations
        for vehicle in tracked_vehicles:
            # Filter: chỉ check vehicle classes
            if vehicle.detection.class_name not in VEHICLE_CLASSES:
                continue
            
            # Filter: chỉ check xe có confidence đủ cao
            if vehicle.detection.confidence < self.min_vehicle_confidence:
                continue
            
            # Filter: chỉ check xe trong ROI (vùng giám sát của đèn đỏ)
            in_roi = self._is_in_roi(vehicle, frame.shape)
            if not in_roi:
                if frame_number % 30 == 0:
                    x1, y1, x2, y2 = vehicle.detection.bbox
                    h, w = frame.shape[:2]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    logger.debug(f"Track {vehicle.track_id} OUTSIDE ROI: cx={cx:.2f}, cy={cy:.2f}")
                continue
            
            self.total_vehicles_tracked += 1
            
            # Get hoặc create vehicle state
            state = self._get_or_create_vehicle_state(vehicle, stop_line_y)
            
            # Update vehicle position (cả X và Y)
            x1, y1, x2, y2 = vehicle.detection.bbox
            vehicle_y = y2  # Bottom of bbox
            vehicle_x = (x1 + x2) // 2  # Center X
            state.update_position(vehicle_y, vehicle_x)
            
            # Check violation
            violation = self._check_vehicle_violation(
                vehicle=vehicle,
                state=state,
                stop_line_y=stop_line_y,
                frame=frame,
                frame_number=frame_number,
                timestamp=timestamp
            )
            
            if violation:
                new_violations.append(violation)
                logger.warning(f"🚨 VIOLATION DETECTED: Track {vehicle.track_id}")
        
        return new_violations
    
    def set_stop_line_manual(self, y_position: int):
        """Manually set stop line position (cho setup ban đầu)"""
        self.stop_line = StopLine(y_position=y_position)
        logger.info(f"📍 Stop line manually set at y={y_position}")
    
    def get_current_state(self) -> dict:
        """Get current detector state cho debugging/display"""
        return {
            'traffic_light': self.traffic_light.current_state,
            'stop_line_y': self.stop_line.line_y if self.stop_line else None,
            'active_vehicles': len(self.vehicle_states),
            'total_violations': len(self.violations),
            'frames_processed': self.total_frames_processed
        }
    
    def get_statistics(self) -> dict:
        """Get violation statistics"""
        by_class = {}
        for v in self.violations.values():
            cls = v.vehicle_class
            by_class[cls] = by_class.get(cls, 0) + 1
        
        return {
            'total_violations': len(self.violations),
            'by_vehicle_class': by_class,
            'current_light_state': self.traffic_light.current_state,
            'frames_processed': self.total_frames_processed,
            'vehicles_tracked': self.total_vehicles_tracked
        }
    
    # ========================================================================
    # TRAFFIC LIGHT HANDLING - Xử lý trạng thái đèn
    # ========================================================================
    
    def _update_traffic_light_state(self, detections: List[Detection], 
                                     timestamp: datetime, frame_number: int):
        """
        Update traffic light state với VOTING MECHANISM
        
        Voting giúp:
        - Tránh flicker khi detection không ổn định
        - Smoothing state transitions
        - Ưu tiên safety (đèn đỏ) khi không rõ ràng
        """
        # Find all traffic light detections
        light_detections = [d for d in detections if d.class_name in LIGHT_CLASSES]
        
        if not light_detections:
            # Không có detection - giữ state trước
            return
        
        # Lấy detection có confidence cao nhất
        best_light = max(light_detections, key=lambda x: x.confidence)
        detected_state = best_light.class_name.replace('_light', '').upper()
        
        # LƯU VỊ TRÍ ĐÈN ĐỎ để xác định lane
        if detected_state == "RED":
            self.red_light_bbox = best_light.bbox
            x1, y1, x2, y2 = best_light.bbox
            # Tính center x (normalized 0-1)
            if hasattr(self, '_frame_width') and self._frame_width > 0:
                self.red_light_center_x = (x1 + x2) / 2 / self._frame_width
        
        # Add vào history
        self.traffic_light.state_history.append(detected_state)
        self.traffic_light.confidence = best_light.confidence
        
        # VOTING: xác định state từ history gần đây - TĂNG LÊN 5 frames để tránh flicker
        if len(self.traffic_light.state_history) >= 5:
            recent = list(self.traffic_light.state_history)[-5:]
            vote_counts = Counter(recent)
            voted_state = vote_counts.most_common(1)[0][0]
            
            # Update state nếu có ít nhất 3/5 đồng ý
            if vote_counts[voted_state] >= 3:
                old_state = self.traffic_light.current_state
                
                if voted_state != old_state:
                    # QUAN TRỌNG: Nếu đang RED và chỉ flicker sang YELLOW rồi về RED
                    # thì KHÔNG reset red_start_time
                    if old_state == "RED" and voted_state == "YELLOW":
                        # Check xem có phải flicker không (đèn đỏ < 2 giây)
                        if self.traffic_light.red_start_time:
                            time_red = (timestamp - self.traffic_light.red_start_time).total_seconds()
                            if time_red < 2.0:
                                # Có thể là flicker - giữ RED
                                logger.debug(f"🚦 Ignoring flicker RED→YELLOW (only {time_red:.1f}s)")
                                return
                    
                    self.traffic_light.current_state = voted_state
                    self.traffic_light.last_change_time = timestamp
                    
                    logger.info(f"🚦 Traffic light: {old_state} → {voted_state}")
                    
                    # Track thời điểm đèn đỏ bắt đầu - KHÔNG reset nếu từ YELLOW quay về RED nhanh
                    if voted_state == "RED":
                        # Nếu trước đó là YELLOW và đèn đỏ chưa reset, giữ nguyên
                        if old_state == "YELLOW" and self.traffic_light.red_start_time:
                            time_since_red = (timestamp - self.traffic_light.red_start_time).total_seconds()
                            if time_since_red < 5.0:  # Trong 5 giây
                                logger.debug(f"🔴 Keeping existing red_start_time (flicker recovery)")
                                return
                        
                        self.traffic_light.red_start_time = timestamp
                        self.traffic_light.red_start_frame = frame_number
                        logger.info(f"🔴 Red light started at frame {frame_number}")
    
    def _handle_light_state_change(self, tracked_vehicles: List[TrackedObject],
                                    stop_line_y: int, timestamp: datetime, 
                                    frame_number: int):
        """
        Handle khi traffic light thay đổi state
        
        ==========================================================================
        CRITICAL: Khi đèn chuyển ĐỎ, phải snapshot vị trí TẤT CẢ xe
        ==========================================================================
        
        Lý do:
        - Cần biết xe nào ở TRƯỚC/SAU vạch tại thời điểm đèn đỏ
        - Xe đã qua vạch = đang đi hợp lệ, KHÔNG PHẠT
        - Xe ở trước vạch mà sau đó vượt qua = VI PHẠM
        """
        current_state = self.traffic_light.current_state
        
        # ========== KHI ĐÈN CHUYỂN ĐỎ ==========
        if current_state == "RED" and self.traffic_light.red_start_frame == frame_number:
            logger.debug(f"📸 Recording vehicle positions at red light start")
            
            for vehicle in tracked_vehicles:
                if vehicle.detection.class_name not in VEHICLE_CLASSES:
                    continue
                
                state = self._get_or_create_vehicle_state(vehicle, stop_line_y)
                vehicle_y = self._get_vehicle_bottom_y(vehicle)
                
                # LƯU VỊ TRÍ khi đèn đỏ bắt đầu
                state.position_when_red_started = vehicle_y
                
                # QUAN TRỌNG: Đánh dấu xe ở TRƯỚC hay SAU vạch
                state.was_before_line_when_red = (vehicle_y <= stop_line_y)
                
                logger.debug(f"  Track {vehicle.track_id}: y={vehicle_y}, "
                           f"before_line={state.was_before_line_when_red}")
        
        # ========== KHI ĐÈN CHUYỂN XANH ==========
        elif current_state == "GREEN":
            if self.vehicle_states:
                logger.debug("🟢 Green light - resetting vehicle states")
                self.vehicle_states.clear()
    
    # ========================================================================
    # STOP LINE HANDLING - Xử lý vạch dừng
    # ========================================================================
    
    def _update_stop_line(self, detections: List[Detection]):
        """Update stop line từ detections"""
        stop_line_det = next(
            (d for d in detections if d.class_name == 'stop_line'), 
            None
        )
        
        if stop_line_det:
            self.stop_line = StopLine(detection=stop_line_det)
    
    # ========================================================================
    # VEHICLE STATE MANAGEMENT - Quản lý state xe
    # ========================================================================
    
    def _get_or_create_vehicle_state(self, vehicle: TrackedObject, 
                                      stop_line_y: int) -> VehicleState:
        """
        Get existing vehicle state hoặc create mới
        
        Với xe mới xuất hiện TRONG KHI đèn đỏ:
        - Cần xác định vị trí hiện tại
        - Nếu đã ở SAU vạch → không phạt (có thể đi từ trước)
        - Nếu ở TRƯỚC vạch → có thể vi phạm nếu vượt qua
        """
        track_id = vehicle.track_id
        
        if track_id not in self.vehicle_states:
            vehicle_y = self._get_vehicle_bottom_y(vehicle)
            
            # Create new state
            state = VehicleState(track_id=track_id)
            
            # Nếu đèn đang đỏ, record vị trí ban đầu
            if self.traffic_light.current_state == "RED":
                state.position_when_red_started = vehicle_y
                state.was_before_line_when_red = (vehicle_y <= stop_line_y)
            
            self.vehicle_states[track_id] = state
            logger.debug(f"New vehicle state: Track {track_id}, y={vehicle_y}")
        
        return self.vehicle_states[track_id]
    
    def _get_vehicle_bottom_y(self, vehicle: TrackedObject) -> int:
        """
        Get bottom Y coordinate của vehicle
        = Phần dưới cùng của xe (mũi xe trong ảnh)
        """
        return vehicle.detection.bbox[3]  # y2 = bottom
    
    # ========================================================================
    # VIOLATION DETECTION - CORE LOGIC
    # ========================================================================
    
    def _check_vehicle_violation(self, vehicle: TrackedObject, state: VehicleState,
                                  stop_line_y: int, frame: np.ndarray,
                                  frame_number: int, timestamp: datetime) -> Optional[Violation]:
        """
        ==========================================================================
        CORE VIOLATION DETECTION LOGIC - ĐƠN GIẢN HÓA
        ==========================================================================
        
        Logic mới đơn giản:
        1. Đèn đang ĐỎ
        2. Qua grace period
        3. Xe đang DI CHUYỂN (y position thay đổi đáng kể)
        4. Chưa ghi nhận trước đó
        5. Đủ số frame xác nhận
        
        KHÔNG CẦN stop_line - chỉ cần xe di chuyển khi đèn đỏ là vi phạm
        """
        track_id = vehicle.track_id
        
        # ========== ĐÃ VI PHẠM - SKIP ==========
        if track_id in self.violations:
            return None
        
        if state.violation_confirmed:
            return None
        
        # ========== ĐIỀU KIỆN 1: ĐÈN PHẢI ĐỎ ==========
        if self.traffic_light.current_state != "RED":
            # Reset violation count khi không phải đèn đỏ
            state.violation_frames_count = 0
            return None
        
        # ========== ĐIỀU KIỆN 2: KHÔNG TRONG GRACE PERIOD ==========
        red_start = self.traffic_light.red_start_time
        if red_start is None:
            return None
        
        time_since_red = (timestamp - red_start).total_seconds()
        vehicle_y = self._get_vehicle_bottom_y(vehicle)
        
        if time_since_red < self.grace_period:
            return None
        
        # ========== ĐIỀU KIỆN 3: XE KHÔNG ĐI NGANG (từ lane khác) ==========
        # Bỏ check này vì crossing_distance đã filter xe đi ngang rồi
        # Xe đi ngang sẽ có crossing_distance âm hoặc nhỏ
        
        # ========== ĐIỀU KIỆN 4: XE PHẢI QUA VẠCH (crossing_distance > 0) ==========
        # Crossing distance = vehicle_y - stop_line_y
        # Dương = xe đã qua vạch (về phía camera)
        # Âm = xe ở trước vạch hoặc đi ngược chiều -> KHÔNG PHẠT
        
        crossing_distance = vehicle_y - stop_line_y
        is_past_stop_line = crossing_distance > self.stop_line_threshold
        
        # Log MỌI XE đang được check khi đèn đỏ
        if self.traffic_light.current_state == "RED":
            logger.debug(f"🔍 Track {track_id}: y={vehicle_y}, stop_line={stop_line_y}, cross_dist={crossing_distance}, past={is_past_stop_line}, red_dur={time_since_red:.1f}s")
        
        # QUAN TRỌNG: Nếu crossing_distance âm nhiều -> xe đi ngược chiều, SKIP
        if crossing_distance < -50:  # Xe đi ngược chiều hoặc lane khác
            return None
        
        # CHỈ VI PHẠM khi xe QUA VẠCH (crossing_distance > threshold)
        if not is_past_stop_line:
            return None
        
        # ========== ĐIỀU KIỆN 5: MULTI-FRAME CONFIRMATION ==========
        state.violation_frames_count += 1
        
        logger.debug(f"Track {track_id}: violation frame {state.violation_frames_count}/{self.min_frames}")
        
        if state.violation_frames_count < self.min_frames:
            return None
        
        # ==========================================================
        # ✅ VI PHẠM ĐƯỢC XÁC NHẬN
        # ==========================================================
        state.violation_confirmed = True
        state.crossing_frame = frame_number
        state.crossing_time = timestamp
        
        violation = self._create_violation(
            vehicle=vehicle,
            state=state,
            stop_line_y=stop_line_y,
            frame=frame,
            frame_number=frame_number,
            timestamp=timestamp,
            time_since_red=time_since_red
        )
        
        self.violations[track_id] = violation
        return violation
    
    def _is_vehicle_moving(self, state: VehicleState) -> bool:
        """Check if vehicle is moving (not stationary)"""
        positions = list(state.y_positions)
        if len(positions) < 2:
            return True  # Assume moving if not enough data
        
        # Check if Y changed significantly
        recent = positions[-3:] if len(positions) >= 3 else positions
        y_diff = max(recent) - min(recent)
        return y_diff > 5  # Threshold: 5 pixels movement
    
    def _is_vehicle_moving_sideways(self, state: VehicleState) -> bool:
        """
        Check if vehicle is moving SIDEWAYS (left-right) - xe đi ngang từ lane khác
        
        Xe đi ngang có đặc điểm:
        - X thay đổi nhiều (> 30px)
        - Y thay đổi ít hoặc giảm (đi ra xa camera)
        
        Returns True nếu xe đang đi ngang -> KHÔNG PHẠT
        """
        x_positions = list(state.x_positions)
        y_positions = list(state.y_positions)
        
        if len(x_positions) < 3 or len(y_positions) < 3:
            return False  # Không đủ data
        
        recent_x = x_positions[-5:] if len(x_positions) >= 5 else x_positions
        recent_y = y_positions[-5:] if len(y_positions) >= 5 else y_positions
        
        x_diff = abs(max(recent_x) - min(recent_x))
        y_diff = max(recent_y) - min(recent_y)  # Y tăng = đi về phía camera
        
        # Xe đi ngang: X thay đổi nhiều (>50px), Y thay đổi ít (<30px)
        is_sideways = x_diff > 50 and y_diff < 30
        
        if is_sideways:
            logger.debug(f"Xe đi ngang detected: x_diff={x_diff}, y_diff={y_diff}")
        
        return is_sideways
    
    def _is_vehicle_moving_any_direction(self, state: VehicleState) -> bool:
        """
        Check if vehicle is moving in ANY direction
        
        Đơn giản: Xe di chuyển (Y thay đổi đáng kể) khi đèn đỏ = VI PHẠM
        Không quan tâm hướng đi - đi thẳng, quẹo trái, quẹo phải đều phạt
        """
        positions = list(state.y_positions)
        if len(positions) < 2:
            return True  # Assume moving if not enough data
        
        # Lấy các vị trí gần đây
        recent = positions[-4:] if len(positions) >= 4 else positions
        
        # Check có di chuyển không - threshold thấp (3px)
        y_diff = max(recent) - min(recent)
        return y_diff > 3  # Giảm threshold để nhạy hơn
    
    def _is_vehicle_moving_forward(self, state: VehicleState) -> bool:
        """
        Check if vehicle is moving FORWARD (Y increasing = moving towards camera)
        
        Quan trọng: Chỉ phạt xe đang tiến tới, không phạt xe:
        - Đứng yên
        - Đang lùi
        - Di chuyển ngang
        """
        positions = list(state.y_positions)
        if len(positions) < 3:
            return False  # Cần đủ data để xác nhận hướng
        
        recent = positions[-5:] if len(positions) >= 5 else positions
        
        # Check 1: Có di chuyển không (không đứng yên)
        y_diff = max(recent) - min(recent)
        if y_diff < 10:  # Threshold: 10 pixels
            return False  # Đứng yên
        
        # Check 2: Y đang tăng (di chuyển tới camera)
        # So sánh vị trí đầu vs cuối
        is_forward = recent[-1] > recent[0] + 5  # Có tiến tới ít nhất 5px
        
        return is_forward
    
    def _detect_crossing_motion(self, state: VehicleState, stop_line_y: int) -> bool:
        """
        Detect xe đang VƯỢT QUA stop line
        
        ==========================================================================
        QUAN TRỌNG: Cần có CHUYỂN ĐỘNG, không chỉ check vị trí tĩnh
        ==========================================================================
        
        Criteria:
        1. Có trajectory history (ít nhất 5 vị trí)
        2. Có vị trí TRƯỚC vạch trong history
        3. Vị trí hiện tại SAU vạch
        4. Đang di chuyển tới (y tăng)
        
        Tại sao cần kiểm tra chuyển động?
        - Tránh phạt xe đứng yên SAU vạch
        - Tránh false positive từ detection noise
        - Xác nhận hành vi VƯỢT QUA, không phải chỉ "ở sau vạch"
        """
        positions = list(state.y_positions)
        
        # Cần đủ history (giảm từ 5 xuống 3 để detect nhanh hơn)
        if len(positions) < 3:
            return False
        
        recent = positions[-3:]
        
        # Check 1: Có vị trí TRƯỚC vạch trong history gần
        had_before = any(y <= stop_line_y for y in recent[:-1])
        
        # Check 2: Vị trí hiện tại SAU vạch (với threshold)
        current_after = recent[-1] > (stop_line_y + self.stop_line_threshold)
        
        # Check 3: Đang di chuyển tới (y tăng overall)
        is_moving_forward = recent[-1] > recent[0]
        
        return had_before and current_after and is_moving_forward
    
    # ========================================================================
    # VIOLATION CREATION & EVIDENCE - Tạo vi phạm và bằng chứng
    # ========================================================================
    
    def _create_violation(self, vehicle: TrackedObject, state: VehicleState,
                          stop_line_y: int, frame: np.ndarray,
                          frame_number: int, timestamp: datetime,
                          time_since_red: float) -> Violation:
        """Create violation record với evidence"""
        
        vehicle_y = self._get_vehicle_bottom_y(vehicle)
        crossing_distance = vehicle_y - stop_line_y
        
        violation_id = f"VL_{timestamp.strftime('%Y%m%d_%H%M%S')}_{vehicle.track_id:04d}"
        
        violation = Violation(
            violation_id=violation_id,
            track_id=vehicle.track_id,
            timestamp=timestamp,
            frame_number=frame_number,
            vehicle_class=vehicle.detection.class_name,
            vehicle_bbox=vehicle.detection.bbox,
            vehicle_confidence=vehicle.detection.confidence,
            light_state="RED",
            red_light_duration=time_since_red,
            stop_line_y=stop_line_y,
            crossing_distance=crossing_distance,
            location=self.location,
            camera_id=self.camera_id
        )
        
        # Collect evidence frames
        self._collect_evidence_frames(violation, frame_number)
        
        logger.info(f"📋 Created violation: {violation_id}")
        logger.info(f"   - Vehicle: {vehicle.detection.class_name} (Track {vehicle.track_id})")
        logger.info(f"   - Red light duration: {time_since_red:.1f}s")
        logger.info(f"   - Crossing distance: {crossing_distance:.0f}px")
        
        return violation
    
    def _collect_evidence_frames(self, violation: Violation, current_frame: int):
        """
        Collect 3 evidence frames: before, during, after
        
        Theo chuẩn quốc tế về bằng chứng vi phạm giao thông
        Lưu kèm detections để annotate sau
        """
        fps = 30  # Assume 30 fps
        
        # Target frames: 1 giây trước, hiện tại, 1 giây sau
        target_frames = [
            current_frame - fps,      # Pre-violation (~1s trước)
            current_frame,            # During violation
            # current_frame + fps     # Post-violation (chưa có)
        ]
        
        for target in target_frames:
            # Tìm frame gần nhất trong buffer
            for frame_data in self.frame_buffer:
                if frame_data['frame_number'] == target:
                    # Lưu cả frame và detections
                    evidence_data = {
                        'frame': frame_data['frame'].copy(),
                        'detections': frame_data.get('detections', [])
                    }
                    violation.evidence_frames.append(evidence_data)
                    break
            else:
                # Frame không tìm thấy - dùng current nếu là target
                if target == current_frame:
                    for frame_data in self.frame_buffer:
                        if frame_data['frame_number'] == current_frame:
                            evidence_data = {
                                'frame': frame_data['frame'].copy(),
                                'detections': frame_data.get('detections', [])
                            }
                            violation.evidence_frames.append(evidence_data)
                            break
    
    def save_violation_evidence(self, violation: Violation, 
                                output_dir: Path) -> List[str]:
        """Save violation evidence images to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        labels = ['pre', 'during', 'post']
        
        for i, evidence_data in enumerate(violation.evidence_frames):
            label = labels[i] if i < len(labels) else f'frame_{i}'
            filename = f"{violation.violation_id}_{label}.jpg"
            filepath = output_dir / filename
            
            # Extract frame and detections
            if isinstance(evidence_data, dict):
                frame = evidence_data['frame']
                detections = evidence_data.get('detections', [])
            else:
                # Backward compatibility - nếu là frame cũ
                frame = evidence_data
                detections = []
            
            # Annotate frame với ALL detections
            annotated = self._annotate_evidence_frame(
                frame=frame,
                violation=violation,
                label=label.upper(),
                detections=detections
            )
            
            cv2.imwrite(str(filepath), annotated)
            saved_paths.append(str(filepath))
        
        violation.evidence_paths = saved_paths
        logger.info(f"💾 Saved {len(saved_paths)} evidence images for {violation.violation_id}")
        
        return saved_paths
    
    def _annotate_evidence_frame(self, frame: np.ndarray, violation: Violation,
                                  label: str, detections: List = None) -> np.ndarray:
        """Annotate evidence frame với ALL bounding boxes và info"""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Color mapping for different classes
        colors = {
            'red_light': (0, 0, 255),      # Red
            'green_light': (0, 255, 0),    # Green
            'yellow_light': (0, 255, 255), # Yellow
            'stop_line': (255, 255, 0),    # Cyan
            'car': (255, 128, 0),          # Orange
            'motobike': (255, 0, 128),     # Pink
            'truck': (128, 0, 255),        # Purple
        }
        
        # Draw ALL detections
        if detections:
            for det in detections:
                try:
                    if hasattr(det, 'bbox'):
                        dx1, dy1, dx2, dy2 = det.bbox
                        class_name = det.class_name
                        conf = det.confidence
                    elif isinstance(det, dict):
                        dx1, dy1, dx2, dy2 = det['bbox']
                        class_name = det.get('class_name', 'unknown')
                        conf = det.get('confidence', 0)
                    else:
                        continue
                    
                    color = colors.get(class_name, (200, 200, 200))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (int(dx1), int(dy1)), (int(dx2), int(dy2)), color, 2)
                    
                    # Label
                    det_label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated, det_label, (int(dx1), int(dy1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except Exception as e:
                    logger.debug(f"Skip detection annotation: {e}")
        
        # Highlight the violating vehicle (thick red box)
        x1, y1, x2, y2 = violation.vehicle_bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
        vehicle_label = f"VI PHAM: {violation.vehicle_class.upper()} - Track {violation.track_id}"
        cv2.putText(annotated, vehicle_label, (x1, y1 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw stop line (VÀNG)
        line_y = violation.stop_line_y
        cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 255), 3)
        cv2.putText(annotated, "STOP LINE", (10, line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Timestamp (top-left)
        ts_text = violation.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated, ts_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Violation label (top-right)
        violation_text = f"VI PHAM - {label}"
        text_size = cv2.getTextSize(violation_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(annotated, violation_text, (w - text_size[0] - 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Red light indicator (top-center)
        cv2.circle(annotated, (w // 2, 30), 15, (0, 0, 255), -1)
        cv2.putText(annotated, "RED", (w // 2 + 20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Info box (bottom)
        info_y = h - 60
        cv2.rectangle(annotated, (0, info_y), (w, h), (0, 0, 0), -1)
        info_text = f"ID: {violation.violation_id} | Red: {violation.red_light_duration:.1f}s | Location: {violation.location}"
        cv2.putText(annotated, info_text, (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _can_detect_violations(self) -> bool:
        """Check có đủ components cho detection không"""
        if self.traffic_light.current_state == "UNKNOWN":
            return False
        if self.stop_line is None or not self.stop_line.is_valid:
            return False
        return True
    
    def reset(self):
        """Reset detector state"""
        self.vehicle_states.clear()
        self.violations.clear()
        self.traffic_light = TrafficLightState()
        self.frame_buffer.clear()
        logger.info("🔄 ViolationDetector reset")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_violation_detector(config: dict) -> ViolationDetector:
    """Factory function to create violation detector"""
    return ViolationDetector(config)
