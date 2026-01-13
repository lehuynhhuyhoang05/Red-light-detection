# CODE REVIEW & RECOMMENDATIONS

## Tổng quan

Hệ thống hiện tại đã có cấu trúc tốt với 3 module chính:
- `detector.py`: Object detection using Roboflow API
- `tracker.py`: Object tracking using ByteTrack
- `violation_logic.py`: Violation detection logic

## ✅ Điểm Tốt

1. **Architecture rõ ràng**: Tách biệt detection, tracking, và violation logic
2. **Dataclass sử dụng hiệu quả**: `Detection`, `TrackedObject`, `Violation`
3. **Logging đầy đủ**: Sử dụng `loguru` để tracking
4. **Factory pattern**: `create_detector()` cho flexibility
5. **Multiple model support**: YOLOv11, YOLO-NAS, RT-DETR

## 🔧 Cần Cải thiện

### 1. **detector.py**

#### Issue: Class ID không khớp
```python
# Hiện tại
class_id = int(box.cls[0])  # YOLO classes
class_name = self.class_names.get(class_id, f"class_{class_id}")
```

**Vấn đề**: `class_id` từ YOLO có thể không khớp với custom classes của bạn
- YOLO COCO: car=2, motorcycle=3, traffic light=9
- Model custom của bạn: car=0, motobike=1, red_light=3, etc.

**Fix**:
```python
class RoboflowDetector:
    def __init__(self, api_key: str, workspace: str, project: str, version: int = 1):
        # ... existing code ...
        
        # Define custom class mapping
        self.class_names = {
            0: 'car',
            1: 'green_light',
            2: 'motobike',
            3: 'red_light',
            4: 'stop_line',
            5: 'yellow_light'
        }
```

#### Issue: Missing class_id in Detection dataclass
```python
@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    # ❌ Missing class_id field
```

**Fix**:
```python
@dataclass
class Detection:
    class_name: str
    class_id: int  # ✅ Add this
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
```

### 2. **violation_logic.py**

#### Issue: Vehicle class filter quá strict
```python
# Line 150
if vehicle.detection.class_name not in ['vehicle', 'motorcycle', 'car', 'truck']:
    continue
```

**Vấn đề**: Model của bạn có class `car` và `motobike` (không phải `motorcycle`)

**Fix**:
```python
# Use class names from your model
VEHICLE_CLASSES = {'car', 'motobike'}  # Match your model classes

if vehicle.detection.class_name not in VEHICLE_CLASSES:
    continue
```

#### Issue: Traffic light detection không robust
```python
# Line 207
light_det = None
for det in detections:
    if det.class_name in ['red_light', 'yellow_light', 'green_light']:
        light_det = det
        break  # ❌ Chỉ lấy first detection
```

**Vấn đề**: Nếu có nhiều đèn trong frame (multi-lane), chỉ detect 1 đèn

**Fix**:
```python
def _update_traffic_light(self, detections: List[Detection], timestamp: datetime):
    """Update traffic light state with voting mechanism"""
    
    # Collect all traffic light detections
    light_detections = [d for d in detections 
                       if d.class_name in ['red_light', 'yellow_light', 'green_light']]
    
    if not light_detections:
        return
    
    # Use highest confidence detection
    best_light = max(light_detections, key=lambda x: x.confidence)
    new_state = best_light.class_name.replace('_light', '').upper()
    
    # State change detection
    if new_state != self.current_light_state:
        logger.info(f"Traffic light: {self.current_light_state} → {new_state}")
        self.current_light_state = new_state
        self.light_change_time = timestamp
        
        if new_state == 'GREEN':
            self._reset_vehicle_states()
```

### 3. **tracker.py**

#### Issue: Missing class_id conversion
```python
# Line 89
class_id = np.array([det.class_id for det in detections])
```

**Vấn đề**: `Detection` dataclass không có `class_id` field (xem fix ở detector.py)

### 4. **Chưa có License Plate Recognition**

Như đã đề xuất trong Chương 4, cần thêm module LPR:

```python
# src/license_plate.py
from typing import Optional, Dict
import numpy as np
import easyocr
from loguru import logger

class LicensePlateRecognizer:
    """License Plate Recognition for Vietnamese plates"""
    
    def __init__(self, config: dict):
        self.enabled = config.get('enabled', False)
        if not self.enabled:
            return
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=True)
        logger.info("✅ License Plate Recognizer initialized")
    
    def detect_and_read(self, vehicle_image: np.ndarray) -> Optional[Dict]:
        """
        Detect and read license plate from vehicle crop
        
        Returns:
            {
                'plate_text': '30A-12345',
                'confidence': 0.92,
                'bbox': [x1, y1, x2, y2]
            }
        """
        if not self.enabled:
            return None
        
        try:
            # Read text from image
            results = self.reader.readtext(vehicle_image)
            
            if not results:
                return None
            
            # Find best candidate (highest confidence)
            best_result = max(results, key=lambda x: x[2])
            bbox, text, confidence = best_result
            
            # Clean and validate
            plate_text = self._clean_text(text)
            
            if not self._validate_vn_plate(plate_text):
                return None
            
            return {
                'plate_text': plate_text,
                'confidence': confidence,
                'bbox': self._convert_bbox(bbox)
            }
            
        except Exception as e:
            logger.error(f"LPR error: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text"""
        # Remove spaces
        text = text.replace(' ', '')
        # Uppercase
        text = text.upper()
        return text
    
    def _validate_vn_plate(self, text: str) -> bool:
        """Validate Vietnamese plate format"""
        import re
        # 30A-12345 or 30A12345
        pattern = r'^\d{2}[A-Z]{1,2}-?\d{4,5}$'
        return bool(re.match(pattern, text))
    
    def _convert_bbox(self, ocr_bbox):
        """Convert EasyOCR bbox to [x1,y1,x2,y2]"""
        # OCR bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in ocr_bbox]
        ys = [p[1] for p in ocr_bbox]
        return [min(xs), min(ys), max(xs), max(ys)]
```

**Integration vào violation_logic.py**:
```python
from .license_plate import LicensePlateRecognizer

class ViolationDetector:
    def __init__(self, config: dict):
        # ... existing code ...
        
        # Add LPR
        lpr_config = config.get('license_plate', {})
        self.lpr = LicensePlateRecognizer(lpr_config)
    
    def _create_violation(self, vehicle, frame, frame_number, timestamp, violation_id):
        """Enhanced with LPR"""
        
        # Crop vehicle region
        x1, y1, x2, y2 = vehicle.detection.bbox
        vehicle_crop = frame[y1:y2, x1:x2]
        
        # Try to read license plate
        plate_info = self.lpr.detect_and_read(vehicle_crop)
        
        # Create violation
        violation = Violation(
            violation_id=violation_id,
            track_id=vehicle.track_id,
            timestamp=timestamp,
            frame_number=frame_number,
            vehicle_class=vehicle.detection.class_name,
            vehicle_bbox=vehicle.detection.bbox,
            light_state=self.current_light_state,
            stop_line_y=self.stop_line.line_y,
            confidence=vehicle.detection.confidence,
            location=self.location_config.get('intersection', 'Unknown'),
            evidence_frames=[frame.copy()],
            license_plate=plate_info['plate_text'] if plate_info else None,
            license_plate_confidence=plate_info['confidence'] if plate_info else None
        )
        
        if plate_info:
            logger.info(f"✅ Plate detected: {plate_info['plate_text']}")
        
        return violation
```

## 📋 Action Items

### Ưu tiên cao (Cần fix ngay)
1. ✅ Fix class_id mapping trong detector.py
2. ✅ Add class_id field vào Detection dataclass
3. ✅ Fix vehicle class filter trong violation_logic.py (car, motobike)
4. ✅ Improve traffic light detection (multiple lights)

### Ưu tiên trung bình (Nên làm)
5. 🔄 Add License Plate Recognition module
6. 🔄 Add unit tests cho các module chính
7. 🔄 Add configuration validation
8. 🔄 Add performance monitoring (FPS, latency)

### Ưu tiên thấp (Nice to have)
9. 📋 Add GUI for manual review
10. 📋 Add database integration
11. 📋 Add REST API
12. 📋 Add multi-camera support

## 🚀 Next Steps

1. **Fix code issues ngay** (Action items 1-4)
2. **Test với video thực tế** sử dụng model trained (mAP 87.9%)
3. **Thu thập performance metrics**:
   - FPS trên RTX 4090
   - Detection accuracy trong điều kiện thực tế
   - False positive/negative rate
4. **Implement License Plate Recognition** sau khi core system stable
5. **Document API** và user guide

## 📝 Config File Mẫu

```yaml
# config.yaml
model:
  type: "roboflow"  # hoặc "yolov11", "rt-detr"
  
  roboflow:
    api_key: "YOUR_API_KEY"
    workspace: "your-workspace"
    project: "red-light-violation-detect-hecrg"
    version: 3
    confidence: 40
    overlap: 30

tracking:
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8

violation:
  min_frames: 3
  grace_period: 1.0  # seconds
  stop_line_threshold: 20  # pixels

license_plate:
  enabled: true
  confidence_threshold: 0.7

location:
  intersection: "Nguyễn Văn Linh - Nguyễn Hữu Thọ"
  district: "Quận 7"
  city: "TP.HCM"

output:
  save_evidence: true
  evidence_dir: "./evidence"
  generate_report: true
```

## 🎯 Kết luận

Code base hiện tại đã có foundation tốt. Những fix nhỏ về class mapping và logic sẽ làm cho hệ thống hoạt động tốt hơn với model trained. License Plate Recognition là feature quan trọng tiếp theo cần implement để hoàn thiện hệ thống xử phạt.
