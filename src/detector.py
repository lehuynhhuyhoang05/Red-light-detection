"""
Object Detection Module - Standardized Interface
Supports YOLOv11, YOLO-NAS, RT-DETR
"""

import cv2
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from loguru import logger

# Class mapping cho model đã train
CLASS_NAMES = {
    0: 'car',
    1: 'green_light', 
    2: 'motobike',
    3: 'red_light',
    4: 'stop_line',
    5: 'yellow_light'
}

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

@dataclass
class Detection:
    """Detection result container"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int] = field(default=(0, 0))  # center x, y
    class_id: int = field(default=-1)  # Class ID từ model
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Set class_id từ class_name nếu chưa có
        if self.class_id == -1:
            self.class_id = CLASS_IDS.get(self.class_name, -1)


class BaseDetector(ABC):
    """Abstract Base Class for all detectors"""
    
    def __init__(self, config: dict):
        self.config = config
        if torch:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
            logger.warning("Torch not found, using CPU")
        self.class_names = CLASS_NAMES
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame
        Args:
            frame: BGR numpy array
        Returns:
            List of Detection objects
        """
        pass

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on class
            color = self._get_color(det.class_name)
            
            # Draw box
            thickness = 2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for each class"""
        colors = {
            'car': (255, 0, 0),           # Blue
            'motobike': (0, 255, 255),    # Yellow
            'red_light': (0, 0, 255),     # Red
            'yellow_light': (0, 255, 255), # Yellow
            'green_light': (0, 255, 0),   # Green
            'stop_line': (255, 255, 0),   # Cyan
        }
        return colors.get(class_name, (255, 255, 255))


class YOLOv11Detector(BaseDetector):
    """YOLOv11 Detector using Ultralytics"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv11 model"""
        try:
            from ultralytics import YOLO
            
            model_config = self.config['model']['yolov11']
            weights_path = model_config['weights']
            
            # Convert to absolute path
            weights_path = Path(weights_path)
            if not weights_path.is_absolute():
                # Try relative to project root
                project_root = Path(__file__).parent.parent
                weights_path = project_root / weights_path
            
            if not weights_path.exists():
                logger.error(f"Weights not found at {weights_path}")
                raise FileNotFoundError(f"Model weights not found: {weights_path}")
            
            self.model = YOLO(str(weights_path))
            self.model.to(self.device)
            
            self.img_size = model_config.get('img_size', 640)
            self.conf_threshold = model_config.get('conf_threshold', 0.5)
            self.iou_threshold = model_config.get('iou_threshold', 0.45)
            
            logger.info(f"YOLOv11 model loaded: {weights_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv11: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using YOLOv11"""
        try:
            results = self.model(
                frame,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            
            detections = []
            
            if results.boxes is not None:
                boxes = results.boxes.cpu().numpy()
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


class YOLONASDetector(BaseDetector):
    """YOLO-NAS Detector using SuperGradients"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._load_model()
    
    def _load_model(self):
        """Load YOLO-NAS model"""
        try:
            from super_gradients.training import models
            
            model_config = self.config['model']['yolo_nas']
            variant = model_config['variant']
            weights_path = model_config.get('weights', None)
            
            if weights_path and Path(weights_path).exists():
                self.model = models.get(variant, checkpoint_path=weights_path)
            else:
                logger.warning(f"Using pretrained {variant}")
                self.model = models.get(variant, pretrained_weights="coco")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.img_size = model_config.get('img_size', 640)
            self.conf_threshold = model_config.get('conf_threshold', 0.5)
            
            logger.info(f"YOLO-NAS model loaded: {variant}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-NAS: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using YOLO-NAS"""
        try:
            predictions = self.model.predict(frame, conf=self.conf_threshold)
            pred = predictions._images_prediction_lst[0]
            
            detections = []
            
            if pred.prediction.bboxes_xyxy is not None:
                boxes = pred.prediction.bboxes_xyxy
                labels = pred.prediction.labels
                confidences = pred.prediction.confidence
                
                for box, label, conf in zip(boxes, labels, confidences):
                    class_id = int(label)
                    confidence = float(conf)
                    x1, y1, x2, y2 = map(int, box)
                    
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


class RTDETRDetector(BaseDetector):
    """RT-DETR Detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._load_model()
    
    def _load_model(self):
        """Load RT-DETR model"""
        try:
            from ultralytics import RTDETR
            
            model_config = self.config['model']['rt_detr']
            weights_path = model_config['weights']
            
            if not Path(weights_path).exists():
                logger.warning(f"Weights not found, using pretrained")
                variant = model_config['variant']
                weights_path = f"{variant}.pt"
            
            self.model = RTDETR(weights_path)
            self.model.to(self.device)
            
            self.img_size = model_config.get('img_size', 640)
            self.conf_threshold = model_config.get('conf_threshold', 0.5)
            
            logger.info(f"RT-DETR model loaded: {weights_path}")
            
        except Exception as e:
            logger.error(f"Failed to load RT-DETR: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using RT-DETR"""
        try:
            results = self.model(
                frame,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                verbose=False
            )[0]
            
            detections = []
            
            if results.boxes is not None:
                boxes = results.boxes.cpu().numpy()
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


def create_detector(config: dict) -> BaseDetector:
    """Factory function to create detector based on config"""
    model_type = config.get('model', {}).get('type', 'yolov11').lower()
    
    detectors = {
        'yolov11': YOLOv11Detector,
        'yolo-nas': YOLONASDetector,
        'yolonas': YOLONASDetector,
        'rt-detr': RTDETRDetector,
        'rtdetr': RTDETRDetector,
    }
    
    detector_class = detectors.get(model_type)
    if detector_class is None:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(detectors.keys())}")
    
    return detector_class(config)
