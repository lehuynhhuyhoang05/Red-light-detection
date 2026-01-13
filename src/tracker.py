"""
Object Tracking Module
Uses ByteTrack from Supervision library
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from loguru import logger
import supervision as sv
from .detector import Detection


@dataclass
class TrackedObject:
    """Container for tracked object"""
    track_id: int
    detection: Detection
    trajectory: List[tuple] = field(default_factory=list)  # History of center positions
    frame_count: int = 0
    is_lost: bool = False
    metadata: dict = field(default_factory=dict)
    
    def update_position(self, detection: Detection):
        """Update object position"""
        self.detection = detection
        self.trajectory.append(detection.center)
        self.frame_count += 1
        
        # Keep only last 30 positions
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)
    
    def get_velocity(self) -> Optional[tuple]:
        """Calculate velocity vector (vx, vy) in pixels/frame"""
        if len(self.trajectory) < 2:
            return None
        
        # Use last 5 frames for smoothing
        n = min(5, len(self.trajectory))
        dx = self.trajectory[-1][0] - self.trajectory[-n][0]
        dy = self.trajectory[-1][1] - self.trajectory[-n][1]
        
        return (dx / n, dy / n)
    
    def predict_position(self, n_frames: int = 1) -> tuple:
        """Predict future position based on velocity"""
        velocity = self.get_velocity()
        if velocity is None:
            return self.detection.center
        
        vx, vy = velocity
        current_x, current_y = self.detection.center
        
        pred_x = int(current_x + vx * n_frames)
        pred_y = int(current_y + vy * n_frames)
        
        return (pred_x, pred_y)


class ObjectTracker:
    """Multi-object tracker using ByteTrack"""
    
    def __init__(self, config: dict):
        self.config = config
        self.tracking_config = config.get('tracking', {})
        
        # Initialize ByteTrack
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.tracking_config.get('track_thresh', 0.5),
            lost_track_buffer=self.tracking_config.get('track_buffer', 30),
            minimum_matching_threshold=self.tracking_config.get('match_thresh', 0.8),
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        
        # Store tracked objects
        self.tracked_objects: dict[int, TrackedObject] = {}
        self.next_id = 0
        
        logger.info("Object tracker initialized")
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of TrackedObject with tracking IDs
        """
        if not detections:
            return []
        
        # Convert detections to supervision format
        xyxy = np.array([det.bbox for det in detections])
        confidence = np.array([det.confidence for det in detections])
        class_id = np.array([det.class_id for det in detections])
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Update tracker
        sv_detections = self.tracker.update_with_detections(sv_detections)
        
        # Create/update TrackedObject instances
        tracked_objects = []
        
        for i, track_id in enumerate(sv_detections.tracker_id):
            if track_id is None:
                continue
            
            track_id = int(track_id)
            
            # Get corresponding detection
            det = detections[i]
            
            # Update or create tracked object
            if track_id in self.tracked_objects:
                tracked_obj = self.tracked_objects[track_id]
                tracked_obj.update_position(det)
            else:
                tracked_obj = TrackedObject(
                    track_id=track_id,
                    detection=det
                )
                tracked_obj.trajectory.append(det.center)
                tracked_obj.frame_count = 1
                self.tracked_objects[track_id] = tracked_obj
            
            tracked_objects.append(tracked_obj)
        
        # Clean up lost tracks
        self._cleanup_lost_tracks(tracked_objects)
        
        return tracked_objects
    
    def _cleanup_lost_tracks(self, active_tracks: List[TrackedObject]):
        """Remove tracks that are no longer active"""
        active_ids = {obj.track_id for obj in active_tracks}
        lost_ids = set(self.tracked_objects.keys()) - active_ids
        
        for track_id in lost_ids:
            if track_id in self.tracked_objects:
                del self.tracked_objects[track_id]
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get tracked object by ID"""
        return self.tracked_objects.get(track_id)
    
    def reset(self):
        """Reset tracker"""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.tracking_config.get('track_thresh', 0.5),
            lost_track_buffer=self.tracking_config.get('track_buffer', 30),
            minimum_matching_threshold=self.tracking_config.get('match_thresh', 0.8),
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        self.tracked_objects.clear()
        logger.info("Tracker reset")


class TrajectoryAnalyzer:
    """Analyze object trajectories for violation detection"""
    
    @staticmethod
    def crosses_line(trajectory: List[tuple], line_y: int, 
                     direction: str = 'down') -> bool:
        """
        Check if trajectory crosses a horizontal line
        
        Args:
            trajectory: List of (x, y) positions
            line_y: Y coordinate of line
            direction: 'down' or 'up'
            
        Returns:
            True if crossed
        """
        if len(trajectory) < 2:
            return False
        
        for i in range(len(trajectory) - 1):
            y1 = trajectory[i][1]
            y2 = trajectory[i + 1][1]
            
            if direction == 'down':
                # Moving down (y increasing)
                if y1 < line_y <= y2:
                    return True
            else:
                # Moving up (y decreasing)
                if y1 > line_y >= y2:
                    return True
        
        return False
    
    @staticmethod
    def is_stopped(trajectory: List[tuple], threshold: int = 5) -> bool:
        """
        Check if object is stopped
        
        Args:
            trajectory: List of (x, y) positions
            threshold: Maximum movement in pixels
            
        Returns:
            True if stopped
        """
        if len(trajectory) < 5:
            return False
        
        # Check last 5 positions
        recent = trajectory[-5:]
        
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        return x_range < threshold and y_range < threshold
    
    @staticmethod
    def get_direction(trajectory: List[tuple]) -> str:
        """
        Get movement direction
        
        Returns:
            'up', 'down', 'left', 'right', 'stationary'
        """
        if len(trajectory) < 2:
            return 'stationary'
        
        dx = trajectory[-1][0] - trajectory[0][0]
        dy = trajectory[-1][1] - trajectory[0][1]
        
        if abs(dx) < 5 and abs(dy) < 5:
            return 'stationary'
        
        if abs(dy) > abs(dx):
            return 'down' if dy > 0 else 'up'
        else:
            return 'right' if dx > 0 else 'left'
