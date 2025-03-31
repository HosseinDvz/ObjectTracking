from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from detection import BoundingBox, Detection, DetectionList

@dataclass
class Track:
    """Represents a tracked object across multiple frames"""
    id: int
    bbox: BoundingBox
    class_id: int
    age: int = 1  # How many frames this track has existed
    time_since_update: int = 0  # Frames since last matched with detection
    
    def update(self, detection: Detection) -> None:
        """Update track with new detection"""
        self.bbox = detection.bbox
        self.age += 1
        self.time_since_update = 0
    
    def predict(self) -> None:
        """Simple prediction (no motion model, just increment counters)"""
        self.time_since_update += 1
    
    def get_state(self) -> Tuple[BoundingBox, int]:
        """Get current state for visualization/output"""
        return (self.bbox, self.id)


class TrackerBase(ABC):
    """Abstract base class for object trackers"""
    
    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id: int = 0
    
    @abstractmethod
    def match_detections_to_tracks(self, detections: DetectionList) -> Dict[int, int]:
        """
        Match detections to existing tracks
        Returns: Dict mapping detection indices to track indices
        """
        pass
    
    def update(self, detections: DetectionList) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections in current frame
            
        Returns:
            List of active tracks
        """
        # Predict new locations of tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matches = self.match_detections_to_tracks(detections)
        
        # Update matched tracks
        for det_idx, track_idx in matches.items():
            self.tracks[track_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        matched_det_indices = set(matches.keys())
        for i in range(len(detections)):
            if i not in matched_det_indices:
                self.tracks.append(Track(
                    id=self.next_id,
                    bbox=detections[i].bbox,
                    class_id=detections[i].class_id
                ))
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        return self.tracks
    
    @property
    def max_age(self) -> int:
        """Maximum time since update before a track is removed"""
        return 1  # Base implementation keeps tracks for 1 frame


class IOUTracker(TrackerBase):
    """IOU-based tracker (similar to original SimpleTracker)"""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 1):
        super().__init__()
        self.iou_threshold = iou_threshold
        self._max_age = max_age
    
    def match_detections_to_tracks(self, detections: DetectionList) -> Dict[int, int]:
        """Match detections to tracks using IOU"""
        matches = {}
        
        # For each detection, find best matching track
        for det_idx, detection in enumerate(detections):
            best_track_idx = None
            best_iou = self.iou_threshold
            
            for track_idx, track in enumerate(self.tracks):
                # Skip already matched tracks
                if track_idx in matches.values():
                    continue
                    
                iou = detection.bbox.iou(track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx is not None:
                matches[det_idx] = best_track_idx
                
        return matches
    
    @property
    def max_age(self) -> int:
        return self._max_age


class KalmanTracker(TrackerBase):
    """
    Kalman filter-based tracker for smoother tracking
    Requires opencv-contrib-python for Kalman filter implementation
    """
    
    def __init__(self, max_age: int = 10, min_hits: int = 3):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("opencv-contrib-python is required for KalmanTracker")
            
        super().__init__()
        self._max_age = max_age
        self.min_hits = min_hits  # Minimum hits before track is displayed
        self.kalman_filters = {}  # Track ID -> Kalman filter
        
    def create_kalman_filter(self) -> 'cv2.KalmanFilter':
        """Create a Kalman filter for tracking in 2D space"""
        kf = self.cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], 
                                       [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        return kf
        
    def match_detections_to_tracks(self, detections: DetectionList) -> Dict[int, int]:
        """Match detections to tracks using both IOU and predicted position"""
        matches = {}
        
        # Use IOU as base matching strategy
        for det_idx, detection in enumerate(detections):
            best_track_idx = None
            best_iou = 0.3  # IOU threshold
            
            for track_idx, track in enumerate(self.tracks):
                # Skip already matched tracks
                if track_idx in matches.values():
                    continue
                    
                iou = detection.bbox.iou(track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx is not None:
                matches[det_idx] = best_track_idx
                
        return matches
    
    def update(self, detections: DetectionList) -> List[Track]:
        """Override update to handle Kalman prediction/correction"""
        # Similar to base update but with Kalman filter logic
        # Predict new locations of tracks using Kalman
        for track in self.tracks:
            if track.id in self.kalman_filters:
                kf = self.kalman_filters[track.id]
                predicted = kf.predict()
                center_x, center_y = predicted[0, 0], predicted[1, 0]
                # Update bbox center while maintaining width/height
                width, height = track.bbox.width, track.bbox.height
                track.bbox = BoundingBox(
                    center_x - width/2, center_y - height/2,
                    center_x + width/2, center_y + height/2
                )
            track.predict()
        
        # Match detections to tracks
        matches = self.match_detections_to_tracks(detections)
        
        # Update matched tracks with Kalman correction
        for det_idx, track_idx in matches.items():
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            track.update(detection)
            
            # Create or update Kalman filter
            if track.id not in self.kalman_filters:
                self.kalman_filters[track.id] = self.create_kalman_filter()
                
            # Correct Kalman estimate with measurement
            center_x, center_y = detection.bbox.center
            measurement = np.array([[center_x], [center_y]], np.float32)
            self.kalman_filters[track.id].correct(measurement)
        
        # Create new tracks for unmatched detections
        matched_det_indices = set(matches.keys())
        for i in range(len(detections)):
            if i not in matched_det_indices:
                new_track = Track(
                    id=self.next_id,
                    bbox=detections[i].bbox,
                    class_id=detections[i].class_id
                )
                self.tracks.append(new_track)
                # Create new Kalman filter
                self.kalman_filters[self.next_id] = self.create_kalman_filter()
                # Initialize with first position
                center_x, center_y = detections[i].bbox.center
                measurement = np.array([[center_x], [center_y]], np.float32)
                self.kalman_filters[self.next_id].correct(measurement)
                self.next_id += 1
        
        # Remove old tracks and their Kalman filters
        new_tracks = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                new_tracks.append(track)
            else:
                # Remove Kalman filter for deleted track
                if track.id in self.kalman_filters:
                    del self.kalman_filters[track.id]
        
        self.tracks = new_tracks
        return [t for t in self.tracks if t.age >= self.min_hits]
    
    @property
    def max_age(self) -> int:
        return self._max_age 