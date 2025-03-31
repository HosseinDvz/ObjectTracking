from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates [x1, y1, x2, y2]"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)
    
    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))
    
    @staticmethod
    def from_list(coords: List[float]) -> 'BoundingBox':
        return BoundingBox(coords[0], coords[1], coords[2], coords[3])
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box"""
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        bb1_area = self.area
        bb2_area = other.area
        
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area + 1e-6)
        return iou


@dataclass
class Detection:
    """Represents a single object detection"""
    bbox: BoundingBox
    confidence: float
    class_id: int
    
    @property
    def area(self) -> float:
        return self.bbox.area
    
    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center


class DetectionList:
    """Container for multiple detections with filtering capabilities"""
    def __init__(self, detections: List[Detection] = None):
        self.detections = detections or []
    
    def add(self, detection: Detection) -> None:
        self.detections.append(detection)
    
    def filter_by_class(self, class_id: int) -> 'DetectionList':
        return DetectionList([d for d in self.detections if d.class_id == class_id])
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionList':
        return DetectionList([d for d in self.detections if d.confidence >= threshold])
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def __getitem__(self, idx: int) -> Detection:
        return self.detections[idx]
    
    def __iter__(self):
        return iter(self.detections) 