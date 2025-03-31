import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from detection import BoundingBox, Detection
from tracker import Track
from camera_controller import CameraController

class Visualizer:
    """Handles visualization of detections, tracks, and camera guidance"""
    
    def __init__(self, 
                 frame_width: int, 
                 frame_height: int,
                 colors: Optional[Dict[int, Tuple[int, int, int]]] = None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Default colors for different object classes
        self.colors = colors or {
            1: (0, 255, 0),  # Green for person
            2: (255, 0, 0),  # Blue for bicycle
            3: (0, 0, 255),  # Red for car
            0: (255, 255, 0)  # Cyan for others
        }
        
        self.camera_controller = CameraController(frame_width, frame_height)
        
    def draw_detection(self, 
                      frame: np.ndarray, 
                      detection: Detection, 
                      color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Draw a single detection on the frame"""
        if color is None:
            color = self.colors.get(detection.class_id, self.colors[0])
            
        x1, y1, x2, y2 = detection.bbox.as_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        confidence_text = f"{detection.confidence:.2f}"
        cv2.putText(frame, confidence_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                   
        return frame
        
    def draw_track(self, 
                  frame: np.ndarray, 
                  track: Track,
                  color: Optional[Tuple[int, int, int]] = None,
                  highlight: bool = False) -> np.ndarray:
        """Draw a tracked object on the frame"""
        if color is None:
            color = self.colors.get(track.class_id, self.colors[0])
            
        # Draw thicker rectangle for highlighted tracks
        thickness = 3 if highlight else 2
            
        x1, y1, x2, y2 = track.bbox.as_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw ID and age
        text = f"ID: {track.id} Age: {track.age}"
        cv2.putText(frame, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw center point
        center_x, center_y = int(track.bbox.center[0]), int(track.bbox.center[1])
        cv2.circle(frame, (center_x, center_y), 4, color, -1)
                   
        return frame
        
    def draw_camera_guidance(self, 
                            frame: np.ndarray, 
                            tracks: List[Track]) -> np.ndarray:
        """Draw camera movement guidance based on tracked objects"""
        message, (dx, dy, dir_x, dir_y) = self.camera_controller.get_movement_commands(tracks)
        
        # Draw frame center crosshair
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
        
        # Draw deadzone rectangle
        deadzone_x = self.camera_controller.deadzone_x
        deadzone_y = self.camera_controller.deadzone_y
        cv2.rectangle(frame, 
                     (center_x - deadzone_x, center_y - deadzone_y),
                     (center_x + deadzone_x, center_y + deadzone_y),
                     (0, 255, 255), 1)
        
        # Draw movement messages
        cv2.putText(frame, message, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        
        # Draw direction arrows
        if dir_x != "none":
            arrow_x = 20 if dir_x == "left" else self.frame_width - 20
            cv2.arrowedLine(frame, 
                           (center_x, center_y),
                           (arrow_x, center_y),
                           (0, 255, 255), 2)
                           
        if dir_y != "none":
            arrow_y = 20 if dir_y == "up" else self.frame_height - 20
            cv2.arrowedLine(frame,
                           (center_x, center_y),
                           (center_x, arrow_y),
                           (0, 255, 255), 2)
        
        return frame
        
    def visualize(self, 
                 frame: np.ndarray,
                 tracks: List[Track],
                 detections: Optional[List[Detection]] = None) -> np.ndarray:
        """Create visualization with all elements"""
        result = frame.copy()
        
        # Draw all detections if provided
        if detections:
            for detection in detections:
                result = self.draw_detection(result, detection)
        
        # Find target (largest person)
        target = self.camera_controller.select_target(tracks)
        
        # Draw all tracks
        for track in tracks:
            highlight = target is not None and track.id == target.id
            result = self.draw_track(result, track, highlight=highlight)
            
        # Draw camera guidance
        result = self.draw_camera_guidance(result, tracks)
        
        return result 