from typing import Tuple, Optional, List
import math
from detection import BoundingBox, Detection
from tracker import Track


class CameraController:
    """
    Controls camera movement to follow a target
    
    Can be extended to interface with actual camera control hardware
    or used for visual guidance
    """
    
    def __init__(self, frame_width: int, frame_height: int, deadzone_percent: float = 0.1):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Deadzone is a region in the center where we don't move the camera
        # (reduces jitter)
        self.deadzone_x = int(frame_width * deadzone_percent)
        self.deadzone_y = int(frame_height * deadzone_percent)
        
    def select_target(self, tracks: List[Track]) -> Optional[Track]:
        """Select the track to follow (default: largest visible person)"""
        if not tracks:
            return None
            
        # Find track with the largest area
        return max(tracks, key=lambda t: t.bbox.area)
        
    def calculate_movement(self, target: Track) -> Tuple[int, int, str, str]:
        """
        Calculate how the camera should move to center the target
        
        Returns:
            (dx, dy, direction_x, direction_y)
        """
        center_x, center_y = target.bbox.center
        
        # Calculate movement needed
        dx = int(center_x - self.center_x)
        dy = int(center_y - self.center_y)
        
        # Apply deadzone
        if abs(dx) < self.deadzone_x:
            dx = 0
        if abs(dy) < self.deadzone_y:
            dy = 0
            
        # Determine directions
        direction_x = "right" if dx < 0 else "left" if dx > 0 else "none"
        direction_y = "up" if dy < 0 else "down" if dy > 0 else "none"
        
        return (abs(dx), abs(dy), direction_x, direction_y)
        
    def get_movement_commands(self, tracks: List[Track]) -> Tuple[str, Tuple[int, int, str, str]]:
        """
        Get camera movement commands based on tracked objects
        
        Returns:
            (status_message, (dx, dy, direction_x, direction_y))
        """
        target = self.select_target(tracks)
        
        if target is None:
            return "No target detected", (0, 0, "none", "none")
            
        dx, dy, dir_x, dir_y = self.calculate_movement(target)
        
        message = f"Target ID: {target.id}, Area: {target.bbox.area:.0f}"
        if dir_x != "none" or dir_y != "none":
            message += f" - Move: {dx if dir_x != 'none' else 0} {dir_x}, {dy if dir_y != 'none' else 0} {dir_y}"
            
        return message, (dx, dy, dir_x, dir_y) 