import torch
from torchvision.transforms import functional as F  # type: ignore
from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # type: ignore
import collections as cl
import numpy as np
import cv2

# Setup MPS - Metal Performance Shaders for Apple Silicon hardware acceleration
# Falls back to CPU if MPS is not available
if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.backends.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print("Using device:", device)

# Load Pretrained SSDlite object detector with MobileNetV3 backbone
# SSDlite is a lightweight single-shot detector optimized for mobile and edge devices
# It's pre-trained on COCO dataset which includes 'person' as class ID 1
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.to(device).eval()  # Move model to device and set to evaluation mode


class SimpleTracker:
  """
  Simple ByteTrack-style IOU tracker.

  This tracker associates detections between frames using Intersection over Union (IOU).
  New detections that overlap sufficiently with existing tracks are associated with those tracks.
  Detections without matches are assigned new track IDs.
  """

  def __init__(self, iou_threshold: float = 0.3):
    self.next_id = 0  # Counter for generating unique track IDs
    self.tracks = cl.deque()
    self.iou_threshold = iou_threshold  # Minimum IOU required to match detections to existing tracks

  def iou(self, boxA, boxB):
    """
    Calculate Intersection over Union between two bounding boxes.

    IOU = area of intersection / area of union
    Higher IOU means boxes overlap more.
    """
    # Find coordinates of intersection rectangle
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    # Calculate area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate areas of both bounding boxes
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))

    # Calculate IOU (add small epsilon to avoid division by zero)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

  def update(self, detections):
    """
    Update tracks with new detections from current frame.

    For each detection, try to match it with an existing track based on IOU.
    If matched, update the track's bounding box.
    If not matched, create a new track with a unique ID.

    Returns list of (x1, y1, x2, y2, track_id) for visualization.
    """
    updated_tracks = cl.deque()
    for det in detections:
      matched = False
      for (track_box, track_id) in self.tracks:
        if self.iou(det, track_box) > self.iou_threshold:
          updated_tracks.append((det, track_id))  # Update track with new detection
          matched = True
          break
      if not matched:
        updated_tracks.append((det, self.next_id))  # Create new track
        self.next_id += 1
    self.tracks = updated_tracks
    # Convert box coordinates to integers for drawing
    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), id) for b, id in self.tracks]


# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
tracker = SimpleTracker()

print("Press Q to quit.")
# Define inference size - smaller for faster processing
width = 320
height = 320

# Main Loop: Human Detection + Tracking
while cv2.waitKey(1) & 0xFF != ord('q'):
  ret, frame = cap.read()
  if not ret:
    break

  # Keep original frame for display and resize a copy for inference
  original_frame = frame.copy()
  resized_frame = cv2.resize(frame, (width, height))

  # Convert OpenCV BGR image to PyTorch tensor (add batch dimension, move to device)
  img_tensor = F.to_tensor(resized_frame).unsqueeze(0).to(device)

  # Run object detection model
  with torch.no_grad():  # Disable gradient calculations for inference
    output = model(img_tensor)[0]

  # Extract detection results
  boxes = output['boxes'].cpu().numpy()
  scores = output['scores'].cpu().numpy()
  labels = output['labels'].cpu().numpy()

  # Filter detections: keep only persons (class 1) with confidence above threshold
  threshold = 0.5
  person_boxes = [
      [box[0], box[1], box[2], box[3]]
      for box, score, label in zip(boxes, scores, labels)
      if score > threshold and label == 1
  ]

  # Calculate scaling factors to map detection boxes back to original frame size
  original_image_y = original_frame.shape[0]
  original_image_x = original_frame.shape[1]
  scale_x = original_image_x / width
  scale_y = original_image_y / height
  screen_center_x = original_image_x // 2
  screen_center_y = original_image_y // 2

  # Apply scaling to bounding boxes
  scaled_boxes = [
      [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y]
      for box in person_boxes
  ]

  # Update tracks with new detections
  tracked = tracker.update(scaled_boxes)

  # Visualization and camera guidance logic
  # The code finds the largest person (by bounding box area) and calculates
  # how a camera should move to center that person in the frame

  # Draw boxes and IDs
  # Initialize tracking variables
  max_area = 0
  p1 = (0, 0)
  p2 = (0, 0)
  p1_max = (0, 0)
  p2_max = (0, 0)
  id = -1
  # For every bounding box being tracked
  for x1, y1, x2, y2, track_id in tracked:
    p1 = (x1, y1)
    p2 = (x2, y2)
    # Calculate the area of the bounding box
    area = (x2 - x1) * (y2 - y1)
    # Find the bounding box with the largest area
    if area > max_area:
      max_area = area
      id = track_id
      p1_max = p1
      p2_max = p2

    # Draw a regular bounding box as white
    cv2.rectangle(original_frame, p1, p2, (255, 255, 255), 2)
    cv2.putText(original_frame, f'Person ID: {track_id} Area {area}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

  # If there is a largest bounding box, draw it and calculate camera movement
  if id >= 0:
    center_x = p1[0] + (p2[0] - p1[0]) // 2
    center_y = p1[1] + (p2[1] - p1[1]) // 2
    cv2.rectangle(original_frame, p1_max, p2_max, (0, 0, 255), 2)
    cv2.putText(original_frame, f'Person ID: {id} Area {max_area} ({p1[0]}, {p1[1]}), ({p2[0]}, {p2[1]}) Center ({center_x}, {center_y})', (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    move_x = center_x - screen_center_x
    # If the horizontal center is not in the center of the screen, calculate the movement
    if move_x != 0:
      if move_x > 0:
        move_x_str = "left"
      else:
        move_x_str = "right"
        move_x = -move_x
      cv2.putText(original_frame, f'({screen_center_x}) Camera should move {move_x} {move_x_str}.', (0, original_image_y - 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    move_y = center_y - screen_center_y
    # If the vertical center is not in the center of the screen, calculate the movement
    if move_y != 0:
      if move_y > 0:
        move_y_str = "down"
      else:
        move_y_str = "up"
        move_y = -move_y
      cv2.putText(original_frame, f'({screen_center_y}) Camera should move {move_y} {move_y_str}.', (0, original_image_y - 15),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
  cv2.imshow("Tracking", original_frame)

cap.release()
cv2.destroyAllWindows()
