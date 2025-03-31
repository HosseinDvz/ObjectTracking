from typing import Self
import numpy as np
import torch
from torchvision.transforms import functional as F  # type: ignore
from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # type: ignore
import collections as cl
import cv2

# Set MPS, CUDA or CPU
if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.backends.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print("Using device:", device)


class BoundingBox:
  """
  A bounding box is a list of 4 integers: [x1, y1, x2, y2]
  0 is x coordinate of top left point
  1 is y coordinate of top left point
  2 is x coordinate of bottom right point
  3 is y coordinate of bottom right point
  """

  def __init__(self, box: list[int] | None = None):
    if box is None:
      self.box = [0, 0, 0, 0]
      return
    self.box = list(map(int, box))

  def __getitem__(self, i: int):
    return self.box[i]

  def iou(self, other: Self) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    """
    # Find coordinates of intersection rectangle
    # lowest and rightmost top left point
    xA, yA = max(self[0], other[0]), max(self[1], other[1])
    # highest and leftmost bottom right point
    xB, yB = min(self[2], other[2]), min(self[3], other[3])

    # Calculate area of intersection
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Calculate areas of the two bounding boxes
    boxAArea = (self[2] - self[0]) * (self[3] - self[1])
    boxBArea = (other[2] - other[0]) * (other[3] - other[1])

    # Calculate IOU, with small epsilon to avoid division by zero
    return float(intersection_area) / (boxAArea + boxBArea - intersection_area + 1e-5)

  def scale(self, scale_x: float, scale_y: float) -> Self:
    """
    Scale the bounding box by a given scale factor.
    """
    self.box = [int(self[0] * scale_x), int(self[1] * scale_y), int(self[2] * scale_x), int(self[3] * scale_y)]
    return self

  def p1(self) -> tuple[int, int]:
    """
    Calculate the top left point of the bounding box.
    """
    return (self[0], self[1])

  def p2(self) -> tuple[int, int]:
    """
    Calculate the bottom right point of the bounding box.
    """
    return (self[2], self[3])

  def height(self) -> int:
    """
    Calculate the height of the bounding box.
    """
    return self[3] - self[1]

  def width(self) -> int:
    """
    Calculate the width of the bounding box.
    """
    return self[2] - self[0]

  def area(self) -> int:
    """
    Calculate the area of the bounding box.
    """
    return self.height() * self.width()

  def center(self) -> tuple[int, int]:
    """
    Calculate the center of the bounding box.
    """
    return (self[0] + self.width() // 2, self[1] + self.height() // 2)


class Tracked(BoundingBox):
  """
  A tracked object is a bounding box with a score and a label.
  """

  def __init__(self, box: list[int]):
    super().__init__(box)
    self.id = -1
    self.tracked = False

  def set_score_label(self, score: float, label: int) -> Self:
    self.score = score
    self.label = label
    return self


class SimpleTracker:
  """
  Intersection over Union tracker.

  Associates detections between frames using Intersection over Union.
  New detections that are sufficiently similar to existing tracked objects are associated with those tracked objects.
  Detections without matches are assigned new ids.
  """

  def __init__(self, iou_threshold: float = 0.3):
    self.next_id: int = 0  # Counter for generating new ids
    self.tracks: cl.deque[Tracked] = cl.deque()
    self.iou_threshold: float = iou_threshold  # Minimum IOU required to match detections to existing tracks

  def update(self, detections: list[Tracked]) -> cl.deque[Tracked]:
    """
    Update tracked objects with new detections from current frame.

    For each detection, try to match it with an existing tracked object based on IOU.
    If matched, update the tracked object's bounding box.
    If not matched, create a new tracked object with a unique ID.

    Returns list of tracked objects.
    """
    updated_tracks: cl.deque[Tracked] = cl.deque()
    for det in detections:
      for track_box in self.tracks:
        if track_box.tracked:
          continue
        if det.iou(track_box) > self.iou_threshold:
          det.id = track_box.id
          track_box.tracked = True
          break
      else:
        det.id = self.next_id
        self.next_id += 1
      updated_tracks.append(det)
    current_tracks = updated_tracks
    self.tracks = current_tracks
    return current_tracks


# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
tracker = SimpleTracker()

# Define inference size - smaller for faster processing
width = 320
height = 320

# Load Pretrained SSDlite object detector with MobileNetV3
# A lightweight single shot detector optimized for mobile and edge devices
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.to(device).eval()

white = (200, 200, 200)
blue = (200, 0, 0)
green = (0, 200, 0)
red = (0, 0, 200)

font = cv2.FONT_HERSHEY_SIMPLEX

def display_text(frame: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int], font_scale: float = 0.5, font_thickness: int = 2) -> np.ndarray:
  cv2.putText(frame, text, position, font, font_scale, color, font_thickness)
  return frame

print("Press Q to quit.")
# Detection and tracking
object_max = BoundingBox()
while cv2.waitKey(1) & 0xFF != ord('q'):
  ret, frame = cap.read()
  if not ret:
    break

  # Keep original frame for display and resize a copy for inference
  original_frame = frame.copy()
  resized_frame = cv2.resize(frame, (width, height))

  # Convert OpenCV BGR image to PyTorch tensor
  img_tensor = F.to_tensor(resized_frame).unsqueeze(0).to(device)

  # Perform inference with object detection model
  with torch.no_grad():
    output = model(img_tensor)[0]

  # Extract detection results
  boxes = output['boxes'].cpu().numpy()
  scores = output['scores'].cpu().numpy()
  labels = output['labels'].cpu().numpy()

  # Calculate scaling factors to map detection boxes back to original frame size
  # row
  original_image_y = original_frame.shape[0]
  # column
  original_image_x = original_frame.shape[1]
  scale_x = original_image_x / width
  scale_y = original_image_y / height

  # Filter detections: keep only persons (class 1) with confidence above threshold
  threshold = 0.4
  person_boxes = [
      Tracked(box).scale(scale_x, scale_y)
      for box, score, label in zip(boxes, scores, labels)
      if label == 1 and score > threshold
  ]

  screen_center_x = original_image_x // 2
  screen_center_y = original_image_y // 2

  # Update tracks with new detections
  tracked = tracker.update([person.set_score_label(score, label) for person, score, label in zip(person_boxes, scores, labels)])

  # Visualization and camera guidance logic
  # The code finds the largest person bounding box and calculates
  # how a camera should move to center that person in the frame

  # Draw boxes and IDs
  # Initialize tracking variables
  max_area = -1
  id = -1
  # For every bounding box being tracked
  for person in tracked:
    # Calculate the area of the bounding box
    area = person.area()
    # Find the bounding box with the largest area
    if area > max_area:
      max_area = area
      id = person.id
      object_max = person
    p1 = person.p1()
    p2 = person.p2()
    # Draw a regular bounding box as white
    print("1")
    print(p1, p2)
    cv2.rectangle(original_frame, p1, p2, white, 2)
    display_text(original_frame, f'Person ID: {person.id} Area {area}', (p1[0], p1[1] - 10), white)

  # If there is a largest bounding box, draw it and calculate camera movement
  if id >= 0:
    p1_max = object_max.p1()
    p2_max = object_max.p2()
    center_x = object_max.center()[0]
    center_y = object_max.center()[1]
    print("2")
    print(p1_max, p2_max)
    cv2.rectangle(original_frame, p1_max, p2_max, blue, 4)
    display_text(original_frame, f'Person ID: {id} Area {max_area} ({p1_max[0]}, {p1_max[1]}), ({p2_max[0]}, {p2_max[1]}) Center ({center_x}, {center_y})', (0, 15), blue)
    move_x = center_x - screen_center_x
    # If the horizontal center is not in the center of the screen, calculate the movement
    if move_x != 0:
      if move_x > 0:
        move_x_str = "left"
      else:
        move_x_str = "right"
        move_x = -move_x
      display_text(original_frame, f'({screen_center_x}) Camera should move {move_x} {move_x_str}.', (0, original_image_y - 30), green)
    move_y = center_y - screen_center_y
    # If the vertical center is not in the center of the screen, calculate the movement
    if move_y != 0:
      if move_y > 0:
        move_y_str = "down"
      else:
        move_y_str = "up"
        move_y = -move_y
      display_text(original_frame, f'({screen_center_y}) Camera should move {move_y} {move_y_str}.', (0, original_image_y - 15), red)
  cv2.imshow("Tracking", original_frame)

cap.release()
cv2.destroyAllWindows()
