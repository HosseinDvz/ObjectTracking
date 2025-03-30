import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2

# ─────────────────────────────────────────────
# 1. Setup Device (MPS or CPU)
# ─────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ─────────────────────────────────────────────
# 2. Load Lightweight Detector (SSDlite)
# ─────────────────────────────────────────────
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.to(device).eval()

# ─────────────────────────────────────────────
# 3. ByteTrack-Style Tracker (Simple IOU Tracker)
# ─────────────────────────────────────────────
class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3):
        self.next_id = 0
        self.tracks = []  # (box, id)
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for (track_box, track_id) in self.tracks:
                if self.iou(det, track_box) > self.iou_threshold:
                    updated_tracks.append((det, track_id))
                    matched = True
                    break
            if not matched:
                updated_tracks.append((det, self.next_id))
                self.next_id += 1
        self.tracks = updated_tracks
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), id) for b, id in self.tracks]

# ─────────────────────────────────────────────
# 4. Start Webcam
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
tracker = SimpleTracker()

print("Press Q to quit.")
width = 320
height = 320

# ─────────────────────────────────────────────
# 5. Main Loop: Human Detection + Tracking
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    original_frame = frame.copy()
    resized_frame = cv2.resize(frame, (width, height))

    # Convert to tensor
    img_tensor = F.to_tensor(resized_frame).unsqueeze(0).to(device)

    # Detect humans
    with torch.no_grad():
        output = model(img_tensor)[0]

    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()

    # Only keep high-confidence PERSON detections (COCO class ID 1)
    threshold = 0.5
    person_boxes = [
        [box[0], box[1], box[2], box[3]]
        for box, score, label in zip(boxes, scores, labels)
        if score > threshold and label == 1
    ]
    original_image_y = original_frame.shape[0]    
    original_image_x = original_frame.shape[1]
    # Scale boxes back to original frame size
    scale_x = original_image_x / width
    scale_y = original_image_y / height
    screen_center_x = original_image_x // 2
    screen_center_y = original_image_y // 2
    
    scaled_boxes = [
        [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y]
        for box in person_boxes
    ]

    # Track humans
    tracked = tracker.update(scaled_boxes)
    
    # Draw boxes and IDs
    max_area = 0
    p1 = (0, 0)
    p2 = (0, 0)
    p1_max = (0, 0)
    p2_max = (0, 0)
    id = -1
    for x1, y1, x2, y2, track_id in tracked:
        p1 = (x1, y1)
        p2 = (x2, y2)
        
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            id = track_id
            p1_max = p1
            p2_max = p2

        cv2.rectangle(original_frame, p1, p2, (0, 255, 0), 2)        
        cv2.putText(original_frame, f'Person ID: {track_id} Area {area}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    if id >= 0:
        center_x = p1[0] + (p2[0] - p1[0]) // 2
        center_y = p1[1] + (p2[1] - p1[1]) // 2
        cv2.rectangle(original_frame, p1_max, p2_max, (0, 0, 255), 2)
        cv2.putText(original_frame, f'Person ID: {id} Area {max_area} ({p1[0]}, {p1[1]}), ({p2[0]}, {p2[1]}) Center ({center_x}, {center_y})', (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        move_x = center_x - screen_center_x
        if move_x != 0:
            if move_x > 0:
                move_x_str = "left"
            else:
                move_x_str = "right"
                move_x = -move_x
            cv2.putText(original_frame, f'({screen_center_x}) Camera should move {move_x} {move_x_str}.', (0, original_image_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        move_y = center_y - screen_center_y
        if move_y != 0:
            if move_y > 0:
                move_y_str = "down"
            else:
                move_y_str = "up"
                move_y = -move_y
            cv2.putText(original_frame, f'({screen_center_y}) Camera should move {move_y} {move_y_str}.', (0, original_image_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    cv2.imshow("Tracking", original_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
