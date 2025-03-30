import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from detection import BoundingBox, Detection, DetectionList
from tracker import IOUTracker, KalmanTracker
from visualization import Visualizer

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.to(device).eval()

# Initialize tracker (choose one)
# tracker = IOUTracker(iou_threshold=0.3, max_age=5)
tracker = KalmanTracker(max_age=10, min_hits=3)

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize visualizer
visualizer = Visualizer(640, 480)

# Define inference size
width, height = 320, 320

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Keep original frame for display and resize for inference
    original_frame = frame.copy()
    resized_frame = cv2.resize(frame, (width, height))
    
    # Convert to tensor
    img_tensor = F.to_tensor(resized_frame).unsqueeze(0).to(device)
    
    # Run detection
    with torch.no_grad():
        output = model(img_tensor)[0]
        
    # Extract results
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    
    # Convert to detections
    detections = DetectionList()
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5 and label == 1:  # Person class
            # Scale to original frame
            scale_x = original_frame.shape[1] / width
            scale_y = original_frame.shape[0] / height
            
            scaled_box = BoundingBox(
                box[0] * scale_x, 
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y
            )
            
            detections.add(Detection(
                bbox=scaled_box,
                confidence=float(score),
                class_id=int(label)
            ))
    
    # Update tracker
    tracks = tracker.update(detections)
    
    # Visualize
    result = visualizer.visualize(original_frame, tracks, detections)
    
    # Show result
    cv2.imshow("Object Tracking", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows() 