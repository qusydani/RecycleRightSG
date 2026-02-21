# src/predictor.py
from dataclasses import dataclass
from typing import List, Dict
from PIL import Image
from ultralytics import YOLO

from src.labels import BLUE_BIN_OK

@dataclass
class Detection:
    label: str
    confidence: float
    box: List[float] # [x1, y1, width, height]
    advice: str

class TrashnetPredictor:
    def __init__(self, model_path: str = "yolov8n.pt", abstain_threshold: float = 0.55):
        # We will use the base YOLOv8 nano model 
        self.model = YOLO(model_path) 
        self.abstain_threshold = abstain_threshold

    def predict_pil(self, img: Image.Image) -> List[Detection]:
        # Run YOLO inference
        results = self.model(img, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.abstain_threshold:
                    continue # Skip low confidence detections

                # Get class label
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                # Get bounding box coordinates (xywh format: x-center, y-center, width, height)
                # We convert to top-left x, top-left y, width, height for the HTML canvas
                xyxy = box.xyxy[0].tolist() 
                x1, y1, x2, y2 = xyxy
                box_coords = [x1, y1, x2 - x1, y2 - y1]

                # Generate advice
                if label in BLUE_BIN_OK or label in ['bottle', 'cup']: # Added generic YOLO classes for testing
                    advice = "Blue bin OK if clean & dry."
                else:
                    advice = "Do not put in blue bin."

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    box=box_coords,
                    advice=advice
                ))
                
        return detections