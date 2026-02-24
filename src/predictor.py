# src/predictor.py
from dataclasses import dataclass
from typing import List, Dict
from PIL import Image
from ultralytics import YOLO

@dataclass
class Detection:
    label: str
    confidence: float
    box: List[float] # [x, y, width, height]
    advice: str

# This maps the YOLO visual classes to the specific NEA guidelines
NEA_ADVICE_MAP = {
    # PAPER
    "cardboard": "Blue Bin OK! Please flatten before recycling.",
    "paper": "Blue Bin OK! Make sure it is clean before recycling.",
    
    # PLASTIC & FOAM
    "plastic": "Blue Bin OK! Please empty and rinse before recycling. Foil packaging goes to general waste.",
    "styrofoam": "DO NOT PUT IN BLUE BIN! Dispose as general waste.",
    
    # GLASS
    "glass": "Blue Bin OK! Please empty and rinse. Pyrex, tempered glass, and ceramics go to general waste.",
    
    # METAL
    "metal": "Blue Bin OK! Please empty and rinse when necessary. Rusted metals are general waste.",
    "aluminium": "Blue Bin OK! Please empty and rinse before recycling.",
    
    # E-WASTE & OTHERS
    "e-waste": "DO NOT PUT IN BLUE BIN! Can be recycled at e-waste collection points.",
    "clothing": "DO NOT PUT IN BLUE BIN! Donate if in good condition."
}

# Fallback advice if the dataset has a class we didn't explicitly map
FALLBACK_ADVICE = "Not sure, don’t risk contaminating the blue bin."

class TrashnetPredictor:
    def __init__(self, model_path: str = "models/best.pt", abstain_threshold: float = 0.55):
        # Once trained, this will load your custom YOLO model
        self.model = YOLO(model_path) 
        self.abstain_threshold = abstain_threshold

    def predict_pil(self, img: Image.Image) -> List[Detection]:
        results = self.model(img, verbose=False)
        
        # Get the total area of the image
        img_width, img_height = img.size
        img_area = img_width * img_height
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.abstain_threshold:
                    continue

                # Get coordinates
                xyxy = box.xyxy[0].tolist() 
                x1, y1, x2, y2 = xyxy
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # Ignore boxes that cover more than 95% of the screen
                if box_area > (0.95 * img_area):
                    continue

                class_id = int(box.cls[0])
                label = self.model.names[class_id] 
                advice = NEA_ADVICE_MAP.get(label, FALLBACK_ADVICE)

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    box=[x1, y1, box_width, box_height],
                    advice=advice
                ))
                
        return detections