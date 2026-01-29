# src/predictor.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from src.labels import CLASSES, BLUE_BIN_OK

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

@dataclass
class Prediction:
    predicted_label: str
    confidence: float
    top3: List[Dict]
    advice: str
    abstained: bool

class TrashnetPredictor:
    def __init__(self, ckpt_path: str, device: str | None = None, abstain_threshold: float = 0.55):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.abstain_threshold = abstain_threshold

        ckpt = torch.load(ckpt_path, map_location=self.device)
        model_name = ckpt.get("model_name", "mobilenet_v2")
        classes = ckpt.get("classes", CLASSES)
        assert classes == CLASSES, f"Checkpoint classes {classes} != {CLASSES}"

        if model_name != "mobilenet_v2":
            raise ValueError(f"Unsupported model_name: {model_name}")

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model.to(self.device)

        self.model = model
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.inference_mode()
    def predict_pil(self, img: Image.Image) -> Prediction:
        if img.mode != "RGB":
            img = img.convert("RGB")

        x = self.preprocess(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

        conf, idx = torch.max(probs, dim=0)
        conf = float(conf.item())
        idx = int(idx.item())
        label = CLASSES[idx]

        topk = torch.topk(probs, k=3)
        top3 = [{"label": CLASSES[int(i)], "p": float(p)} for p, i in zip(topk.values, topk.indices)]

        abstained = conf < self.abstain_threshold
        if abstained:
            advice = "Not sure — don’t risk contaminating the blue bin."
        else:
            if label in BLUE_BIN_OK:
                advice = "Blue bin OK if clean & dry (rinse/dry if needed)."
            else:
                advice = "Do not put in blue bin."

        return Prediction(
            predicted_label=label,
            confidence=conf,
            top3=top3,
            advice=advice,
            abstained=abstained,
        )
