# app/feedback.py
from pydantic import BaseModel
from fastapi import APIRouter
from pathlib import Path
import csv
from datetime import datetime

router = APIRouter()
LOG_PATH = Path("reports/feedback.csv")

class FeedbackIn(BaseModel):
    filename: str | None = None
    predicted_label: str
    correct_label: str
    confidence: float
    abstained: bool

@router.post("/feedback")
def feedback(item: FeedbackIn):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts", "filename", "predicted_label", "correct_label", "confidence", "abstained"])
        w.writerow([datetime.utcnow().isoformat(), item.filename, item.predicted_label, item.correct_label, item.confidence, item.abstained])

    return {"ok": True}
