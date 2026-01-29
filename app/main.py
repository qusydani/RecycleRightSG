# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import io

from src.predictor import TrashnetPredictor
from app.feedback import router as feedback_router

app = FastAPI()
predictor = TrashnetPredictor("models/trashnet_mobilenetv2.pt", abstain_threshold=0.55)
app.include_router(feedback_router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    pred = predictor.predict_pil(img)
    return {
        "filename": file.filename,
        "predicted_label": pred.predicted_label,
        "confidence": pred.confidence,
        "top3": pred.top3,
        "abstained": pred.abstained,
        "advice": pred.advice,
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <body>
        <h3>BlueBin Buddy</h3>
        <form action="/predict" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept="image/*"/>
          <input type="submit"/>
        </form>
        <p>Then open <a href="/docs">/docs</a> for the interactive API.</p>
      </body>
    </html>
    """