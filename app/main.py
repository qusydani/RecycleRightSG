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
    <!DOCTYPE html>
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BlueBin Buddy</title>
        <style>
            body { font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            #video-container { margin-top: 20px; }
            video { width: 100%; max-width: 400px; border-radius: 8px; background: #000; }
            #result { margin-top: 15px; padding: 10px; border-radius: 5px; background: #f0f0f0; min-height: 50px;}
            .hidden { display: none; }
            button { padding: 10px; margin-top: 10px; cursor: pointer; }
        </style>
      </head>
      <body>
        <h3>BlueBin Buddy</h3>
        
        <div>
            <button onclick="setMode('upload')">Upload Photo</button>
            <button onclick="setMode('camera')">Live Camera</button>
        </div>

        <div id="upload-section">
            <form action="/predict" enctype="multipart/form-data" method="post" style="margin-top:20px;">
              <input name="file" type="file" accept="image/*"/>
              <input type="submit" value="Analyze"/>
            </form>
        </div>

        <div id="camera-section" class="hidden">
            <div id="video-container">
                <video id="video" autoplay playsinline></video>
            </div>
            <button id="start-cam-btn" onclick="startCamera()">Start Camera</button>
            <button id="stop-cam-btn" class="hidden" onclick="stopCamera()">Stop Camera</button>
            <canvas id="canvas" class="hidden"></canvas>
        </div>

        <div id="result" class="hidden">
            <strong>Prediction:</strong> <span id="pred-label">-</span><br>
            <strong>Confidence:</strong> <span id="pred-conf">-</span><br>
            <strong>Advice:</strong> <span id="pred-advice">-</span>
        </div>

        <p><a href="/docs">View API Docs</a></p>

        <script>
            let videoStream = null;
            let intervalId = null;

            function setMode(mode) {
                document.getElementById('upload-section').classList.toggle('hidden', mode !== 'upload');
                document.getElementById('camera-section').classList.toggle('hidden', mode !== 'camera');
                if (mode === 'upload') stopCamera();
            }

            async function startCamera() {
                const video = document.getElementById('video');
                try {
                    // facingMode: "environment" prefers the back camera on mobile phones
                    videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
                    video.srcObject = videoStream;
                    
                    document.getElementById('start-cam-btn').classList.add('hidden');
                    document.getElementById('stop-cam-btn').classList.remove('hidden');
                    document.getElementById('result').classList.remove('hidden');

                    // Start capturing frames every 1.5 seconds
                    intervalId = setInterval(captureAndPredict, 1500);
                } catch (err) {
                    alert("Error accessing camera: " + err.message);
                }
            }

            function stopCamera() {
                if (videoStream) {
                    videoStream.getTracks().forEach(track => track.stop());
                }
                if (intervalId) clearInterval(intervalId);
                document.getElementById('start-cam-btn').classList.remove('hidden');
                document.getElementById('stop-cam-btn').classList.add('hidden');
            }

            async function captureAndPredict() {
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Convert canvas to Blob and send to FastAPI
                    canvas.toBlob(async (blob) => {
                        const formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');

                        try {
                            const response = await fetch('/predict', {
                                method: 'POST',
                                body: formData
                            });
                            const data = await response.json();
                            
                            // Update UI
                            document.getElementById('pred-label').innerText = data.predicted_label;
                            document.getElementById('pred-conf').innerText = (data.confidence * 100).toFixed(1) + "%";
                            document.getElementById('pred-advice').innerText = data.advice;
                            
                        } catch (e) {
                            console.error("Prediction error:", e);
                        }
                    }, 'image/jpeg', 0.8); // 0.8 is compression quality to save bandwidth
                }
            }
        </script>
      </body>
    </html>
    """