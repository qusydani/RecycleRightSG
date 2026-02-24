# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import io

from src.predictor import TrashnetPredictor
from app.feedback import router as feedback_router

app = FastAPI()
predictor = TrashnetPredictor("models/best.pt", abstain_threshold=0.40)
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

    detections = predictor.predict_pil(img)
    
    return {
        "filename": file.filename,
        "detections": [
            {
                "label": d.label,
                "confidence": d.confidence,
                "box": d.box,
                "advice": d.advice
            } for d in detections
        ]
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>BlueBin Buddy</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #0056b3; /* Blue Bin Blue */
                --primary-hover: #004494;
                --success: #28a745; /* Eco Green */
                --bg-color: #f4f7f6;
                --card-bg: #ffffff;
                --text-main: #333333;
                --text-muted: #666666;
            }

            * { box-sizing: border-box; }

            body { 
                font-family: 'Inter', sans-serif; 
                background-color: var(--bg-color); 
                color: var(--text-main);
                margin: 0; 
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            /* Header Styling */
            .header {
                background-color: var(--primary);
                color: white;
                width: 100%;
                padding: 20px;
                text-align: center;
                font-size: 1.5rem;
                font-weight: 700;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }

            .container {
                width: 100%;
                max-width: 500px;
                padding: 0 20px;
            }

            /* Card Layout */
            .card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                margin-bottom: 20px;
            }

            /* Button Styling */
            .btn-group {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }

            button { 
                flex: 1;
                padding: 14px; 
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer; 
                transition: all 0.2s ease;
                background-color: #e2e8f0;
                color: var(--text-main);
            }

            button.active {
                background-color: var(--primary);
                color: white;
            }

            button:active { transform: scale(0.98); }

            .action-btn {
                width: 100%;
                background-color: var(--success);
                color: white;
                margin-top: 15px;
            }

            .action-btn.stop {
                background-color: #dc3545; /* Red for stop */
            }

            /* Video and Image Containers */
            #video-container, #upload-container { 
                position: relative; 
                width: 100%; 
                border-radius: 8px;
                overflow: hidden;
                background: #000;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }

            video, #upload-img { 
                width: 100%; 
                display: block; 
            }

            canvas { 
                position: absolute; 
                top: 0; 
                left: 0; 
                width: 100%; 
                height: 100%; 
                pointer-events: none; 
            }

            /* File Input Styling */
            input[type="file"] { 
                width: 100%;
                padding: 10px;
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                background: #f8fafc;
                margin-bottom: 15px;
                color: var(--text-muted);
            }

            /* Results Section */
            #result { 
                background: #e6f4ea; 
                border-left: 5px solid var(--success);
                padding: 15px; 
                border-radius: 8px;
                font-size: 0.95rem;
                line-height: 1.5;
            }

            .hidden { display: none !important; }
            .links { text-align: center; margin-top: 20px; font-size: 0.9rem; }
            .links a { color: var(--primary); text-decoration: none; }
        </style>
      </head>
      <body>
        
        <div class="header">
            BlueBin Buddy
        </div>
        
        <div class="container">
            <div class="card">
                <div class="btn-group">
                    <button id="tab-upload" class="active" onclick="setMode('upload')">Upload Photo</button>
                    <button id="tab-camera" onclick="setMode('camera')">Live Camera</button>
                </div>

                <div id="upload-section">
                    <input id="upload-file" type="file" accept="image/*" onchange="previewAndPredict()"/>
                    
                    <div id="upload-container" class="hidden">
                        <img id="upload-img">
                        <canvas id="upload-canvas"></canvas>
                    </div>
                </div>

                <div id="camera-section" class="hidden">
                    <div id="video-container">
                        <video id="video" autoplay playsinline></video>
                        <canvas id="canvas" class="hidden"></canvas>
                    </div>
                    <button id="start-cam-btn" class="action-btn" onclick="startCamera()">Start Camera</button>
                    <button id="stop-cam-btn" class="action-btn stop hidden" onclick="stopCamera()">Stop Camera</button>
                </div>
            </div>

            <div id="result" class="card hidden"></div>

            <div class="links">
                <p><a href="/docs">Developer API Specs</a></p>
            </div>
        </div>

        <script>
            let videoStream = null;
            let intervalId = null;

            function setMode(mode) {
                // Toggle UI Sections
                document.getElementById('upload-section').classList.toggle('hidden', mode !== 'upload');
                document.getElementById('camera-section').classList.toggle('hidden', mode !== 'camera');
                document.getElementById('result').classList.add('hidden'); 
                
                // Toggle Button active states for styling
                document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
                document.getElementById('tab-camera').classList.toggle('active', mode === 'camera');
                
                if (mode === 'upload') stopCamera();
            }

            // --- UPLOAD LOGIC ---
            async function previewAndPredict() {
                const fileInput = document.getElementById('upload-file');
                if (!fileInput.files || fileInput.files.length === 0) return;
                
                const file = fileInput.files[0];
                const imgElement = document.getElementById('upload-img');
                const container = document.getElementById('upload-container');
                const canvas = document.getElementById('upload-canvas');
                const resultDiv = document.getElementById('result');
                
                container.classList.remove('hidden');
                resultDiv.classList.remove('hidden');
                resultDiv.innerHTML = "<strong> Analyzing uploaded image...</strong>";
                
                imgElement.onload = async () => {
                    canvas.width = imgElement.naturalWidth;
                    canvas.height = imgElement.naturalHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        
                        let resultsHtml = '';
                        
                        if (data.detections && data.detections.length > 0) {
                            data.detections.forEach(det => {
                                const [x, y, w, h] = det.box;
                                const scale = canvas.width / 400; 
                                const fontSize = Math.max(20, 15 * scale);
                                const padding = 10 * scale;
                                
                                ctx.strokeStyle = '#00FF00';
                                ctx.lineWidth = Math.max(4, 3 * scale);
                                ctx.strokeRect(x, y, w, h);
                                
                                const confPercent = (det.confidence * 100).toFixed(0);
                                const labelText = `${det.label} ${confPercent}%`;
                                
                                ctx.font = `${fontSize}px Arial`;
                                ctx.textBaseline = 'top'; 
                                
                                const textWidth = ctx.measureText(labelText).width;
                                const labelWidth = textWidth + (padding * 2);
                                const labelHeight = fontSize + padding;
                                
                                let labelX = x;
                                let labelY = y - labelHeight;
                                
                                if (labelY < 0) labelY = y; 
                                if (labelY + labelHeight > canvas.height) labelY = canvas.height - labelHeight; 
                                if (labelX < 0) labelX = 0; 
                                if (labelX + labelWidth > canvas.width) labelX = canvas.width - labelWidth; 
                                
                                ctx.fillStyle = '#00FF00';
                                ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
                                
                                ctx.fillStyle = '#000000';
                                ctx.fillText(labelText, labelX + padding, labelY + (padding / 2));

                                resultsHtml += `<strong>${det.label.toUpperCase()}</strong> (${confPercent}%): <br>${det.advice}<br><br>`;
                            });
                        } else {
                            resultsHtml = "No recognizable items detected.";
                        }
                        
                        resultDiv.innerHTML = resultsHtml;
                    } catch (e) {
                        console.error("Prediction error:", e);
                        resultDiv.innerHTML = "Error analyzing image.";
                    }
                };
                imgElement.src = URL.createObjectURL(file);
            }

            // --- CAMERA LOGIC ---
            async function startCamera() {
                const video = document.getElementById('video');
                try {
                    videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
                    video.srcObject = videoStream;
                    
                    document.getElementById('start-cam-btn').classList.add('hidden');
                    document.getElementById('stop-cam-btn').classList.remove('hidden');
                    document.getElementById('result').classList.remove('hidden');

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
                document.getElementById('result').classList.add('hidden');
            }

            async function captureAndPredict() {
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                
                canvas.classList.remove('hidden');
                canvas.style.position = 'absolute';
                canvas.style.left = video.offsetLeft + 'px';
                canvas.style.top = video.offsetTop + 'px';
                canvas.style.width = video.offsetWidth + 'px';
                canvas.style.height = video.offsetHeight + 'px';
                
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    canvas.toBlob(async (blob) => {
                        const formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');

                        try {
                            const response = await fetch('/predict', { method: 'POST', body: formData });
                            const data = await response.json();
                            
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            let resultsHtml = '';
                            
                            if (data.detections && data.detections.length > 0) {
                                data.detections.forEach(det => {
                                    const [x, y, w, h] = det.box;
                                    ctx.strokeStyle = '#00FF00'; 
                                    ctx.lineWidth = 4;
                                    ctx.strokeRect(x, y, w, h);
                                    
                                    ctx.font = '20px Arial';
                                    ctx.textBaseline = 'top';
                                    const confPercent = (det.confidence * 100).toFixed(0);
                                    const labelText = `${det.label} ${confPercent}%`;
                                    
                                    const padding = 5;
                                    const textWidth = ctx.measureText(labelText).width;
                                    const labelWidth = textWidth + (padding * 2);
                                    const labelHeight = 25; 
                                    
                                    let labelX = x;
                                    let labelY = y - labelHeight;
                                    
                                    if (labelY < 0) labelY = y;
                                    if (labelY + labelHeight > canvas.height) labelY = canvas.height - labelHeight;
                                    if (labelX < 0) labelX = 0;
                                    if (labelX + labelWidth > canvas.width) labelX = canvas.width - labelWidth;
                                    
                                    ctx.fillStyle = '#00FF00';
                                    ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
                                    
                                    ctx.fillStyle = '#000000'; 
                                    ctx.fillText(labelText, labelX + padding, labelY + (padding / 2));

                                    resultsHtml += `<strong>${det.label.toUpperCase()}</strong> (${confPercent}%): <br>${det.advice}<br><br>`;
                                });
                            } else {
                                resultsHtml = "Scanning for items...";
                            }
                            
                            document.getElementById('result').innerHTML = resultsHtml;
                            
                        } catch (e) {
                            console.error("Prediction error:", e);
                        }
                    }, 'image/jpeg', 0.8);
                }
            }
        </script>
      </body>
    </html>
    """