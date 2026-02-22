# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import io

from src.predictor import TrashnetPredictor
from app.feedback import router as feedback_router

app = FastAPI()
predictor = TrashnetPredictor("models/best.pt", abstain_threshold=0.55)
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

    # Get list of detections from YOLO
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
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BlueBin Buddy</title>
        <style>
            body { font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            #video-container { margin-top: 20px; position: relative; }
            video { width: 100%; max-width: 400px; border-radius: 8px; background: #000; display: block; }
            
            /* New styles to overlay the canvas on uploaded photos */
            #upload-container { position: relative; margin-top: 20px; width: 100%; max-width: 400px; }
            #upload-img { width: 100%; display: block; border-radius: 8px; }
            #upload-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
            
            #result { margin-top: 15px; padding: 10px; border-radius: 5px; background: #f0f0f0; min-height: 50px;}
            .hidden { display: none !important; }
            button { padding: 10px; margin-top: 10px; cursor: pointer; }
            input[type="file"] { margin-top: 20px; margin-bottom: 10px; display: block; }
        </style>
      </head>
      <body>
        <h3>BlueBin Buddy</h3>
        
        <div>
            <button onclick="setMode('upload')">Upload Photo</button>
            <button onclick="setMode('camera')">Live Camera</button>
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
            <button id="start-cam-btn" onclick="startCamera()">Start Camera</button>
            <button id="stop-cam-btn" class="hidden" onclick="stopCamera()">Stop Camera</button>
        </div>

        <div id="result" class="hidden"></div>

        <p><a href="/docs">View API Docs</a></p>

        <script>
            let videoStream = null;
            let intervalId = null;

            function setMode(mode) {
                document.getElementById('upload-section').classList.toggle('hidden', mode !== 'upload');
                document.getElementById('camera-section').classList.toggle('hidden', mode !== 'camera');
                document.getElementById('result').classList.add('hidden'); // Clear results on mode switch
                if (mode === 'upload') {
                    stopCamera();
                }
            }

            // UPLOAD LOGIC
            async function previewAndPredict() {
                const fileInput = document.getElementById('upload-file');
                if (!fileInput.files || fileInput.files.length === 0) return;
                
                const file = fileInput.files[0];
                const imgElement = document.getElementById('upload-img');
                const container = document.getElementById('upload-container');
                const canvas = document.getElementById('upload-canvas');
                const resultDiv = document.getElementById('result');
                
                // Show container and loading state
                container.classList.remove('hidden');
                resultDiv.classList.remove('hidden');
                resultDiv.innerHTML = "<strong>Analyzing uploaded image...</strong>";
                
                // When image loads on screen, send it to the API and draw the boxes
                imgElement.onload = async () => {
                    // Match the canvas pixel resolution to the actual photo resolution
                    canvas.width = imgElement.naturalWidth;
                    canvas.height = imgElement.naturalHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        
                        let resultsHtml = '';
                        
                        if (data.detections && data.detections.length > 0) {
                            data.detections.forEach(det => {
                                const [x, y, w, h] = det.box;
                                
                                // Dynamically scale box thickness and font size based on uploaded photo resolution
                                const scale = canvas.width / 400; 
                                const fontSize = Math.max(20, 15 * scale);
                                const padding = 10 * scale;
                                
                                // Draw Neon Green Box
                                ctx.strokeStyle = '#00FF00';
                                ctx.lineWidth = Math.max(4, 3 * scale);
                                ctx.strokeRect(x, y, w, h);
                                
                                const confPercent = (det.confidence * 100).toFixed(0);
                                const labelText = `${det.label} ${confPercent}%`;
                                
                                // Draw Label Background
                                ctx.font = `${fontSize}px Arial`;
                                ctx.fillStyle = '#00FF00';
                                const textWidth = ctx.measureText(labelText).width;
                                ctx.fillRect(x, y - fontSize - padding, textWidth + padding * 2, fontSize + padding);
                                
                                // Draw Label Text
                                ctx.fillStyle = '#000000';
                                ctx.fillText(labelText, x + padding, y - (padding/2));

                                // Append HTML result text below the photo
                                resultsHtml += `<strong>${det.label}</strong> (${confPercent}%): ${det.advice}<br><br>`;
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
                
                // Trigger the image load in the browser
                imgElement.src = URL.createObjectURL(file);
            }


            // CAMERA LOGIC
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
                            const response = await fetch('/predict', {
                                method: 'POST',
                                body: formData
                            });
                            const data = await response.json();
                            
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            let resultsHtml = '';
                            
                            if (data.detections && data.detections.length > 0) {
                                data.detections.forEach(det => {
                                    const [x, y, w, h] = det.box;
                                    ctx.strokeStyle = '#00FF00';
                                    ctx.lineWidth = 4;
                                    ctx.strokeRect(x, y, w, h);
                                    
                                    ctx.fillStyle = '#00FF00';
                                    ctx.fillRect(x, y - 25, ctx.measureText(det.label).width + 60, 25);
                                    
                                    ctx.fillStyle = '#000000';
                                    ctx.font = '20px Arial';
                                    const confPercent = (det.confidence * 100).toFixed(0);
                                    ctx.fillText(`${det.label} ${confPercent}%`, x + 5, y - 5);

                                    resultsHtml += `<strong>${det.label}</strong> (${confPercent}%): ${det.advice}<br><br>`;
                                });
                            } else {
                                resultsHtml = "No recognizable items detected.";
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