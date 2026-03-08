# app/main.py
import csv
import io
import logging
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from app.config import settings
from app.explainability import generate_gradcam_b64
from app.feedback import router as feedback_router
from src.predictor import TrashnetPredictor

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App & predictor ────────────────────────────────────────────────────────────
app = FastAPI(
    title="BlueBin Buddy",
    version="2.0.0",
    description="AI-powered recycling guide for Singapore's blue bin system.",
)
predictor = TrashnetPredictor(
    settings.MODEL_PATH,
    abstain_threshold=settings.CONFIDENCE_THRESHOLD,
)
app.include_router(feedback_router)

# ── In-memory prediction stats (resets on server restart) ─────────────────────
_prediction_requests = 0
_abstain_count = 0
_class_counts: dict = defaultdict(int)

MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024

logger.info(
    "BlueBin Buddy started — model=%s threshold=%.2f",
    settings.MODEL_PATH,
    settings.CONFIDENCE_THRESHOLD,
)


# ── Utility ────────────────────────────────────────────────────────────────────

async def _load_image(file: UploadFile) -> tuple[Image.Image, bytes]:
    """Validate and load an uploaded image. Returns (PIL Image, raw bytes)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file (JPEG, PNG, etc.).")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large. Maximum size is {settings.MAX_UPLOAD_MB} MB."
        )

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Could not decode image. Please upload a valid image file.")

    return img, contents


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.MODEL_NAME, "environment": settings.ENVIRONMENT}


@app.get("/metrics")
def metrics():
    """Model performance metadata."""
    return {
        "model": settings.MODEL_NAME,
        "mAP50": settings.MODEL_MAP50,
        "mAP50_95": settings.MODEL_MAP50_95,
        "size_mb": settings.MODEL_SIZE_MB,
        "classes": 9,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
    }


@app.get("/model/versions")
def model_versions():
    """List available model versions and which is active."""
    return [
        {
            "name": "YOLOv26n",
            "mAP50": 0.888,
            "mAP50_95": 0.712,
            "size_mb": 5.4,
            "active": True,
            "description": "Fine-tuned on 63 K recycling images. Current production model.",
        },
        {
            "name": "YOLOv8n",
            "mAP50": 0.869,
            "mAP50_95": 0.698,
            "size_mb": 6.5,
            "active": False,
            "description": "Baseline model. Superseded by YOLOv26n.",
        },
    ]


@app.get("/analytics")
def analytics():
    """Prediction statistics since server start + feedback summary."""
    total_detections = sum(_class_counts.values())
    contam_classes = {"styrofoam", "e-waste", "clothing"}
    contam_detections = sum(_class_counts.get(c, 0) for c in contam_classes)

    feedback_count = 0
    feedback_path = Path("reports/feedback.csv")
    if feedback_path.exists():
        with feedback_path.open(encoding="utf-8") as f:
            feedback_count = max(0, sum(1 for _ in f) - 1)  # subtract header

    return {
        "model": settings.MODEL_NAME,
        "mAP50": settings.MODEL_MAP50,
        "prediction_requests": _prediction_requests,
        "total_detections": total_detections,
        "abstain_count": _abstain_count,
        "class_distribution": dict(_class_counts),
        "contamination_rate": round(contam_detections / max(total_detections, 1), 3),
        "abstain_rate": round(_abstain_count / max(_prediction_requests, 1), 3),
        "feedback_corrections": feedback_count,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _prediction_requests, _abstain_count

    img, _ = await _load_image(file)

    t0 = time.perf_counter()
    detections = predictor.predict_pil(img)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    _prediction_requests += 1
    if not detections:
        _abstain_count += 1
    for d in detections:
        _class_counts[d.label] += 1

    logger.info(
        "predict  file=%s  detections=%d  classes=%s  latency_ms=%.1f",
        file.filename,
        len(detections),
        [d.label for d in detections],
        elapsed_ms,
    )

    return {
        "filename": file.filename,
        "latency_ms": elapsed_ms,
        "detections": [
            {
                "label": d.label,
                "confidence": round(d.confidence, 4),
                "box": d.box,
                "advice": d.advice,
            }
            for d in detections
        ],
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """Generate an EigenCAM attention heatmap for the uploaded image."""
    img, _ = await _load_image(file)

    try:
        heatmap_b64 = generate_gradcam_b64(predictor.model, img)
    except RuntimeError as exc:
        logger.warning("GradCAM failed: %s", exc)
        raise HTTPException(422, str(exc))

    return {"heatmap_b64": heatmap_b64}


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return _HTML


_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>BlueBin Buddy</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --primary:       #0056b3;
      --primary-hover: #004494;
      --success:       #2e7d32;
      --success-light: #e8f5e9;
      --warn:          #f57f17;
      --warn-light:    #fff8e1;
      --danger:        #c62828;
      --danger-light:  #ffebee;
      --bg:            #f0f4f3;
      --card:          #ffffff;
      --text:          #212121;
      --muted:         #616161;
      --border:        #e0e0e0;
      --radius:        12px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}

    /* ── Header ────────────────────────────────────────────────────────────── */
    .header {{
      background: var(--primary);
      color: white;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 2px 8px rgba(0,0,0,.15);
    }}
    .header h1 {{ font-size: 1.3rem; font-weight: 700; }}
    .model-badge {{
      background: rgba(255,255,255,.18);
      border: 1px solid rgba(255,255,255,.3);
      border-radius: 20px;
      padding: 4px 12px;
      font-size: .75rem;
      font-weight: 600;
      letter-spacing: .3px;
    }}

    /* ── Layout ────────────────────────────────────────────────────────────── */
    .container {{
      max-width: 520px;
      margin: 0 auto;
      padding: 16px;
    }}
    .card {{
      background: var(--card);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
      margin-bottom: 16px;
    }}

    /* ── Tabs ──────────────────────────────────────────────────────────────── */
    .tab-bar {{
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .tab-btn {{
      flex: 1;
      padding: 12px 8px;
      border: 2px solid var(--border);
      border-radius: 8px;
      font-size: .9rem;
      font-weight: 600;
      cursor: pointer;
      background: var(--card);
      color: var(--muted);
      transition: all .2s;
    }}
    .tab-btn.active {{
      background: var(--primary);
      color: white;
      border-color: var(--primary);
    }}
    .tab-btn:active {{ transform: scale(.97); }}

    /* ── Buttons ───────────────────────────────────────────────────────────── */
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 12px 18px;
      border: none;
      border-radius: 8px;
      font-size: .95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all .2s;
    }}
    .btn:active {{ transform: scale(.97); }}
    .btn-primary   {{ background: var(--success); color: white; width: 100%; justify-content: center; margin-top: 12px; }}
    .btn-secondary {{ background: var(--primary); color: white; width: 100%; justify-content: center; margin-top: 8px; }}
    .btn-stop      {{ background: var(--danger);  color: white; width: 100%; justify-content: center; margin-top: 8px; }}
    .btn-sm        {{ padding: 6px 12px; font-size: .82rem; border-radius: 6px; }}
    .btn-correct   {{ background: var(--success-light); color: var(--success); border: 1px solid var(--success); }}
    .btn-wrong     {{ background: var(--danger-light);  color: var(--danger);  border: 1px solid var(--danger);  }}
    .btn-submit    {{ background: var(--primary); color: white; margin-top: 6px; }}

    /* ── File input ────────────────────────────────────────────────────────── */
    input[type="file"] {{
      width: 100%;
      padding: 12px;
      border: 2px dashed var(--border);
      border-radius: 8px;
      background: #fafafa;
      color: var(--muted);
      font-family: inherit;
      cursor: pointer;
    }}

    /* ── Image / video containers ──────────────────────────────────────────── */
    #upload-container, #video-container {{
      position: relative;
      width: 100%;
      border-radius: 8px;
      overflow: hidden;
      background: #111;
      margin-top: 14px;
    }}
    #upload-img, video {{ width: 100%; display: block; }}
    canvas {{
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none;
    }}

    /* ── Heatmap ───────────────────────────────────────────────────────────── */
    #heatmap-section {{
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .heatmap-header {{
      background: #1a237e;
      color: white;
      padding: 8px 12px;
      font-size: .8rem;
      font-weight: 600;
    }}
    #heatmap-img {{ width: 100%; display: block; }}

    /* ── Detection cards ───────────────────────────────────────────────────── */
    .det-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 10px;
      background: #fafafa;
    }}
    .det-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}
    .label-badge {{
      padding: 4px 10px;
      border-radius: 20px;
      font-size: .8rem;
      font-weight: 700;
      letter-spacing: .4px;
      text-transform: uppercase;
    }}
    .conf-high  {{ background: #c8e6c9; color: #1b5e20; }}
    .conf-med   {{ background: #fff9c4; color: #f57f17; }}
    .conf-low   {{ background: #ffccbc; color: #bf360c; }}
    .advice-text {{ font-size: .88rem; color: var(--muted); line-height: 1.5; }}
    .feedback-row {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }}
    .correct-options {{
      margin-top: 10px;
      padding: 10px;
      background: var(--bg);
      border-radius: 8px;
    }}
    .correct-options select {{
      width: 100%;
      padding: 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: inherit;
      font-size: .88rem;
      background: white;
    }}
    .feedback-thanks {{
      font-size: .82rem;
      color: var(--success);
      font-weight: 600;
      margin-top: 6px;
    }}
    .no-detection {{
      text-align: center;
      color: var(--muted);
      padding: 20px;
      font-size: .95rem;
    }}

    /* ── Stats section ─────────────────────────────────────────────────────── */
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stat-card {{
      background: var(--bg);
      border-radius: 10px;
      padding: 14px;
      text-align: center;
    }}
    .stat-num  {{ font-size: 1.6rem; font-weight: 700; color: var(--primary); }}
    .stat-lbl  {{ font-size: .75rem; color: var(--muted); margin-top: 2px; }}
    .chart-wrap {{ max-height: 260px; }}

    /* ── Loading overlay ───────────────────────────────────────────────────── */
    #loading-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,.45);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 999;
      color: white;
      font-weight: 600;
      gap: 14px;
    }}
    .spinner {{
      width: 44px; height: 44px;
      border: 4px solid rgba(255,255,255,.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin .75s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    /* ── Toast ─────────────────────────────────────────────────────────────── */
    #toast-container {{
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 1000;
      width: 90%;
      max-width: 380px;
    }}
    .toast {{
      padding: 12px 18px;
      border-radius: 10px;
      font-size: .9rem;
      font-weight: 600;
      box-shadow: 0 4px 14px rgba(0,0,0,.2);
      animation: fadeIn .25s ease;
      text-align: center;
    }}
    .toast.success {{ background: #2e7d32; color: white; }}
    .toast.error   {{ background: #c62828; color: white; }}
    .toast.info    {{ background: #0056b3; color: white; }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}

    /* ── Misc ──────────────────────────────────────────────────────────────── */
    .hidden {{ display: none !important; }}
    .links {{
      text-align: center;
      margin: 6px 0 20px;
      font-size: .82rem;
      color: var(--muted);
    }}
    .links a {{ color: var(--primary); text-decoration: none; }}
    .section-title {{ font-size: 1rem; font-weight: 700; margin-bottom: 12px; }}
  </style>
</head>
<body>

<header class="header">
  <h1>♻ BlueBin Buddy</h1>
  <div class="model-badge">{settings.MODEL_NAME} · mAP50 {settings.MODEL_MAP50*100:.1f}%</div>
</header>

<div class="container">

  <!-- ── Tabs ──────────────────────────────────────────────────────────────── -->
  <div class="tab-bar">
    <button id="tab-upload" class="tab-btn active" onclick="setMode('upload')">📷 Upload</button>
    <button id="tab-camera" class="tab-btn"        onclick="setMode('camera')">📸 Camera</button>
    <button id="tab-stats"  class="tab-btn"        onclick="setMode('stats')">📊 Stats</button>
  </div>

  <!-- ── Upload section ───────────────────────────────────────────────────── -->
  <div id="upload-section">
    <div class="card">
      <input id="upload-file" type="file" accept="image/*" onchange="previewAndPredict(event)">

      <div id="upload-container" class="hidden">
        <img id="upload-img" alt="Uploaded image">
        <canvas id="upload-canvas"></canvas>
      </div>

      <button id="explain-btn" class="btn btn-secondary hidden" onclick="runExplain()">
        🔬 Explain (GradCAM)
      </button>

      <div id="heatmap-section" class="hidden">
        <div class="heatmap-header">🔥 Attention Heatmap — areas the model focused on</div>
        <img id="heatmap-img" alt="GradCAM heatmap">
      </div>
    </div>

    <div id="upload-results" class="hidden"></div>
  </div>

  <!-- ── Camera section ───────────────────────────────────────────────────── -->
  <div id="camera-section" class="hidden">
    <div class="card">
      <div id="video-container">
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas" class="hidden"></canvas>
      </div>
      <button id="start-cam-btn" class="btn btn-primary"  onclick="startCamera()">▶ Start Camera</button>
      <button id="stop-cam-btn"  class="btn btn-stop hidden" onclick="stopCamera()">■ Stop Camera</button>
    </div>
    <div id="camera-result" class="card hidden"></div>
  </div>

  <!-- ── Stats section ────────────────────────────────────────────────────── -->
  <div id="stats-section" class="hidden">
    <div class="card">
      <div class="section-title">Model Info</div>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-num">{settings.MODEL_MAP50*100:.1f}%</div>
          <div class="stat-lbl">mAP50</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">{settings.MODEL_MAP50_95*100:.1f}%</div>
          <div class="stat-lbl">mAP50-95</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">9</div>
          <div class="stat-lbl">Classes</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">{settings.MODEL_SIZE_MB} MB</div>
          <div class="stat-lbl">Model Size</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="section-title">Session Activity</div>
      <div id="session-stats" class="stat-grid">
        <div class="stat-card">
          <div class="stat-num" id="stat-requests">—</div>
          <div class="stat-lbl">Predictions</div>
        </div>
        <div class="stat-card">
          <div class="stat-num" id="stat-detections">—</div>
          <div class="stat-lbl">Detections</div>
        </div>
        <div class="stat-card">
          <div class="stat-num" id="stat-contam">—</div>
          <div class="stat-lbl">Contamination Rate</div>
        </div>
        <div class="stat-card">
          <div class="stat-num" id="stat-feedback">—</div>
          <div class="stat-lbl">Feedback Logged</div>
        </div>
      </div>

      <div class="chart-wrap">
        <canvas id="class-chart"></canvas>
      </div>
      <p id="no-data-msg" style="text-align:center; color: var(--muted); font-size:.88rem; margin-top:10px;">
        No predictions yet — upload an image to see stats.
      </p>
    </div>
  </div>

  <div class="links">
    <a href="/docs">API Docs</a> &nbsp;·&nbsp;
    <a href="/metrics">Metrics JSON</a> &nbsp;·&nbsp;
    <a href="/model/versions">Model Versions</a>
  </div>
</div>

<!-- ── Loading overlay ──────────────────────────────────────────────────────── -->
<div id="loading-overlay" class="hidden">
  <div class="spinner"></div>
  <span>Analyzing…</span>
</div>

<!-- ── Toast container ──────────────────────────────────────────────────────── -->
<div id="toast-container"></div>

<script>
/* ── Constants ──────────────────────────────────────────────────────────────── */
const CLASSES = ['aluminium','cardboard','clothing','e-waste','glass','metal','paper','plastic','styrofoam'];
const CHART_COLORS = [
  '#42a5f5','#66bb6a','#ef5350','#ffa726','#26c6da','#ab47bc','#8d6e63','#78909c','#ffee58'
];

/* ── State ──────────────────────────────────────────────────────────────────── */
let videoStream  = null;
let intervalId   = null;
let currentFile  = null;
let currentDetections = [];
let chartInstance = null;
let currentFilename = '';

/* ── Utility ─────────────────────────────────────────────────────────────────── */
function showLoading(show) {{
  document.getElementById('loading-overlay').classList.toggle('hidden', !show);
}}

function showToast(message, type = 'info') {{
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${{type}}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}}

function getConfClass(conf) {{
  if (conf >= 0.80) return 'conf-high';
  if (conf >= 0.60) return 'conf-med';
  return 'conf-low';
}}

function getBoxColor(label) {{
  const bad = new Set(['styrofoam','e-waste','clothing']);
  return bad.has(label) ? '#ef5350' : '#66bb6a';
}}

/* ── Tab switching ───────────────────────────────────────────────────────────── */
function setMode(mode) {{
  ['upload','camera','stats'].forEach(m => {{
    document.getElementById(`${{m}}-section`).classList.toggle('hidden', m !== mode);
    document.getElementById(`tab-${{m}}`).classList.toggle('active', m === mode);
  }});
  if (mode !== 'camera') stopCamera();
  if (mode === 'stats') loadStats();
}}

/* ── Upload & predict ────────────────────────────────────────────────────────── */
async function previewAndPredict(event) {{
  const file = event.target.files[0];
  if (!file) return;

  currentFile = file;
  currentFilename = file.name;
  currentDetections = [];

  const imgEl      = document.getElementById('upload-img');
  const container  = document.getElementById('upload-container');
  const canvas     = document.getElementById('upload-canvas');
  const resultsDiv = document.getElementById('upload-results');
  const explainBtn = document.getElementById('explain-btn');
  const heatmap    = document.getElementById('heatmap-section');

  // Reset previous state
  heatmap.classList.add('hidden');
  resultsDiv.classList.add('hidden');
  explainBtn.classList.add('hidden');
  container.classList.remove('hidden');

  // Preview image
  imgEl.onload = async () => {{
    canvas.width  = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    // Run prediction
    showLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {{
      const resp = await fetch('/predict', {{ method: 'POST', body: formData }});
      if (!resp.ok) {{
        const err = await resp.json();
        throw new Error(err.detail || 'Prediction failed');
      }}
      const data = await resp.json();
      currentDetections = data.detections || [];

      drawDetections(canvas, currentDetections);
      renderResults(currentDetections, file.name);

      explainBtn.classList.remove('hidden');
      showToast(`Detected ${{currentDetections.length}} item(s) in ${{data.latency_ms}} ms`, 'success');
    }} catch (e) {{
      showToast(`Error: ${{e.message}}`, 'error');
      resultsDiv.innerHTML = `<div class="no-detection">${{e.message}}</div>`;
      resultsDiv.classList.remove('hidden');
    }} finally {{
      showLoading(false);
    }}
  }};
  imgEl.src = URL.createObjectURL(file);
}}

function drawDetections(canvas, detections) {{
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  detections.forEach(det => {{
    const [x, y, w, h] = det.box;
    const color   = getBoxColor(det.label);
    const scale   = canvas.width / 640;
    const lw      = Math.max(3, 2.5 * scale);
    const fs      = Math.max(16, 14 * scale);
    const pad     = 8 * scale;

    ctx.strokeStyle = color;
    ctx.lineWidth   = lw;
    ctx.strokeRect(x, y, w, h);

    const confPct = (det.confidence * 100).toFixed(0);
    const label   = `${{det.label}} ${{confPct}}%`;

    ctx.font         = `bold ${{fs}}px Inter, Arial`;
    ctx.textBaseline = 'top';
    const tw  = ctx.measureText(label).width;
    const lbW = tw + pad * 2;
    const lbH = fs + pad;

    let lx = Math.max(0, Math.min(x, canvas.width  - lbW));
    let ly = y - lbH < 0 ? y : y - lbH;

    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, lbW, lbH);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, lx + pad, ly + pad / 2);
  }});
}}

function renderResults(detections, filename) {{
  const div = document.getElementById('upload-results');

  if (!detections || detections.length === 0) {{
    div.innerHTML = `<div class="card"><div class="no-detection">
      🤔 No recognisable items detected.<br>
      <small>Try a clearer photo or move the item closer.</small>
    </div></div>`;
    div.classList.remove('hidden');
    return;
  }}

  let html = '<div class="card">';
  detections.forEach((det, idx) => {{
    const confPct   = (det.confidence * 100).toFixed(0);
    const confClass = getConfClass(det.confidence);
    const isContam  = ['styrofoam','e-waste','clothing'].includes(det.label);

    html += `
    <div class="det-card">
      <div class="det-header">
        <span class="label-badge ${{confClass}}">${{det.label}}</span>
        <span class="label-badge ${{confClass}}">${{confPct}}% confidence</span>
      </div>
      <div class="advice-text">
        ${{isContam ? '🚫' : '✅'}} ${{det.advice}}
      </div>
      <div class="feedback-row">
        <button class="btn btn-sm btn-correct" onclick="markCorrect(${{idx}})">✓ Correct</button>
        <button class="btn btn-sm btn-wrong"   onclick="showCorrection(${{idx}})">✗ Wrong</button>
      </div>
      <div id="correction-${{idx}}" class="correct-options hidden">
        <select id="correct-label-${{idx}}">
          ${{CLASSES.map(c => `<option value="${{c}}" ${{c===det.label?'selected':''}}>${{c}}</option>`).join('')}}
        </select>
        <button class="btn btn-sm btn-submit" onclick="submitCorrection(${{idx}})">Submit Correction</button>
      </div>
      <div id="thanks-${{idx}}" class="feedback-thanks hidden">Thanks! Your feedback helps improve the model. 🙏</div>
    </div>`;
  }});
  html += '</div>';

  div.innerHTML = html;
  div.classList.remove('hidden');
}}

/* ── Feedback ─────────────────────────────────────────────────────────────────── */
function markCorrect(idx) {{
  const det = currentDetections[idx];
  postFeedback(det.label, det.label, det.confidence, false);
  document.getElementById(`thanks-${{idx}}`).classList.remove('hidden');
  document.querySelector(`#upload-results .det-card:nth-child(${{idx+1}}) .feedback-row`).classList.add('hidden');
}}

function showCorrection(idx) {{
  document.getElementById(`correction-${{idx}}`).classList.toggle('hidden');
}}

async function submitCorrection(idx) {{
  const det = currentDetections[idx];
  const correctLabel = document.getElementById(`correct-label-${{idx}}`).value;
  postFeedback(det.label, correctLabel, det.confidence, false);
  document.getElementById(`correction-${{idx}}`).classList.add('hidden');
  document.getElementById(`thanks-${{idx}}`).classList.remove('hidden');
  showToast('Correction submitted — thank you!', 'success');
}}

async function postFeedback(predicted, correct, confidence, abstained) {{
  try {{
    await fetch('/feedback', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        filename: currentFilename,
        predicted_label: predicted,
        correct_label: correct,
        confidence: confidence,
        abstained: abstained,
      }}),
    }});
  }} catch (e) {{
    console.warn('Feedback POST failed:', e);
  }}
}}

/* ── GradCAM / Explain ───────────────────────────────────────────────────────── */
async function runExplain() {{
  if (!currentFile) return;
  showLoading(true);

  const formData = new FormData();
  formData.append('file', currentFile);

  try {{
    const resp = await fetch('/explain', {{ method: 'POST', body: formData }});
    const data = await resp.json();

    if (!resp.ok) throw new Error(data.detail || 'Heatmap generation failed');

    const heatmapImg = document.getElementById('heatmap-img');
    heatmapImg.src   = 'data:image/png;base64,' + data.heatmap_b64;

    document.getElementById('heatmap-section').classList.remove('hidden');
    showToast('Attention heatmap ready!', 'success');
  }} catch (e) {{
    showToast(`GradCAM: ${{e.message}}`, 'error');
  }} finally {{
    showLoading(false);
  }}
}}

/* ── Camera ──────────────────────────────────────────────────────────────────── */
async function startCamera() {{
  try {{
    videoStream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'environment' }} }});
    document.getElementById('video').srcObject = videoStream;
    document.getElementById('start-cam-btn').classList.add('hidden');
    document.getElementById('stop-cam-btn').classList.remove('hidden');
    document.getElementById('camera-result').classList.remove('hidden');
    intervalId = setInterval(captureAndPredict, 1500);
    showToast('Camera started', 'info');
  }} catch (err) {{
    showToast(`Camera error: ${{err.message}}`, 'error');
  }}
}}

function stopCamera() {{
  if (videoStream) videoStream.getTracks().forEach(t => t.stop());
  if (intervalId) clearInterval(intervalId);
  videoStream = intervalId = null;
  document.getElementById('start-cam-btn').classList.remove('hidden');
  document.getElementById('stop-cam-btn').classList.add('hidden');
  document.getElementById('camera-result').classList.add('hidden');
  const canvas = document.getElementById('canvas');
  canvas.classList.add('hidden');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}}

async function captureAndPredict() {{
  const video  = document.getElementById('video');
  const canvas = document.getElementById('canvas');

  if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.classList.remove('hidden');

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async blob => {{
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');

    try {{
      const resp = await fetch('/predict', {{ method: 'POST', body: formData }});
      const data = await resp.json();

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      let html = '';
      if (data.detections && data.detections.length > 0) {{
        drawDetections(canvas, data.detections);
        html = data.detections.map(det => {{
          const pct = (det.confidence * 100).toFixed(0);
          const icon = ['styrofoam','e-waste','clothing'].includes(det.label) ? '🚫' : '✅';
          return `<strong>${{icon}} ${{det.label.toUpperCase()}} (${{pct}}%)</strong><br>
                  <span style="font-size:.85rem;color:var(--muted)">${{det.advice}}</span>`;
        }}).join('<hr style="margin:8px 0; border:none; border-top:1px solid var(--border)">');
      }} else {{
        html = '<span style="color:var(--muted)">Scanning… no items detected.</span>';
      }}
      document.getElementById('camera-result').innerHTML = html;
    }} catch (e) {{
      console.error('Camera predict error:', e);
    }}
  }}, 'image/jpeg', 0.8);
}}

/* ── Stats tab ───────────────────────────────────────────────────────────────── */
async function loadStats() {{
  try {{
    const resp = await fetch('/analytics');
    const data = await resp.json();

    document.getElementById('stat-requests').textContent   = data.prediction_requests;
    document.getElementById('stat-detections').textContent = data.total_detections;
    document.getElementById('stat-contam').textContent     = (data.contamination_rate * 100).toFixed(0) + '%';
    document.getElementById('stat-feedback').textContent   = data.feedback_corrections;

    const dist = data.class_distribution || {{}};
    const labels = Object.keys(dist);
    const values = Object.values(dist);

    const noDataMsg = document.getElementById('no-data-msg');

    if (labels.length === 0) {{
      noDataMsg.classList.remove('hidden');
      return;
    }}
    noDataMsg.classList.add('hidden');

    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(document.getElementById('class-chart'), {{
      type: 'doughnut',
      data: {{
        labels,
        datasets: [{{
          data: values,
          backgroundColor: CHART_COLORS.slice(0, labels.length),
          borderWidth: 2,
          borderColor: '#fff',
        }}],
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ font: {{ size: 11 }} }} }},
          title: {{ display: true, text: 'Detections by Class', font: {{ size: 13 }} }},
        }},
      }},
    }});
  }} catch (e) {{
    console.error('Failed to load analytics:', e);
  }}
}}

/* ── Keyboard shortcuts ──────────────────────────────────────────────────────── */
document.addEventListener('keydown', e => {{
  if (e.key === 'Enter' && !document.getElementById('upload-section').classList.contains('hidden')) {{
    document.getElementById('upload-file').click();
  }}
}});
</script>
</body>
</html>"""
