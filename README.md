---
title: BlueBin Buddy
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# ♻ BlueBin Buddy

An ML-powered recycling assistant for Singapore's blue bin system. Point your camera at an item, our BlueBin Buddy identifies what it is, whether it belongs in the blue bin, and gives you NEA-compliant disposal guidance in real time.

Live demo: [huggingface.co/spaces/qusydani/BlueBinBuddy](https://huggingface.co/spaces/qusydani/BlueBinBuddy)

[![Video Demo](https://img.youtube.com/vi/tl826S2Zgi4/maxresdefault.jpg)](https://www.youtube.com/watch?v=tl826S2Zgi4)
---

## Features

- **Live camera detection** — real-time bounding box overlay via the back camera on mobile
- **Image upload** — drag-and-drop or file picker with instant prediction
- **9-class detection** — aluminium, cardboard, clothing, e-waste, glass, metal, paper, plastic, styrofoam
- **Contamination awareness** — flags items that must NOT go in the blue bin (styrofoam, e-waste, clothing)
- **Safety-first abstain logic** — withholds predictions below 0.40 confidence rather than guessing
- **D-RISE explainability** — saliency heatmap showing which regions of the image drove the prediction
- **Feedback loop** — users can correct wrong predictions with corrections logged to CSV for active learning
- **Session stats** — live prediction counts, contamination rate, and per-class distribution chart

---

## Model Performance

| Model              | mAP50     | mAP50-95 | Size   |
| ------------------ | --------- | -------- | ------ |
| YOLOv26n (active)  | **88.8%** | 71.2%    | 5.4 MB |
| YOLOv8n (baseline) | 86.9%     | —        | —      |

Both models fine-tuned from COCO pre-trained weights on the RecycleRightSG dataset which was manually compiled using Roboflow.

---

## Dataset

- **Source:** Roboflow (labelled) + TrashNet (augmented)
- **Size:** 63,416 train / 6,128 val / 1,514 test images
- **Classes (9):** aluminium, cardboard, clothing, e-waste, glass, metal, paper, plastic, styrofoam
- **Config:** `data/master_dataset/data.yaml`

---

## Tech Stack

| Layer               | Technology                           |
| ------------------- | ------------------------------------ |
| ML model            | YOLOv26n (Ultralytics), fine-tuned   |
| Inference           | PyTorch (CPU), Pillow                |
| Explainability      | D-RISE (perturbation-based saliency) |
| Backend             | FastAPI + Uvicorn                    |
| Frontend            | JS, Canvas API, Chart.js             |
| Experiment tracking | MLflow                               |
| Config              | Pydantic Settings                    |
| Feedback logging    | CSV (reports/feedback.csv)           |
| Deployment          | Docker, Hugging Face Spaces          |

---

## API Endpoints

| Method | Endpoint          | Description                                  |
| ------ | ----------------- | -------------------------------------------- |
| `GET`  | `/`               | Web UI                                       |
| `GET`  | `/health`         | Health check                                 |
| `GET`  | `/metrics`        | Model performance metadata                   |
| `GET`  | `/model/versions` | Available model versions                     |
| `GET`  | `/analytics`      | Session prediction statistics                |
| `POST` | `/predict`        | Image → detections JSON                      |
| `POST` | `/explain`        | Image → D-RISE saliency heatmap (base64 PNG) |
| `POST` | `/feedback`       | Log user correction for active learning      |

---

## Local Development

### Prerequisites

- Python 3.11+
- GPU optional (CPU inference supported)

### Setup

```bash
# Clone the repo
git clone https://github.com/qusydani/RecycleRightSG.git
cd RecycleRightSG

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install CPU-only PyTorch first (avoids downloading the 2.5 GB GPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

### Environment configuration

Create a `.env` file in the project root to override defaults:

```env
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.40
MAX_UPLOAD_MB=10
ENVIRONMENT=development
```

All settings are defined in `app/config.py` and documented with their defaults.

---

## Training

```bash
python src/train.py \
  --data data/master_dataset/data.yaml \
  --model yolo26n.pt \
  --epochs 50 \
  --imgsz 640
```

Training uses early stopping (`patience=5`) and saves the best checkpoint automatically to `models/best.pt`. All runs are tracked with MLflow — view the experiment dashboard with:

```bash
mlflow ui
```

---

## Testing

```bash
# Run full test suite (21 tests)
pytest tests/ -v

# Run a specific file
pytest tests/test_api.py -v
```

The test suite mocks the YOLO model at import time so no model file is required to run tests. All 21 unit and integration tests pass on CPU in CI.

---

## Project Structure

```
RecycleRightSG/
├── app/
│   ├── main.py              # FastAPI server + full HTML/CSS/JS UI
│   ├── config.py            # Pydantic settings (env-configurable)
│   ├── explainability.py    # D-RISE saliency heatmap generation
│   └── feedback.py          # Feedback router + CSV logging
├── src/
│   ├── predictor.py         # YOLO inference wrapper + NEA advice mapping
│   └── train.py             # Training script with MLflow tracking
├── data/
│   └── master_dataset/
│       └── data.yaml        # YOLO dataset config
├── models/
│   └── best.pt              # Active model weights (Git LFS)
├── tests/
│   ├── conftest.py          # Pytest fixtures + YOLO mock
│   ├── test_predictor.py    # Unit tests (NEA mapping, filters)
│   └── test_api.py          # Integration tests (all endpoints)
├── reports/
│   └── feedback.csv         # User correction log
├── notebooks/
│   ├── model_comparison.ipynb
│   └── error_analysis.ipynb
├── .github/workflows/ci.yml # GitHub Actions (lint → test → docker build)
├── Dockerfile               # Multi-stage, port 7860 (HF Spaces compatible)
└── requirements.txt
```

---

## Deployment

The app is deployed to Hugging Face Spaces using the Docker SDK. On every push to the `huggingface` remote, HF Spaces rebuilds the container automatically.

```bash
# Push to GitHub (triggers CI)
git push origin main

# Push to Hugging Face Spaces (triggers redeploy)
git push huggingface main
```

Model weights (`.pt` files) are tracked with Git LFS via `.gitattributes`.
