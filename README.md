# BlueBin Buddy

BlueBin Buddy is an ML-powered web app for Singapore’s blue recycling bins that classifies an uploaded item photo into 6 material types (cardboard, glass, metal, paper, plastic, trash) and returns contamination-aware guidance with confidence and a “Not sure” fallback.

## Features
- 6-class material classification (TrashNet baseline + Singapore-context photos)
- Safety-first abstain logic (low confidence → “Not sure”)
- FastAPI inference API with Swagger UI at `/docs`
- Reproducible training, evaluation (classification report + confusion matrix)

## Tech stack
- PyTorch, torchvision
- FastAPI + Uvicorn
- scikit-learn (metrics), Pillow (image IO)
