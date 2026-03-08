# src/train.py
import argparse
import shutil
from pathlib import Path

import mlflow
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv26n on the recycling dataset")
    ap.add_argument("--data",   default="data/master_dataset/data.yaml", help="Path to data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--model",  default="yolo26n.pt", help="Base YOLO weights to fine-tune")
    args = ap.parse_args()

    mlflow.set_experiment("recyclerightsg")

    with mlflow.start_run():
        # ── Log hyperparameters ───────────────────────────────────────────────
        mlflow.log_params({
            "base_model":  args.model,
            "epochs":      args.epochs,
            "imgsz":       args.imgsz,
            "batch":       16,
            "patience":    5,
            "dataset":     args.data,
        })

        # ── Train ─────────────────────────────────────────────────────────────
        print(f"Starting training: model={args.model} epochs={args.epochs} imgsz={args.imgsz}")
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            patience=5,
            cache="disk",
            batch=16,
            plots=True,
            device=0,
        )

        # ── Log metrics ───────────────────────────────────────────────────────
        mlflow.log_metrics({
            "mAP50":     float(results.box.map50),
            "mAP50_95":  float(results.box.map),
            "precision": float(results.box.mp),
            "recall":    float(results.box.mr),
        })

        # ── Log artifacts ─────────────────────────────────────────────────────
        best_weights = Path("runs/detect/train/weights/best.pt")
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            # Also copy to models/ for the inference server
            shutil.copy(best_weights, "models/best.pt")
            print(f"Best model copied to models/best.pt  (mAP50={results.box.map50:.3f})")

        artifacts_dir = Path("runs/detect/train")
        for plot in artifacts_dir.glob("*.png"):
            mlflow.log_artifact(str(plot), artifact_path="plots")

        print(f"\nTraining complete — mAP50: {results.box.map50:.3f}")


if __name__ == "__main__":
    main()
