# src/train.py
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/master_dataset/data.yaml", help="Path to data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    # Load a pre-trained base model (transfer learning)
    model = YOLO("yolov8n.pt")

    # Train the model on your custom dataset
    print(f"Starting training on {args.data}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        plots=True, # Automatically generates confusion matrices and graphs
        device=0
    )

    print("Training complete! The best model is saved in the 'runs/detect/train/weights/' folder.")

if __name__ == "__main__":
    main()