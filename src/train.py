# src/train.py
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/master_dataset/data.yaml", help="Path to data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    # Load the new YOLO26 Nano model
    model = YOLO("yolo26n.pt")

    # Train the model with early stopping
    print(f"Starting YOLO26 training on {args.data}...")
    results = model.train(
        data=args.data,
        epochs=50,
        imgsz=640,
        patience=5,
        cache='disk',
        batch=16,
        plots=True,
        device=0 
    )

    print("Training complete! The best model is saved in the 'runs/detect/train/weights/' folder.")

if __name__ == "__main__":
    main()