import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

from labels import CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model

def build_loader(data_dir: str, batch_size: int = 64):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tfm)
    assert ds.classes == CLASSES, f"Folder classes {ds.classes} != {CLASSES}"
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/trashnet_mobilenetv2.pt")
    ap.add_argument("--data_dir", required=True, help="Evaluate on this folder (must be ImageFolder structure).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    loader, ds = build_loader(args.data_dir, batch_size=args.batch_size)

    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    with (out_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    print(report_text)
    print(f"Saved: {out_dir/'classification_report.txt'}")
    print(f"Saved: {out_dir/'classification_report.json'}")
    print(f"Saved: {out_dir/'confusion_matrix.csv'}")

if __name__ == "__main__":
    main()