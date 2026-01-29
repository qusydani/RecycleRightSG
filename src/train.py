import argparse
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm

from labels import CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_loaders(train_dir, val_dir, img_size=224, batch_size=32):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm_val)

    # ImageFolder sorts class folder names alphabetically, enforce consistent mapping
    assert train_ds.classes == CLASSES, f"Train folder classes {train_ds.classes} != {CLASSES}"
    assert val_ds.classes == CLASSES, f"Val folder classes {val_ds.classes} != {CLASSES}"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/trashnet_mobilenetv2.pt")
    ap.add_argument("--unfreeze_last_n", type=int, default=0,
                    help="Optional: unfreeze last N layers of model.features for fine-tuning (0 = feature extractor only).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_loaders(
        args.train_dir, args.val_dir,
        batch_size=args.batch_size
    )

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all feature layers first
    for p in model.features.parameters():
        p.requires_grad = False

    # Optionally unfreeze last N feature blocks for fine-tuning
    if args.unfreeze_last_n > 0:
        for block in list(model.features.children())[-args.unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    model = model.to(device)

    # Train only parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"train epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for x, y in tqdm(val_loader, desc=f"val epoch {epoch}"):
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1).cpu()
                y_pred.extend(pred.tolist())
                y_true.extend(y.tolist())

        print(classification_report(
            y_true, y_pred,
            target_names=CLASSES,
            digits=3,
            zero_division=0
        ))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_name": "mobilenet_v2",
        "classes": CLASSES,
        "state_dict": model.state_dict(),
    }, out_path)

    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "model_name": "mobilenet_v2"}, f, indent=2)

    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
