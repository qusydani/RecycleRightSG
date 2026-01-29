import argparse
from pathlib import Path
import random
import shutil

from labels import CLASSES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()])

def copy_files(files, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dst_dir / src.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="e.g. data/raw/trashnet/dataset-resized")
    ap.add_argument("--out_dir", default="data/processed/trashnet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    assert abs((args.train + args.val + args.test) - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    random.seed(args.seed)

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        cls_src = src_dir / cls
        if not cls_src.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_src}")

        files = list_images(cls_src)
        random.shuffle(files)

        n = len(files)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        # remainder goes to test
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        copy_files(train_files, out_dir / "train" / cls)
        copy_files(val_files, out_dir / "val" / cls)
        copy_files(test_files, out_dir / "test" / cls)

        print(f"{cls}: total={n} train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print(f"Done. Output at: {out_dir}")

if __name__ == "__main__":
    main()
