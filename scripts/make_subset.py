"""Build data/raw from the downloaded tomato leaf dataset.

Running this records exactly how the working dataset was produced, which classes were
kept and how many images per class, so anyone can rebuild it from the original download.

The full dataset (Kaggle kaustubhb999/tomatoleaf, also on Zenodo record 8311631) has
train/ and val/ folders with ten classes. LeafGuard uses six of them with a cap per class,
which keeps a full pipeline run to a few minutes on four CPU cores.

Usage:
  python scripts/make_subset.py --src data/download/extracted/tomato-dataset
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# short name used everywhere in the project -> folder name in the download
CLASSES = {
    "bacterial_spot": "Tomato___Bacterial_spot",
    "early_blight": "Tomato___Early_blight",
    "healthy": "Tomato___healthy",
    "late_blight": "Tomato___Late_blight",
    "leaf_mold": "Tomato___Leaf_Mold",
    "septoria": "Tomato___Septoria_leaf_spot",
}
SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="unzipped tomato-dataset folder")
    parser.add_argument("--dst", default=Path("data/raw"), type=Path)
    parser.add_argument("--train-cap", type=int, default=200, dest="train_cap")
    parser.add_argument("--val-cap", type=int, default=40, dest="val_cap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    caps = {"train": args.train_cap, "val": args.val_cap}
    total = 0

    for split, cap in caps.items():
        for short_name, folder_name in CLASSES.items():
            source = args.src / split / folder_name
            if not source.is_dir():
                raise SystemExit(f"folder not found: {source}")
            images = sorted(p for p in source.iterdir() if p.suffix.lower() in SUFFIXES)
            random.shuffle(images)
            target = args.dst / split / short_name
            target.mkdir(parents=True, exist_ok=True)
            for image in images[:cap]:
                shutil.copy2(image, target / image.name)
            kept = min(cap, len(images))
            total += kept
            print(f"{split}/{short_name}: kept {kept} of {len(images)}")

    print(f"\n{total} images written to {args.dst}")
    print("Next:  dvc add data/raw && git add data/raw.dvc data/.gitignore && git commit")


if __name__ == "__main__":
    main()
