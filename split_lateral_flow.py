from pathlib import Path
from shutil import copy2
import random
import csv
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data/lateral_flow"

SOURCE = BASE / "combined"
DEST = BASE / "dataset"

SEED = 42
rng = random.Random(SEED)

# Exact counts for each group
SPLITS = {
    "pos": {"train": 102, "val": 22, "test": 23},
    "neg": {"train": 105, "val": 23, "test": 22},
}

if DEST.exists():
    raise SystemExit("Dataset folder already exists. Check it first.")

images_dir = SOURCE / "images"
labels_dir = SOURCE / "labels"

all_images = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

if len(all_images) != 297:
    raise ValueError(f"Expected 297 images, found {len(all_images)}")

# Verify all labels before copying
for image in all_images:
    label = labels_dir / f"{image.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(label)

manifest = []

for group, counts in SPLITS.items():

    images = [
        p for p in all_images
        if p.name.startswith(group + "_")
    ]

    if len(images) != sum(counts.values()):
        raise ValueError(f"Unexpected {group} image count")

    rng.shuffle(images)
    start = 0

    for split, count in counts.items():

        selected = images[start:start + count]
        start += count

        out_images = DEST / "images" / split
        out_labels = DEST / "labels" / split

        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for image in selected:
            label = labels_dir / f"{image.stem}.txt"

            copy2(image, out_images / image.name)
            copy2(label, out_labels / label.name)

            manifest.append([image.name, group, split])

# Save split manifest
with open(DEST / "split_manifest.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "group", "split"])
    writer.writerows(manifest)

# Create YOLO OBB dataset configuration
config = {
    "path": str(DEST.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "test_window"},
}

with open(ROOT / "lateral_flow.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

for split in ("train", "val", "test"):
    count = len(list((DEST / "images" / split).iterdir()))
    print(f"{split}: {count} images")

print("Dataset preparation complete.")
