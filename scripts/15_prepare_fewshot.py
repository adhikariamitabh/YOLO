"""
Project: YOLO-World Remote-Sensing Detection
Course: CS 7267

Step 21: Prepare image-based few-shot datasets.

Targets:
  1 annotated image per class
  5 annotated images per class
  10 annotated images per class

All original labels in selected images are retained.
Only TRAIN images are eligible for selection.
Validation and test data are never used for selection.

No model training is performed by this script.
"""

from pathlib import Path
from collections import Counter
import csv
import json
import random
import shutil

import yaml

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

FULL = ROOT / "data/processed/nwpu_yolo"

MANIFEST = ROOT / "configs/nwpu_split_manifest.csv"

OUTPUT = ROOT / "data/processed/nwpu_fewshot"

CONFIG = ROOT / "configs"

SEED = 42
SHOT_LEVELS = [1, 5, 10]

CONFIG.mkdir(parents=True, exist_ok=True)

with (FULL / "data.yaml").open() as f:
    original_yaml = yaml.safe_load(f)

names = original_yaml["names"]

CLASS_NAMES = [
    names[i] if i in names else names[str(i)]
    for i in range(10)
]

# --------------------------------------------------
# 2. Read original split manifest
# --------------------------------------------------

with MANIFEST.open(newline="") as f:
    all_records = list(csv.DictReader(f))

assert len(all_records) == 800

train_records = [
    r for r in all_records
    if r["split"] == "train"
    and r["image_type"] == "positive"
]

assert len(train_records) == 455

for record in train_records:
    record["classes"] = {
        int(x)
        for x in record["class_ids"].split("|")
        if x.strip()
    }

    assert record["classes"]
    assert all(1 <= c <= 10 for c in record["classes"])

# --------------------------------------------------
# 3. Identify available images for every class
# --------------------------------------------------

ranked = {}
availability = {}

print("\n=== FEW-SHOT DATASET AVAILABILITY ===")

for class_id in range(1, 11):

    candidates = [
        record
        for record in train_records
        if class_id in record["classes"]
    ]

    candidates.sort(key=lambda x: x["image_path"])

    rng = random.Random(SEED + class_id)
    rng.shuffle(candidates)

    ranked[class_id] = candidates
    availability[class_id] = len(candidates)

    print(
        f"Class {class_id - 1:2d} "
        f"{CLASS_NAMES[class_id - 1]:<22} "
        f"available images: {len(candidates)}"
    )

# --------------------------------------------------
# 4. Prepare few-shot datasets
# --------------------------------------------------

summary = {
    "seed": SEED,
    "shot_definition": "annotated images per class",
    "available_images_per_class": {
        CLASS_NAMES[c - 1]: availability[c]
        for c in range(1, 11)
    },
    "subsets": {},
}

selection_rows = []

for shots in SHOT_LEVELS:

    print(f"\n=== PREPARING {shots}-SHOT DATASET ===")

    insufficient = [
        CLASS_NAMES[c - 1]
        for c in range(1, 11)
        if availability[c] < shots
    ]

    if insufficient:
        print(
            "SKIPPED: insufficient distinct training "
            "images for:",
            ", ".join(insufficient)
        )

        summary["subsets"][str(shots)] = {
            "status": "unavailable",
            "insufficient_classes": insufficient,
        }
        continue

    selected = {}

    for class_id in range(1, 11):

        anchors = ranked[class_id][:shots]

        for rank, record in enumerate(anchors, 1):

            selected[record["image_path"]] = record

            selection_rows.append({
                "shot_level": shots,
                "class_id": class_id - 1,
                "class_name": CLASS_NAMES[class_id - 1],
                "rank": rank,
                "image_path": record["image_path"],
            })

    subset = OUTPUT / f"shot_{shots:02d}"

    image_dir = subset / "images/train"
    label_dir = subset / "labels/train"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    expected_names = {
        "positive_" + Path(r["image_path"]).name
        for r in selected.values()
    }

    existing_names = {
        p.name for p in image_dir.iterdir()
        if p.is_file()
    }

    if existing_names - expected_names:
        raise RuntimeError(
            f"Unexpected existing files in {image_dir}. "
            "Review before rerunning."
        )

    box_counts = Counter()
    total_boxes = 0

    # Copy images and corresponding complete labels.
    for record in sorted(
        selected.values(),
        key=lambda r: r["image_path"]
    ):

        original_name = Path(record["image_path"]).name

        name = "positive_" + original_name
        stem = Path(name).stem

        src_image = FULL / "images/train" / name
        src_label = FULL / "labels/train" / f"{stem}.txt"

        if not src_image.exists() or not src_label.exists():
            raise FileNotFoundError(
                f"Missing source: {name}"
            )

        labels = [
            line.strip()
            for line in src_label.read_text().splitlines()
            if line.strip()
        ]

        assert len(labels) == int(record["box_count"])

        for line in labels:
            fields = line.split()
            assert len(fields) == 5

            class_id = int(fields[0])
            assert 0 <= class_id <= 9

            box_counts[class_id] += 1
            total_boxes += 1

        shutil.copy2(src_image, image_dir / name)
        shutil.copy2(src_label, label_dir / f"{stem}.txt")

    # --------------------------------------------------
    # 5. Create dataset YAML
    # --------------------------------------------------

    config = {
        "path": str(subset.resolve()),
        "train": "images/train",
        "val": str((FULL / "images/val").resolve()),
        "test": str((FULL / "images/test").resolve()),
        "nc": 10,
        "names": {
            i: name
            for i, name in enumerate(CLASS_NAMES)
        },
    }

    yaml_path = subset / "data.yaml"

    yaml_path.write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    # --------------------------------------------------
    # 6. Verify subset
    # --------------------------------------------------

    image_files = list(image_dir.iterdir())
    label_files = list(label_dir.glob("*.txt"))

    assert len(image_files) == len(selected)
    assert len(label_files) == len(selected)
    assert sum(box_counts.values()) == total_boxes

    summary["subsets"][str(shots)] = {
        "status": "created",
        "images": len(selected),
        "bounding_boxes": total_boxes,
        "objects_per_class": {
            CLASS_NAMES[i]: box_counts[i]
            for i in range(10)
        },
        "data_yaml": str(yaml_path),
    }

    print("Unique training images:", len(selected))
    print("Annotated objects:", total_boxes)
    print("Dataset YAML:", yaml_path)

    print("\nObject instances per class:")

    for class_id, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {box_counts[class_id]}")

# --------------------------------------------------
# 7. Save reproducibility records
# --------------------------------------------------

selection_path = (
    CONFIG / "nwpu_fewshot_selections.csv"
)

with selection_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "shot_level",
            "class_id",
            "class_name",
            "rank",
            "image_path",
        ],
    )

    writer.writeheader()
    writer.writerows(selection_rows)

summary_path = (
    CONFIG / "nwpu_fewshot_summary.json"
)

summary_path.write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("\n=== FINAL VERIFICATION ===")
print("Original training partition preserved.")
print("Validation partition unchanged.")
print("Test partition unchanged.")
print("No model training performed.")
print("Selections:", selection_path)
print("Summary:", summary_path)

print("\nFew-shot dataset preparation completed.")
