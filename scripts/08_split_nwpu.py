"""
Project: YOLO-World Remote-Sensing Detection
Experiment: DATA-03
Purpose: Reproducible NWPU VHR-10 data splitting

Split: 70% train, 15% validation, 15% test
Random seed: 42

The original dataset remains unchanged.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import re

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/NWPU-VHR-10"
CONFIG = ROOT / "configs"
CONFIG.mkdir(exist_ok=True)

SEED = 42

SPLITS = {
    "train": {"positive": 455, "negative": 105},
    "val":   {"positive": 98,  "negative": 22},
    "test":  {"positive": 97,  "negative": 23},
}

PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_images(folder, image_type):
    records = []

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    for image in images:
        classes = set()
        box_count = 0
        ann_rel = ""

        if image_type == "positive":
            annotation = (
                DATA / "ground_truth" / f"{image.stem}.txt"
            )

            if not annotation.exists():
                raise FileNotFoundError(annotation)

            ann_rel = annotation.relative_to(DATA).as_posix()

            for line in annotation.read_text().splitlines():
                if not line.strip():
                    continue

                match = PATTERN.fullmatch(line.strip())

                if match is None:
                    raise ValueError(
                        f"Invalid annotation: {annotation}"
                    )

                cls = int(match.group(5))

                if cls not in range(1, 11):
                    raise ValueError(f"Invalid class: {cls}")

                classes.add(cls)
                box_count += 1

        records.append({
            "image_path": image.relative_to(DATA).as_posix(),
            "image_type": image_type,
            "annotation_path": ann_rel,
            "class_ids": sorted(classes),
            "box_count": box_count,
        })

    return records


positive = collect_images(
    DATA / "positive_image_set", "positive"
)
negative = collect_images(
    DATA / "negative_image_set", "negative"
)

assert len(positive) == 650
assert len(negative) == 150
assert sum(r["box_count"] for r in positive) == 3896

# Detect exact duplicate image files before splitting.
hashes = defaultdict(list)

for record in positive + negative:
    path = DATA / record["image_path"]

    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()

    hashes[digest].append(record["image_path"])

duplicates = [v for v in hashes.values() if len(v) > 1]

if duplicates:
    raise RuntimeError(
        f"Found {len(duplicates)} groups of identical "
        f"images. Review before splitting: {duplicates[:3]}"
    )

# Reproducible randomized assignment.
# Retry deterministically if a class is absent
# from any of the three partitions.

rng = random.Random(SEED)
all_classes = set(range(1, 11))

for attempt in range(1, 1001):
    pos = positive.copy()
    neg = negative.copy()

    rng.shuffle(pos)
    rng.shuffle(neg)

    partitions = {}
    pi = ni = 0

    for split, counts in SPLITS.items():
        pc = counts["positive"]
        nc = counts["negative"]

        partitions[split] = (
            pos[pi:pi + pc] + neg[ni:ni + nc]
        )

        pi += pc
        ni += nc

    coverage_ok = all(
        all_classes.issubset(
            set().union(
                *(set(r["class_ids"]) for r in records)
            )
        )
        for records in partitions.values()
    )

    if coverage_ok:
        break
else:
    raise RuntimeError(
        "Could not achieve complete class coverage. "
        "Inspect image-level class distribution."
    )

# Ensure images never appear in multiple splits.
paths = [
    r["image_path"]
    for records in partitions.values()
    for r in records
]

assert len(paths) == 800
assert len(set(paths)) == 800

# Save fixed assignments.
manifest = CONFIG / "nwpu_split_manifest.csv"

with manifest.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "split",
        "image_type",
        "image_path",
        "annotation_path",
        "class_ids",
        "box_count",
    ])

    for split, records in partitions.items():
        for r in sorted(
            records, key=lambda x: x["image_path"]
        ):
            writer.writerow([
                split,
                r["image_type"],
                r["image_path"],
                r["annotation_path"],
                "|".join(map(str, r["class_ids"])),
                r["box_count"],
            ])

# Build class distribution and report.
summary = {
    "dataset": "NWPU VHR-10",
    "seed": SEED,
    "selection_attempt": attempt,
    "total_images": 800,
    "total_boxes": 3896,
    "splits": {},
}

print("\n=== NWPU VHR-10 DATA SPLITS ===")
print("Random seed:", SEED)
print("Selection attempt:", attempt)

for split, records in partitions.items():
    types = Counter(r["image_type"] for r in records)

    object_counts = Counter()
    image_counts = Counter()

    for r in records:
        for cls in r["class_ids"]:
            image_counts[cls] += 1

        if r["annotation_path"]:
            annotation = DATA / r["annotation_path"]

            for line in annotation.read_text().splitlines():
                if line.strip():
                    cls = int(
                        PATTERN.fullmatch(line.strip()).group(5)
                    )
                    object_counts[cls] += 1

    summary["splits"][split] = {
        "total": len(records),
        "positive": types["positive"],
        "negative": types["negative"],
        "boxes": sum(r["box_count"] for r in records),
        "images_per_class": dict(sorted(image_counts.items())),
        "objects_per_class": dict(sorted(object_counts.items())),
    }

    print(f"\n{split.upper()}")
    print("Total images:", len(records))
    print("Positive:", types["positive"])
    print("Negative:", types["negative"])
    print("Annotated objects:", sum(object_counts.values()))
    print("Classes present:", sorted(object_counts))

summary_path = CONFIG / "nwpu_split_summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("\n=== VERIFICATION ===")
print("No image overlap: PASS")
print("All 10 classes in every split: PASS")
print("Exact duplicate image check: PASS")
print("Manifest:", manifest)
print("Summary:", summary_path)
print("\nDataset splitting completed successfully.")
