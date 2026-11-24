"""
Project: YOLO-World Remote-Sensing Object Detection
Course: CS 7267

Step 16: Validate NWPU VHR-10 to YOLO conversion.

Checks:
1. All 800 images and labels exist.
2. Original annotation format is valid.
3. Converted class IDs match original classes.
4. YOLO coordinates are normalized correctly.
5. Converted bounding boxes match original pixel coordinates.
6. Negative images have empty annotation files.
7. All 3896 bounding boxes are accounted for.

Original dataset and converted files are not modified.
"""

from pathlib import Path
from collections import Counter
from PIL import Image
import csv
import math
import re

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data/raw/NWPU-VHR-10"
YOLO = ROOT / "data/processed/nwpu_yolo"
MANIFEST = ROOT / "configs/nwpu_split_manifest.csv"

EXPECTED_IMAGES = {
    "train": 560,
    "val": 120,
    "test": 120,
}

EXPECTED_BOXES = {
    "train": 2848,
    "val": 520,
    "test": 528,
}

# Original NWPU annotation:
# (x1,y1),(x2,y2),class_id
# Optional whitespace is allowed.

PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)\s*"
)

errors = []
image_counts = Counter()
box_counts = Counter()
positive_counts = Counter()
negative_counts = Counter()
seen_images = set()
validated_boxes = 0


def add_error(message):
    errors.append(message)


# Read split manifest
with MANIFEST.open(newline="") as f:
    records = list(csv.DictReader(f))

print("\n=== NWPU YOLO CONVERSION VALIDATION ===")

if len(records) != 800:
    add_error(
        f"Manifest contains {len(records)} records, expected 800"
    )


for record in records:

    split = record["split"]
    image_type = record["image_type"]

    if split not in EXPECTED_IMAGES:
        add_error(f"Invalid split: {split}")
        continue

    if image_type not in ("positive", "negative"):
        add_error(f"Invalid image type: {image_type}")
        continue

    src_image = RAW / record["image_path"]

    if not src_image.exists():
        add_error(f"Missing original image: {src_image}")
        continue

    stem = f"{image_type}_{src_image.stem}"

    dst_image = (
        YOLO / "images" / split /
        (stem + src_image.suffix.lower())
    )

    dst_label = (
        YOLO / "labels" / split /
        (stem + ".txt")
    )

    if dst_image in seen_images:
        add_error(f"Duplicate image assignment: {dst_image}")
        continue

    seen_images.add(dst_image)

    if not dst_image.exists():
        add_error(f"Missing converted image: {dst_image}")
        continue

    if not dst_label.exists():
        add_error(f"Missing converted label: {dst_label}")
        continue

    with Image.open(src_image) as img:
        width, height = img.size

    with Image.open(dst_image) as img:
        converted_size = img.size

    if converted_size != (width, height):
        add_error(f"Image dimensions changed: {stem}")
        continue

    # Read converted YOLO labels
    labels = [
        line.strip()
        for line in dst_label.read_text().splitlines()
        if line.strip()
    ]

    original_boxes = []

    if image_type == "positive":

        positive_counts[split] += 1

        annotation = RAW / record["annotation_path"]

        if not annotation.exists():
            add_error(f"Missing annotation: {annotation}")
            continue

        for line_number, line in enumerate(
            annotation.read_text().splitlines(), 1
        ):

            line = line.strip()

            if not line:
                continue

            match = PATTERN.fullmatch(line)

            if match is None:
                add_error(
                    f"Invalid source annotation: "
                    f"{annotation.name}:{line_number}"
                )
                continue

            original_boxes.append(
                tuple(map(int, match.groups()))
            )

    else:

        negative_counts[split] += 1

        if labels:
            add_error(
                f"Negative image has labels: {stem}"
            )

        if record["annotation_path"].strip():
            add_error(
                f"Negative image has source annotation: {stem}"
            )

    expected_count = int(record["box_count"])

    if len(labels) != expected_count:
        add_error(
            f"Incorrect label count: {stem}, "
            f"expected {expected_count}, got {len(labels)}"
        )

    if len(original_boxes) != expected_count:
        add_error(
            f"Incorrect original box count: {stem}"
        )

    # Compare converted labels with original coordinates
    for label, original in zip(labels, original_boxes):

        try:
            values = label.split()

            if len(values) != 5:
                raise ValueError("Expected 5 YOLO fields")

            class_id = int(values[0])

            xc, yc, bw, bh = map(float, values[1:])

            x1, y1, x2, y2, original_class = original

            # Verify class ID conversion: 1-10 -> 0-9
            if class_id != original_class - 1:
                raise ValueError("Class ID mismatch")

            if not 0 <= class_id <= 9:
                raise ValueError("Invalid class ID")

            # Verify normalized coordinates
            if not all(
                math.isfinite(v)
                for v in (xc, yc, bw, bh)
            ):
                raise ValueError("Non-finite coordinate")

            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < bw <= 1
                and 0 < bh <= 1
            ):
                raise ValueError("Invalid normalized coordinates")

            # Convert YOLO normalized coordinates
            # back into pixel coordinates
            recovered = (
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height
            )

            original_coords = (x1, y1, x2, y2)

            # Allow small floating-point rounding error
            tolerance = 0.01

            if any(
                abs(a - b) > tolerance
                for a, b in zip(recovered, original_coords)
            ):
                raise ValueError("Bounding-box mismatch")

            validated_boxes += 1

        except (ValueError, TypeError) as e:
            add_error(f"{stem}: {e}")

    image_counts[split] += 1
    box_counts[split] += len(labels)


# Verify expected counts
for split in ("train", "val", "test"):

    if image_counts[split] != EXPECTED_IMAGES[split]:
        add_error(f"{split}: incorrect image count")

    if box_counts[split] != EXPECTED_BOXES[split]:
        add_error(f"{split}: incorrect bounding-box count")


print("\n=== SPLIT VALIDATION ===")

for split in ("train", "val", "test"):

    print(f"\n{split.upper()}")
    print("Images:", image_counts[split])
    print("Positive:", positive_counts[split])
    print("Negative:", negative_counts[split])
    print("Bounding boxes:", box_counts[split])


print("\n=== TOTAL VALIDATION ===")

print("Images validated:", sum(image_counts.values()))
print("Bounding boxes validated:", validated_boxes)
print("Issues detected:", len(errors))


if errors:

    print("\n=== FIRST 20 ERRORS ===")

    for error in errors[:20]:
        print(error)

    if len(errors) > 20:
        print(f"... and {len(errors) - 20} additional errors")

    raise RuntimeError("YOLO conversion validation FAILED")


if sum(image_counts.values()) != 800:
    raise RuntimeError("Incorrect total image count")

if validated_boxes != 3896:
    raise RuntimeError("Incorrect total bounding-box count")


print("\n================================")
print("YOLO CONVERSION VALIDATION: PASS")
print("================================")
print("All 800 images verified.")
print("All 3896 bounding boxes verified.")
print("Class IDs and normalized coordinates verified.")
print("Original dataset remains unchanged.")
