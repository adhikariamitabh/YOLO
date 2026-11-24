"""
Project: YOLO-World Remote-Sensing Object Detection
Step 15: Convert NWPU VHR-10 annotations to YOLO format.

Input:
    configs/nwpu_split_manifest.csv
    data/raw/NWPU-VHR-10/

Output:
    data/processed/nwpu_yolo/

Original dataset files are not modified.
"""

from pathlib import Path
from collections import Counter
from PIL import Image
import csv
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/NWPU-VHR-10"
MANIFEST = ROOT / "configs/nwpu_split_manifest.csv"
OUTPUT = ROOT / "data/processed/nwpu_yolo"

CLASS_NAMES = [
    "airplane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "vehicle",
]

PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)"
)

EXPECTED_IMAGES = {
    "train": 560,
    "val": 120,
    "test": 120,
}

counts = Counter()
box_counts = Counter()
positive_counts = Counter()
negative_counts = Counter()
processed = set()

with MANIFEST.open(newline="") as f:
    records = list(csv.DictReader(f))

assert len(records) == 800, "Expected 800 images"

for record in records:
    split = record["split"]
    image_type = record["image_type"]

    assert split in EXPECTED_IMAGES
    assert image_type in {"positive", "negative"}

    src_image = DATA / record["image_path"]

    assert src_image.exists(), f"Missing: {src_image}"

    # Prefix filenames to prevent collisions.
    # Positive and negative directories share some names.
    stem = f"{image_type}_{src_image.stem}"

    image_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    dst_image = image_dir / (stem + src_image.suffix.lower())
    dst_label = label_dir / (stem + ".txt")

    assert dst_image not in processed, "Duplicate destination"
    processed.add(dst_image)

    with Image.open(src_image) as img:
        width, height = img.size

    assert width > 0 and height > 0

    labels = []

    if image_type == "positive":
        annotation = DATA / record["annotation_path"]

        assert annotation.exists(), f"Missing: {annotation}"

        for line in annotation.read_text().splitlines():
            if not line.strip():
                continue

            match = PATTERN.fullmatch(line.strip())

            if match is None:
                raise ValueError(
                    f"Invalid annotation: {annotation}: {line}"
                )

            x1, y1, x2, y2, original_class = map(
                int, match.groups()
            )

            assert 0 <= x1 < x2 <= width
            assert 0 <= y1 < y2 <= height
            assert 1 <= original_class <= 10

            # Original class 1-10 -> YOLO class 0-9
            class_id = original_class - 1

            # Normalize using image dimensions
            x_center = (x1 + x2) / (2 * width)
            y_center = (y1 + y2) / (2 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height

            values = [
                x_center, y_center,
                box_width, box_height
            ]

            assert all(0 <= v <= 1 for v in values)
            assert box_width > 0 and box_height > 0

            labels.append(
                f"{class_id} "
                f"{x_center:.8f} {y_center:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            )

            box_counts[split] += 1

        assert len(labels) == int(record["box_count"])
        positive_counts[split] += 1

    else:
        # Negative image: no annotated target objects
        assert not record["annotation_path"]
        assert int(record["box_count"]) == 0
        negative_counts[split] += 1

    # Copy original image, preserving source data
    shutil.copy2(src_image, dst_image)

    # Negative images receive an empty label file
    dst_label.write_text(
        "\n".join(labels) + ("\n" if labels else "")
    )

    counts[split] += 1


# Verify complete conversion
assert counts == EXPECTED_IMAGES
assert sum(box_counts.values()) == 3896

# Create Ultralytics dataset configuration
yaml_path = OUTPUT / "data.yaml"

yaml_lines = [
    f'path: "{OUTPUT.resolve()}"',
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "nc: 10",
    "names:",
]

for i, name in enumerate(CLASS_NAMES):
    yaml_lines.append(f"  {i}: '{name}'")

yaml_path.write_text("\n".join(yaml_lines) + "\n")


print("\n=== NWPU TO YOLO CONVERSION ===")

for split in ("train", "val", "test"):
    print(f"\n{split.upper()}")
    print("Images:", counts[split])
    print("Positive:", positive_counts[split])
    print("Negative:", negative_counts[split])
    print("Bounding boxes:", box_counts[split])

print("\nTotal images:", sum(counts.values()))
print("Total bounding boxes:", sum(box_counts.values()))
print("Classes:", len(CLASS_NAMES))
print("Dataset YAML:", yaml_path)
print("Output directory:", OUTPUT)

print("\nConversion completed successfully.")