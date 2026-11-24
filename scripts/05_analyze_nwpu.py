"""
Project: YOLO-World Remote-Sensing Object Detection
Experiment: DATA-01
Purpose: Audit NWPU VHR-10 images and annotations.

Important:
- Positive images have bounding-box annotations.
- Negative images contain no labeled target objects.
- Both directories contain overlapping filenames.
- Original dataset files are never modified.
"""

from pathlib import Path
from collections import Counter
from PIL import Image
import re

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/NWPU-VHR-10"

POSITIVE = DATA / "positive_image_set"
NEGATIVE = DATA / "negative_image_set"
ANNOTATIONS = DATA / "ground_truth"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

pattern = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\),"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*(\d+)"
)

def get_images(folder):
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

positive_images = get_images(POSITIVE)
negative_images = get_images(NEGATIVE)
annotations = sorted(ANNOTATIONS.glob("*.txt"))

class_counts = Counter()
errors = []
total_boxes = 0

for ann in annotations:

    image_path = positive_images.get(ann.stem)

    if image_path is None:
        errors.append(f"Missing positive image: {ann.name}")
        continue

    with Image.open(image_path) as img:
        width, height = img.size

    for line_no, line in enumerate(
        ann.read_text().splitlines(), 1
    ):
        if not line.strip():
            continue

        match = pattern.fullmatch(line.strip())

        if not match:
            errors.append(
                f"Invalid format: {ann.name}:{line_no}"
            )
            continue

        x1, y1, x2, y2, cls = map(
            int, match.groups()
        )

        total_boxes += 1
        class_counts[cls] += 1

        if not (
            0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height
        ):
            errors.append(
                f"Invalid box: {ann.name}:{line_no} "
                f"box=({x1},{y1},{x2},{y2}) "
                f"image=({width},{height})"
            )

annotated_stems = {p.stem for p in annotations}
unannotated_positive = set(positive_images) - annotated_stems

print("\n=== NWPU VHR-10 DATASET AUDIT ===")
print("Positive images:", len(positive_images))
print("Negative images:", len(negative_images))
print("Total images:", len(positive_images) + len(negative_images))
print("Annotation files:", len(annotations))
print("Unannotated positive images:", len(unannotated_positive))
print("Total bounding boxes:", total_boxes)

print("\n=== CLASS DISTRIBUTION ===")
for cls, count in sorted(class_counts.items()):
    print(f"Class {cls}: {count} objects")

print("\n=== DATA QUALITY ===")
print("Issues detected:", len(errors))

for error in errors[:20]:
    print(error)

if len(errors) > 20:
    print(f"... and {len(errors) - 20} more issues")

print("\nAudit completed.")
