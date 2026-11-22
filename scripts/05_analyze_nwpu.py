"""
Project: YOLO-World Remote-Sensing Detection
Experiment: DATA-01
Purpose: Audit NWPU VHR-10 annotations and images.
"""

from pathlib import Path
from collections import Counter
from PIL import Image
import re

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/NWPU-VHR-10"
ANNOTATIONS = DATA / "ground_truth"

pattern = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\),"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*(\d+)"
)

extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

images = {
    p.stem: p
    for p in DATA.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
}

annotations = sorted(ANNOTATIONS.glob("*.txt"))
class_counts = Counter()
errors = []
total_boxes = 0

for ann in annotations:
    image_path = images.get(ann.stem)

    if image_path is None:
        errors.append(f"Missing image: {ann.name}")
        continue

    with Image.open(image_path) as image:
        width, height = image.size

    for line_number, line in enumerate(
        ann.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        match = pattern.fullmatch(line.strip())

        if not match:
            errors.append(
                f"Invalid format: {ann.name}:{line_number}"
            )
            continue

        x1, y1, x2, y2, cls = map(int, match.groups())

        if not (0 <= x1 < x2 <= width and
                0 <= y1 < y2 <= height):
            errors.append(
                f"Invalid box: {ann.name}:{line_number}"
            )

        class_counts[cls] += 1
        total_boxes += 1

unannotated = set(images) - {
    p.stem for p in annotations
}

print("\n=== NWPU VHR-10 DATASET AUDIT ===")
print("Total images:", len(images))
print("Annotation files:", len(annotations))
print("Images without annotations:", len(unannotated))
print("Total bounding boxes:", total_boxes)

print("\n=== CLASS DISTRIBUTION ===")
for cls, count in sorted(class_counts.items()):
    print(f"Class {cls}: {count} objects")

print("\n=== DATA QUALITY ===")
print("Issues detected:", len(errors))

for error in errors[:20]:
    print(error)

if len(errors) > 20:
    print(f"... and {len(errors)-20} more issues")

print("\nAudit completed.")
