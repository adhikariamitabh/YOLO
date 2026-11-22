"""
Project: YOLO-World Remote-Sensing Detection
Experiment: DATA-02
Purpose: Visual inspection of NWPU VHR-10 annotations.

The original images and annotations are not modified.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import random
import re

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/NWPU-VHR-10"

POSITIVE = DATA / "positive_image_set"
NEGATIVE = DATA / "negative_image_set"
ANNOTATIONS = DATA / "ground_truth"

OUTPUT = ROOT / "results/nwpu_visual_qc"
OUTPUT.mkdir(parents=True, exist_ok=True)

PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)"
)

random_generator = random.Random(42)

positive_images = sorted(POSITIVE.glob("*.jpg"))
negative_images = sorted(NEGATIVE.glob("*.jpg"))

# Sample images reproducibly
selected_positive = random_generator.sample(
    positive_images, 5
)
selected_negative = random_generator.sample(
    negative_images, 2
)

print("=== POSITIVE IMAGES ===")

for image_path in selected_positive:

    annotation_path = (
        ANNOTATIONS / f"{image_path.stem}.txt"
    )

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    count = 0

    for line in annotation_path.read_text().splitlines():
        match = PATTERN.fullmatch(line.strip())

        if not match:
            continue

        x1, y1, x2, y2, cls = map(
            int, match.groups()
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 12)),
            f"Class {cls}",
            fill="yellow"
        )

        count += 1

    output_path = (
        OUTPUT / f"positive_{image_path.name}"
    )

    image.save(output_path)

    print(
        f"{image_path.name}: "
        f"{count} bounding boxes -> {output_path}"
    )

print("\n=== NEGATIVE IMAGES ===")

for image_path in selected_negative:

    image = Image.open(image_path).convert("RGB")

    output_path = (
        OUTPUT / f"negative_{image_path.name}"
    )

    image.save(output_path)

    print(
        f"{image_path.name}: "
        f"No annotations -> {output_path}"
    )

print("\nVisualization completed.")
