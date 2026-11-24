"""
CS 7267 — YOLO-World
Qualitative Detection Review

Combines existing E08 error-analysis images
into one contact sheet.

No model inference or training.
"""

from pathlib import Path
from math import ceil
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent.parent

GALLERY = (
    ROOT /
    "results/report/e08_error_analysis/gallery"
)

OUTPUT = (
    ROOT /
    "results/report/fig04_qualitative_review.jpg"
)

images = sorted(GALLERY.glob("*.png"))

assert len(images) == 12, (
    f"Expected 12 images, found {len(images)}"
)

COLUMNS = 2
ROWS = ceil(len(images) / COLUMNS)

CELL_W = 940
CELL_H = 430
HEADER = 70

sheet = Image.new(
    "RGB",
    (
        COLUMNS * CELL_W,
        HEADER + ROWS * CELL_H
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)

draw.text(
    (20, 20),
    "E08 YOLO-World — Qualitative Detection Review",
    fill="black"
)

for i, path in enumerate(images):

    col = i % COLUMNS
    row = i // COLUMNS

    x = col * CELL_W
    y = HEADER + row * CELL_H

    with Image.open(path) as image:
        preview = ImageOps.contain(
            image.convert("RGB"),
            (900, 365)
        )

    sheet.paste(
        preview,
        (x + 20, y + 5)
    )

    draw.text(
        (x + 20, y + 375),
        path.name,
        fill="black"
    )

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

sheet.save(
    OUTPUT,
    quality=95,
    optimize=True
)

print("\n=== QUALITATIVE REVIEW SHEET ===")
print("Input images:", len(images))
print("Output:", OUTPUT)
print("Training performed: NO")
print("Inference performed: NO")
print("\nQUALITATIVE SHEET: COMPLETE")
