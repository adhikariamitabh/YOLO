"""
CS 7267 — YOLO-World
E08 qualitative error analysis.

Read-only analysis of the final test partition.
Uses the existing E08 best checkpoint.

Diagnostic settings:
  prediction confidence >= 0.25
  correct-class matching IoU >= 0.50

These are diagnostic counts, NOT official
precision, recall, or mAP calculations.

No training, tuning, or model changes.
"""

from pathlib import Path
from collections import defaultdict
import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from ultralytics import YOLOWorld


ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = (
    ROOT / "runs/full/e08_full/weights/best.pt"
)

IMAGE_DIR = (
    ROOT / "data/processed/nwpu_yolo/images/test"
)

LABEL_DIR = (
    ROOT / "data/processed/nwpu_yolo/labels/test"
)

OUTPUT = ROOT / "results/report/e08_error_analysis"
GALLERY = OUTPUT / "gallery"

OUTPUT.mkdir(parents=True, exist_ok=True)
GALLERY.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "aircraft",
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

CONF = 0.25
MATCH_IOU = 0.50

assert WEIGHTS.exists()
assert torch.cuda.is_available()

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

assert len(images) == 120


# --------------------------------------------------
# 1. Utilities
# --------------------------------------------------

def ordered_names(names):
    if isinstance(names, dict):
        return [
            names[k]
            for k in sorted(names, key=lambda x: int(x))
        ]
    return list(names)


def calculate_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1)
        * max(0, iy2 - iy1)
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def load_ground_truth(image_path, width, height):
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    boxes = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        fields = line.split()
        assert len(fields) == 5

        cls = int(fields[0])
        x, y, w, h = map(float, fields[1:])

        boxes.append({
            "cls": cls,
            "box": [
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            ],
        })

    return boxes


# --------------------------------------------------
# 2. Load model
# --------------------------------------------------

print("\n=== E08 ERROR ANALYSIS ===")

model = YOLOWorld(str(WEIGHTS))

checkpoint_classes = ordered_names(model.names)

assert checkpoint_classes == CLASSES, (
    "Checkpoint class names differ from the "
    "E08 hybrid vocabulary. Stop and inspect "
    "the model configuration before predicting."
)

print("Checkpoint:", WEIGHTS)
print("Images:", len(images))
print("Confidence:", CONF)
print("Matching IoU:", MATCH_IOU)
print("Training: NO")


# --------------------------------------------------
# 3. Collect detections
# --------------------------------------------------

class_totals = defaultdict(
    lambda: {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
)

records = []
details = {}

predictions = model.predict(
    source=[str(p) for p in images],
    stream=True,
    imgsz=640,
    device=0,
    conf=CONF,
    iou=0.70,
    max_det=300,
    verbose=False,
)

for result in predictions:

    image_path = Path(result.path)

    image = cv2.imread(str(image_path))
    assert image is not None

    height, width = image.shape[:2]

    gt = load_ground_truth(
        image_path, width, height
    )

    pred = []

    for box in result.boxes:
        pred.append({
            "cls": int(box.cls.item()),
            "conf": float(box.conf.item()),
            "box": box.xyxy[0].cpu().tolist(),
            "matched": False,
        })

    pred.sort(
        key=lambda p: p["conf"],
        reverse=True
    )

    matched_gt = set()

    # Match predictions to ground truth using
    # class-aware greedy IoU matching.
    for prediction in pred:

        best_iou = MATCH_IOU
        best_gt = None

        for index, target in enumerate(gt):

            if index in matched_gt:
                continue

            if prediction["cls"] != target["cls"]:
                continue

            score = calculate_iou(
                prediction["box"],
                target["box"]
            )

            if score >= best_iou:
                best_iou = score
                best_gt = index

        if best_gt is not None:
            matched_gt.add(best_gt)
            prediction["matched"] = True

    for index, target in enumerate(gt):
        c = target["cls"]
        class_totals[c]["gt"] += 1

        if index in matched_gt:
            class_totals[c]["tp"] += 1
            target["matched"] = True
        else:
            class_totals[c]["fn"] += 1
            target["matched"] = False

    for prediction in pred:
        if not prediction["matched"]:
            class_totals[prediction["cls"]]["fp"] += 1

    tp_count = sum(p["matched"] for p in pred)
    fp_count = len(pred) - tp_count
    fn_count = len(gt) - len(matched_gt)

    records.append({
        "image": image_path.name,
        "ground_truth": len(gt),
        "predictions": len(pred),
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
    })

    details[image_path.name] = {
        "path": image_path,
        "gt": gt,
        "pred": pred,
    }


# --------------------------------------------------
# 4. Save per-image CSV
# --------------------------------------------------

csv_path = OUTPUT / "e08_per_image_errors.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )

    writer.writeheader()
    writer.writerows(records)


# --------------------------------------------------
# 5. Save class-level diagnostics
# --------------------------------------------------

class_rows = []

for c, name in enumerate(CLASSES):
    counts = class_totals[c]

    class_rows.append({
        "class": name,
        **counts,
    })

class_csv = OUTPUT / "e08_class_errors.csv"

with class_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["class", "gt", "tp", "fp", "fn"]
    )

    writer.writeheader()
    writer.writerows(class_rows)


# --------------------------------------------------
# 6. Draw comparison images
# --------------------------------------------------

def draw_boxes(image, items, prediction=False):
    canvas = image.copy()

    for item in items:
        x1, y1, x2, y2 = map(int, item["box"])

        matched = item["matched"]

        # Green = matched, red = unmatched.
        color = (
            (0, 180, 0) if matched
            else (0, 0, 255)
        )

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = CLASSES[item["cls"]]

        if prediction:
            label += f" {item['conf']:.2f}"

        cv2.putText(
            canvas,
            label,
            (max(0, x1), max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA
        )

    return canvas


def to_thumbnail(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return ImageOps.contain(
        Image.fromarray(rgb),
        (490, 355)
    )


def create_comparison(info, destination):
    image = cv2.imread(str(info["path"]))
    assert image is not None

    gt_view = draw_boxes(image, info["gt"])
    pred_view = draw_boxes(
        image, info["pred"], prediction=True
    )

    sheet = Image.new(
        "RGB", (1020, 410), "white"
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (10, 8),
        "GROUND TRUTH",
        fill="black"
    )

    draw.text(
        (520, 8),
        "E08 PREDICTIONS",
        fill="black"
    )

    sheet.paste(
        to_thumbnail(gt_view),
        (10, 30)
    )

    sheet.paste(
        to_thumbnail(pred_view),
        (520, 30)
    )

    draw.text(
        (10, 390),
        "Green: matched | Red: unmatched",
        fill="black"
    )

    sheet.save(destination)


# Prioritize the storage-tank examples,
# then images with the most missed objects.
storage_images = [
    name for name, info in details.items()
    if any(
        target["cls"] == 2
        for target in info["gt"]
    )
]

ranked = sorted(
    records,
    key=lambda r: (
        r["fn"],
        r["fp"],
    ),
    reverse=True
)

selected = list(storage_images)

for row in ranked:
    if len(selected) >= 12:
        break

    name = row["image"]

    if name not in selected:
        selected.append(name)

for index, name in enumerate(selected, 1):
    destination = (
        GALLERY / f"{index:02d}_{Path(name).stem}.png"
    )

    create_comparison(details[name], destination)


# --------------------------------------------------
# 7. Save summary
# --------------------------------------------------

totals = {
    key: sum(row[key] for row in records)
    for key in ("tp", "fp", "fn")
}

summary = {
    "experiment": "E08",
    "split": "test",
    "images": len(records),
    "diagnostic_confidence": CONF,
    "matching_iou": MATCH_IOU,
    "counts": totals,
    "classes": class_rows,
    "gallery_images": selected,
    "official_map_replaced": False,
    "training_performed": False,
    "model_selection_performed": False,
    "notes": (
        "Diagnostic counts at a fixed confidence "
        "threshold. They are not official AP metrics. "
        "Unannotated objects may be counted as false "
        "positives. Two train-test similarity candidates "
        "remain unresolved."
    ),
}

summary_file = OUTPUT / "e08_error_summary.json"

summary_file.write_text(
    json.dumps(summary, indent=2) + "\n"
)


# --------------------------------------------------
# 8. Display findings
# --------------------------------------------------

print("\n=== E08 DIAGNOSTIC COUNTS ===")

print("Images analyzed:", len(records))
print("Matched predictions:", totals["tp"])
print("Unmatched predictions:", totals["fp"])
print("Unmatched ground truths:", totals["fn"])

print("\n=== CLASS ERROR COUNTS ===")

print(
    f"{'Class':<24}"
    f"{'GT':>6}"
    f"{'TP':>6}"
    f"{'FP':>6}"
    f"{'FN':>6}"
)

for row in class_rows:
    print(
        f"{row['class']:<24}"
        f"{row['gt']:>6}"
        f"{row['tp']:>6}"
        f"{row['fp']:>6}"
        f"{row['fn']:>6}"
    )

print("\n=== OUTPUT FILES ===")

print("Summary:", summary_file)
print("Per-image CSV:", csv_path)
print("Class CSV:", class_csv)
print("Gallery:", GALLERY)
print("Gallery images:", len(selected))

print("\nTraining: NO")
print("Model changed: NO")
print("Official test mAP changed: NO")

print("\nE08 ERROR ANALYSIS: COMPLETE")
