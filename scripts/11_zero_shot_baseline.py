"""
Project: YOLO-World Remote-Sensing Object Detection
Course: CS 7267
Experiment: E01 - Zero-Shot Baseline

Purpose:
Evaluate pretrained YOLO-World on NWPU VHR-10
without any target-domain training.

Dataset: 120 validation images
Ground truth: 520 annotated objects
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import torch
import ultralytics
import yaml
from ultralytics import YOLOWorld


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models/yolov8s-worldv2.pt"

DATA_YAML = (
    ROOT / "data/processed/nwpu_yolo/data.yaml"
)

OUTPUT = ROOT / "results/zero_shot"
RUNS = ROOT / "runs"

OUTPUT.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ID = "E01"
IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
NMS_IOU = 0.70


# --------------------------------------------------
# 2. Verify files and GPU
# --------------------------------------------------

assert MODEL_PATH.exists(), "Model weights missing"
assert DATA_YAML.exists(), "Dataset YAML missing"
assert torch.cuda.is_available(), "GPU unavailable"

with DATA_YAML.open() as f:
    dataset_config = yaml.safe_load(f)

names = dataset_config["names"]

if isinstance(names, dict):
    class_names = [
        names[i] for i in range(len(names))
    ]
else:
    class_names = list(names)

assert len(class_names) == 10

val_directory = (
    ROOT / "data/processed/nwpu_yolo/images/val"
)

images = [
    p for p in val_directory.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

assert len(images) == 120, (
    f"Expected 120 validation images, found {len(images)}"
)

print("\n=== E01 ZERO-SHOT BASELINE ===")

print("Model:", MODEL_PATH.name)
print("Dataset:", DATA_YAML)
print("Validation images:", len(images))
print("Classes:", class_names)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)


# --------------------------------------------------
# 3. Load pretrained YOLO-World
# --------------------------------------------------

model = YOLOWorld(str(MODEL_PATH))

# Configure text prompts.
# This does not fine-tune the model.
model.set_classes(class_names)

assert list(model.names.values()) == class_names \
    if isinstance(model.names, dict) \
    else list(model.names) == class_names

print("\nCustom text classes configured.")
print("No training will be performed.")


# --------------------------------------------------
# 4. Run zero-shot validation
# --------------------------------------------------

metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    device=0,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    workers=2,
    conf=CONFIDENCE,
    iou=NMS_IOU,
    max_det=300,
    plots=True,
    save_txt=True,
    project=str(RUNS),
    name="e01_zero_shot_baseline",
    exist_ok=True,
    verbose=True,
)


# --------------------------------------------------
# 5. Collect evaluation metrics
# --------------------------------------------------

results = {
    "experiment": EXPERIMENT_ID,
    "description": "Zero-shot class-name baseline",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "model": MODEL_PATH.name,
    "dataset": "NWPU VHR-10",
    "split": "val",
    "images": 120,
    "ground_truth_objects": 520,

    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "confidence_threshold": CONFIDENCE,
    "nms_iou_threshold": NMS_IOU,

    "training": False,
    "fine_tuning": False,

    "pytorch": torch.__version__,
    "ultralytics": ultralytics.__version__,
    "gpu": torch.cuda.get_device_name(0),

    "prompts": class_names,

    "metrics": {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
    },

    "per_class_map50_95": {
        class_names[i]: float(ap)
        for i, ap in enumerate(metrics.box.maps)
    },

    "speed_ms": {
        k: float(v)
        for k, v in metrics.speed.items()
    },
}


# --------------------------------------------------
# 6. Save experiment results
# --------------------------------------------------

output_file = OUTPUT / "e01_baseline_metrics.json"

output_file.write_text(
    json.dumps(results, indent=2) + "\n"
)


# --------------------------------------------------
# 7. Display summary
# --------------------------------------------------

print("\n=== E01 RESULTS ===")

print("Precision:", f"{metrics.box.mp:.4f}")
print("Recall:", f"{metrics.box.mr:.4f}")
print("mAP@0.50:", f"{metrics.box.map50:.4f}")
print("mAP@0.75:", f"{metrics.box.map75:.4f}")
print("mAP@0.50:0.95:", f"{metrics.box.map:.4f}")

print("\n=== PER-CLASS mAP@0.50:0.95 ===")

for name, ap in results["per_class_map50_95"].items():
    print(f"{name}: {ap:.4f}")

print("\nResults saved:", output_file)
print("Validation plots saved:", metrics.save_dir)

print("\nE01 zero-shot evaluation completed.")
