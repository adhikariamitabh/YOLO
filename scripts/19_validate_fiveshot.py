"""
Project: YOLO-World Remote-Sensing Detection
Course: CS 7267

Experiment E06: Standardized Five-Shot Validation

Compare E04 zero-shot, E05 one-shot,
and E06 five-shot models.

No training.
No test-set evaluation.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import torch
import yaml
from ultralytics import YOLOWorld


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = (
    ROOT /
    "runs/fewshot/e06_fiveshot/weights/best.pt"
)

DATA_YAML = (
    ROOT / "configs/e06_fiveshot_hybrid.yaml"
)

E04_FILE = (
    ROOT / "results/zero_shot/e04_hybrid_metrics.json"
)

E05_FILE = (
    ROOT / "results/fewshot/e05_validation_metrics.json"
)

OUTPUT = (
    ROOT / "results/fewshot/e06_validation_metrics.json"
)

RUN_DIR = ROOT / "runs/validation"

IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
NMS_IOU = 0.70


# --------------------------------------------------
# 2. Verify prerequisites
# --------------------------------------------------

for path in (WEIGHTS, DATA_YAML, E04_FILE, E05_FILE):
    if not path.exists():
        raise FileNotFoundError(path)

assert torch.cuda.is_available(), "CUDA unavailable"

with DATA_YAML.open() as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, dict):
    indexed = {int(k): v for k, v in names.items()}
    classes = [indexed[i] for i in range(len(indexed))]
else:
    classes = list(names)

assert len(classes) == 10
assert classes[0] == "aircraft"

e04 = json.loads(E04_FILE.read_text())
e05 = json.loads(E05_FILE.read_text())

assert e05["class_names"] == classes, (
    "E05 and E06 class-name configurations differ."
)


# --------------------------------------------------
# 3. Print configuration
# --------------------------------------------------

print("\n=== E06 STANDARDIZED VALIDATION ===")

print("Checkpoint:", WEIGHTS)
print("Dataset:", DATA_YAML)
print("GPU:", torch.cuda.get_device_name(0))
print("Classes:", classes)
print("Images:", 120)
print("Ground-truth objects:", 520)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Training: NO")
print("Test set used: NO")


# --------------------------------------------------
# 4. Load trained E06 model
# --------------------------------------------------

model = YOLOWorld(str(WEIGHTS))

validator = (
    model.task_map["detect"]["validator"].__name__
)

assert validator == "WorldValidator", (
    f"Unexpected validator: {validator}"
)


# --------------------------------------------------
# 5. Standardized evaluation
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
    project=str(RUN_DIR),
    name="e06_standardized",
    exist_ok=True,
    verbose=True,
)


# --------------------------------------------------
# 6. Collect metrics
# --------------------------------------------------

metric_values = {
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "map50": float(metrics.box.map50),
    "map75": float(metrics.box.map75),
    "map50_95": float(metrics.box.map),
}

per_class = {
    name: float(ap)
    for name, ap in zip(
        classes, metrics.box.maps
    )
}


# --------------------------------------------------
# 7. Compare E04, E05 and E06
# --------------------------------------------------

experiments = {
    "E04": e04["metrics"],
    "E05": e05["metrics"],
    "E06": metric_values,
}

print("\n=== E04 vs E05 vs E06 ===")

print(
    f"{'Metric':<16}"
    f"{'E04':>12}"
    f"{'E05':>12}"
    f"{'E06':>12}"
)

for key in metric_values:

    print(
        f"{key:<16}"
        f"{experiments['E04'][key]:>12.4f}"
        f"{experiments['E05'][key]:>12.4f}"
        f"{experiments['E06'][key]:>12.4f}"
    )


# --------------------------------------------------
# 8. Calculate E05-to-E06 improvements
# --------------------------------------------------

print("\n=== E05 TO E06 IMPROVEMENT ===")

differences = {}

for key, new in metric_values.items():

    old = float(e05["metrics"][key])
    change = new - old

    differences[key] = change

    print(
        f"{key:<16}"
        f"E05={old:.4f} "
        f"E06={new:.4f} "
        f"Change={change:+.4f}"
    )


# --------------------------------------------------
# 9. Compare class-level performance
# --------------------------------------------------

print("\n=== PER-CLASS mAP@0.50:0.95 ===")

print(
    f"{'Class':<24}"
    f"{'E05':>10}"
    f"{'E06':>10}"
    f"{'Change':>12}"
)

per_class_differences = {}

for name in classes:

    old = float(
        e05["per_class_map50_95"][name]
    )

    new = per_class[name]
    change = new - old

    per_class_differences[name] = change

    print(
        f"{name:<24}"
        f"{old:>10.4f}"
        f"{new:>10.4f}"
        f"{change:>+12.4f}"
    )


# --------------------------------------------------
# 10. Save standardized results
# --------------------------------------------------

results = {
    "experiment": "E06",
    "evaluation": "standardized_validation",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "model": "YOLOv8s-WorldV2",
    "checkpoint": str(WEIGHTS),

    "dataset": "NWPU VHR-10",
    "split": "val",

    "training_images": 50,
    "training_objects": 357,

    "validation_images": 120,
    "validation_objects": 520,

    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,

    "confidence_threshold": CONFIDENCE,
    "nms_iou_threshold": NMS_IOU,

    "class_names": classes,

    "metrics": metric_values,

    "per_class_map50_95": per_class,

    "comparison": experiments,

    "improvement_vs_e05": differences,

    "per_class_improvement_vs_e05": (
        per_class_differences
    ),

    "test_set_used": False,
}

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT.write_text(
    json.dumps(results, indent=2) + "\n"
)


# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

print("\n=== E06 VALIDATION COMPLETED ===")

print("Precision:", f"{metrics.box.mp:.4f}")
print("Recall:", f"{metrics.box.mr:.4f}")
print("mAP50:", f"{metrics.box.map50:.4f}")
print("mAP75:", f"{metrics.box.map75:.4f}")
print("mAP50-95:", f"{metrics.box.map:.4f}")

print("\nResults:", OUTPUT)
print("Plots:", metrics.save_dir)

print("\nE06 standardized validation: PASS")
