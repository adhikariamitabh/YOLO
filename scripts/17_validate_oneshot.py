"""
E05: Standardized validation of the one-shot model.

Evaluates the saved best.pt on the unchanged
NWPU validation partition.

Uses the same evaluation settings as E04.
Does not train the model or use the test set.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import torch
import yaml
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = (
    ROOT / "runs/fewshot/e05_oneshot/weights/best.pt"
)

DATA_YAML = ROOT / "configs/e05_oneshot_hybrid.yaml"

E04_METRICS = (
    ROOT / "results/zero_shot/e04_hybrid_metrics.json"
)

OUTPUT = ROOT / "results/fewshot/e05_validation_metrics.json"

for path in (WEIGHTS, DATA_YAML, E04_METRICS):
    if not path.exists():
        raise FileNotFoundError(path)

assert torch.cuda.is_available(), "CUDA GPU unavailable"

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

print("\n=== E05 STANDARDIZED VALIDATION ===")
print("Checkpoint:", WEIGHTS)
print("Validation dataset:", DATA_YAML)
print("GPU:", torch.cuda.get_device_name(0))
print("Classes:", classes)
print("Training: NO")
print("Test set used: NO")

model = YOLOWorld(str(WEIGHTS))

validator = model.task_map["detect"]["validator"].__name__

assert validator == "WorldValidator", (
    f"Unexpected validator: {validator}"
)

metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    device=0,
    imgsz=640,
    batch=8,
    workers=2,
    conf=0.001,
    iou=0.70,
    max_det=300,
    plots=True,
    save_txt=True,
    project=str(ROOT / "runs/validation"),
    name="e05_standardized",
    exist_ok=True,
    verbose=True,
)

values = {
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "map50": float(metrics.box.map50),
    "map75": float(metrics.box.map75),
    "map50_95": float(metrics.box.map),
}

per_class = {
    name: float(ap)
    for name, ap in zip(classes, metrics.box.maps)
}

baseline = json.loads(E04_METRICS.read_text())["metrics"]

print("\n=== E04 vs E05 VALIDATION ===")

print(
    f"{'Metric':<16}"
    f"{'E04':>12}"
    f"{'E05':>12}"
    f"{'Change':>12}"
)

for key, new in values.items():
    old = float(baseline[key])

    print(
        f"{key:<16}"
        f"{old:>12.4f}"
        f"{new:>12.4f}"
        f"{new-old:>+12.4f}"
    )

print("\n=== E05 PER-CLASS mAP@0.50:0.95 ===")

for name, ap in per_class.items():
    print(f"{name:<24} {ap:.4f}")

result = {
    "experiment": "E05",
    "evaluation": "standardized_validation",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "checkpoint": str(WEIGHTS),
    "dataset": "NWPU VHR-10",
    "split": "val",
    "training_images": 10,
    "training_objects": 50,
    "validation_images": 120,
    "validation_objects": 520,
    "image_size": 640,
    "batch_size": 8,
    "confidence_threshold": 0.001,
    "nms_iou_threshold": 0.70,
    "class_names": classes,
    "metrics": values,
    "per_class_map50_95": per_class,
    "zero_shot_e04_metrics": baseline,
    "test_set_used": False,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    json.dumps(result, indent=2) + "\n"
)

print("\nResults saved:", OUTPUT)
print("Validation plots:", metrics.save_dir)
print("\nE05 STANDARDIZED VALIDATION: COMPLETE")
