"""
YOLO-World Remote-Sensing Detection
Experiment E07: Standardized Ten-Shot Validation

Compares E04, E05, E06 and E07.
Uses only the NWPU validation partition.

No training or test-set evaluation.
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
    ROOT / "runs/fewshot/e07_tenshot/weights/best.pt"
)

DATA_YAML = ROOT / "configs/e07_tenshot_hybrid.yaml"

RESULT_DIR = ROOT / "results/fewshot"

OUTPUT = RESULT_DIR / "e07_validation_metrics.json"

EXPERIMENT_FILES = {
    "E04": ROOT / "results/zero_shot/e04_hybrid_metrics.json",
    "E05": RESULT_DIR / "e05_validation_metrics.json",
    "E06": RESULT_DIR / "e06_validation_metrics.json",
}

IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
NMS_IOU = 0.70


# --------------------------------------------------
# 2. Verify prerequisites
# --------------------------------------------------

for path in [WEIGHTS, DATA_YAML, *EXPERIMENT_FILES.values()]:
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

# Ensure that we are evaluating the validation split.
val_path = Path(config["val"])

if not val_path.is_absolute():
    val_path = Path(config["path"]) / val_path

val_path = val_path.resolve()

expected_val = (
    ROOT / "data/processed/nwpu_yolo/images/val"
).resolve()

assert val_path == expected_val, (
    "Unexpected validation directory"
)

images = [
    p for p in val_path.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

assert len(images) == 120

previous = {
    exp: json.loads(path.read_text())
    for exp, path in EXPERIMENT_FILES.items()
}

assert previous["E05"]["class_names"] == classes
assert previous["E06"]["class_names"] == classes


# --------------------------------------------------
# 3. Print experiment details
# --------------------------------------------------

print("\n=== E07 STANDARDIZED VALIDATION ===")

print("Checkpoint:", WEIGHTS)
print("Dataset:", DATA_YAML)
print("GPU:", torch.cuda.get_device_name(0))
print("Training images:", 95)
print("Training objects:", 750)
print("Validation images:", len(images))
print("Validation objects:", 520)
print("Classes:", classes)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Training: NO")
print("Test set used: NO")


# --------------------------------------------------
# 4. Load the E07 trained checkpoint
# --------------------------------------------------

model = YOLOWorld(str(WEIGHTS))

validator = model.task_map["detect"]["validator"].__name__

assert validator == "WorldValidator", (
    f"Unexpected validator: {validator}"
)


# --------------------------------------------------
# 5. Run standardized validation
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
    project=str(ROOT / "runs/validation"),
    name="e07_standardized",
    exist_ok=True,
    verbose=True,
)


# --------------------------------------------------
# 6. Collect metrics
# --------------------------------------------------

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


# --------------------------------------------------
# 7. Overall comparison
# --------------------------------------------------

experiments = {
    exp: data["metrics"]
    for exp, data in previous.items()
}

experiments["E07"] = values

print("\n=== E04 vs E05 vs E06 vs E07 ===")

print(
    f"{'Metric':<16}"
    f"{'E04':>10}"
    f"{'E05':>10}"
    f"{'E06':>10}"
    f"{'E07':>10}"
)

for key in values:
    print(
        f"{key:<16}"
        + "".join(
            f"{experiments[e][key]:>10.4f}"
            for e in ("E04", "E05", "E06", "E07")
        )
    )


# --------------------------------------------------
# 8. Class-level E06 versus E07 comparison
# --------------------------------------------------

print("\n=== PER-CLASS E06 vs E07 ===")

print(
    f"{'Class':<24}"
    f"{'E06':>10}"
    f"{'E07':>10}"
    f"{'Change':>12}"
)

class_changes = {}

for name in classes:
    old = float(
        previous["E06"]["per_class_map50_95"][name]
    )
    new = per_class[name]
    change = new - old
    class_changes[name] = change

    print(
        f"{name:<24}"
        f"{old:>10.4f}"
        f"{new:>10.4f}"
        f"{change:>+12.4f}"
    )


# --------------------------------------------------
# 9. Save experiment results
# --------------------------------------------------

result = {
    "experiment": "E07",
    "evaluation": "standardized_validation",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "model": "YOLOv8s-WorldV2",
    "checkpoint": str(WEIGHTS),
    "dataset": "NWPU VHR-10",
    "split": "val",
    "training_images": 95,
    "training_objects": 750,
    "validation_images": 120,
    "validation_objects": 520,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "confidence_threshold": CONFIDENCE,
    "nms_iou_threshold": NMS_IOU,
    "class_names": classes,
    "metrics": values,
    "per_class_map50_95": per_class,
    "comparison": experiments,
    "per_class_change_vs_e06": class_changes,
    "test_set_used": False,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")


# --------------------------------------------------
# 10. Final summary
# --------------------------------------------------

print("\n=== E07 VALIDATION COMPLETED ===")

for name, value in values.items():
    print(f"{name}: {value:.4f}")

print("\nResults:", OUTPUT)
print("Plots:", metrics.save_dir)
print("Test set used: NO")

print("\nE07 standardized validation: PASS")
