"""
YOLO-World Remote-Sensing Object Detection
Experiment E08: Standardized Full-Data Validation

Compare E04, E05, E06, E07 and E08.
Validation set only. No training or test evaluation.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import torch
import yaml
from ultralytics import YOLOWorld


ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = ROOT / "runs/full/e08_full/weights/best.pt"
DATA_YAML = ROOT / "configs/e08_full_hybrid.yaml"

RESULT_DIR = ROOT / "results/full"
OUTPUT = RESULT_DIR / "e08_validation_metrics.json"

PREVIOUS_FILES = {
    "E04": ROOT / "results/zero_shot/e04_hybrid_metrics.json",
    "E05": ROOT / "results/fewshot/e05_validation_metrics.json",
    "E06": ROOT / "results/fewshot/e06_validation_metrics.json",
    "E07": ROOT / "results/fewshot/e07_validation_metrics.json",
}

IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
NMS_IOU = 0.70


# --------------------------------------------------
# 1. Verify prerequisites
# --------------------------------------------------

for path in [WEIGHTS, DATA_YAML, *PREVIOUS_FILES.values()]:
    if not path.exists():
        raise FileNotFoundError(path)

assert torch.cuda.is_available(), "CUDA GPU unavailable"

with DATA_YAML.open() as f:
    config = yaml.safe_load(f)


def ordered_names(names):
    if isinstance(names, dict):
        indexed = {int(k): v for k, v in names.items()}
        return [indexed[i] for i in range(len(indexed))]
    return list(names)


classes = ordered_names(config["names"])

assert len(classes) == 10
assert classes[0] == "aircraft"

data_root = Path(config["path"]).resolve()

val_path = Path(config["val"])
if not val_path.is_absolute():
    val_path = data_root / val_path

expected_val = (
    ROOT / "data/processed/nwpu_yolo/images/val"
).resolve()

assert val_path.resolve() == expected_val, (
    "Unexpected validation directory"
)

images = [
    p for p in expected_val.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

assert len(images) == 120

previous = {
    name: json.loads(path.read_text())
    for name, path in PREVIOUS_FILES.items()
}

for exp in ("E05", "E06", "E07"):
    assert previous[exp]["class_names"] == classes


# --------------------------------------------------
# 2. Print experiment configuration
# --------------------------------------------------

print("\n=== E08 STANDARDIZED VALIDATION ===")

print("Checkpoint:", WEIGHTS)
print("Dataset:", DATA_YAML)
print("GPU:", torch.cuda.get_device_name(0))
print("Training images:", 560)
print("Training objects:", 2848)
print("Validation images:", len(images))
print("Validation objects:", 520)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Classes:", classes)
print("Training: NO")
print("Test set used: NO")


# --------------------------------------------------
# 3. Load E08 checkpoint
# --------------------------------------------------

model = YOLOWorld(str(WEIGHTS))

validator_name = (
    model.task_map["detect"]["validator"].__name__
)

assert validator_name == "WorldValidator"


# --------------------------------------------------
# 4. Run standardized evaluation
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
    name="e08_standardized",
    exist_ok=False,
    verbose=True,
)


# --------------------------------------------------
# 5. Collect results
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
# 6. Compare E04 through E08
# --------------------------------------------------

experiments = {
    name: result["metrics"]
    for name, result in previous.items()
}

experiments["E08"] = values

print("\n=== E04 vs E05 vs E06 vs E07 vs E08 ===")

print(
    f"{'Metric':<14}"
    + "".join(
        f"{name:>10}"
        for name in ("E04", "E05", "E06", "E07", "E08")
    )
)

for key in values:
    print(
        f"{key:<14}"
        + "".join(
            f"{experiments[exp][key]:>10.4f}"
            for exp in ("E04", "E05", "E06", "E07", "E08")
        )
    )


# --------------------------------------------------
# 7. Compare per-class E07 vs E08
# --------------------------------------------------

print("\n=== PER-CLASS E07 vs E08 ===")

print(
    f"{'Class':<24}"
    f"{'E07':>10}"
    f"{'E08':>10}"
    f"{'Change':>12}"
)

class_changes = {}

for name in classes:
    old = float(
        previous["E07"]["per_class_map50_95"][name]
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
# 8. Save reproducible results
# --------------------------------------------------

sha256 = hashlib.sha256()

with WEIGHTS.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha256.update(chunk)

result = {
    "experiment": "E08",
    "evaluation": "standardized_validation",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "model": "YOLOv8s-WorldV2",
    "checkpoint": str(WEIGHTS),
    "checkpoint_sha256": sha256.hexdigest(),
    "dataset": "NWPU VHR-10",
    "split": "val",
    "training_images": 560,
    "training_objects": 2848,
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
    "per_class_change_vs_e07": class_changes,
    "validation_run": str(metrics.save_dir),
    "test_set_used": False,
}

RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print("\n=== E08 VALIDATION COMPLETED ===")

for name, value in values.items():
    print(f"{name}: {value:.4f}")

print("\nResults:", OUTPUT)
print("Plots:", metrics.save_dir)
print("Test set used: NO")

print("\nE08 standardized validation: PASS")
