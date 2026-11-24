"""
Project: YOLO-World Remote-Sensing Object Detection
Course: CS 7267
Experiment: E02 - Domain-Aware Zero-Shot Prompting

Objective:
Compare aerial-view text prompts against the E01
class-name zero-shot baseline.

No training or fine-tuning is performed.
"""

from pathlib import Path
from datetime import datetime, timezone
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

CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "results/zero_shot"
RUNS_DIR = ROOT / "runs"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ID = "E02"

IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
NMS_IOU = 0.70


# --------------------------------------------------
# 2. Read original class definitions
# --------------------------------------------------

with DATA_YAML.open() as f:
    original_config = yaml.safe_load(f)

names = original_config["names"]

if isinstance(names, dict):
    indexed_names = {
        int(k): v for k, v in names.items()
    }
    class_names = [
        indexed_names[i]
        for i in range(len(indexed_names))
    ]
else:
    class_names = list(names)

assert len(class_names) == 10


# --------------------------------------------------
# 3. Define domain-aware prompts
# --------------------------------------------------

domain_prompts = [
    "airplane viewed from above",
    "ship viewed from above",
    "storage tank viewed from above",
    "baseball diamond viewed from above",
    "tennis court viewed from above",
    "basketball court viewed from above",
    "ground track field viewed from above",
    "harbor viewed from above",
    "bridge viewed from above",
    "vehicle viewed from above",
]

assert len(domain_prompts) == len(class_names)

print("\n=== E02 DOMAIN-AWARE PROMPTS ===")

for i, (original, prompt) in enumerate(
    zip(class_names, domain_prompts)
):
    print(f"{i}: {original} -> {prompt}")


# --------------------------------------------------
# 4. Create experiment-specific dataset YAML
# --------------------------------------------------

# Class IDs stay the same.
# Image paths and ground-truth labels stay the same.
# Only the text descriptions change.

experiment_config = original_config.copy()

experiment_config["names"] = {
    i: prompt
    for i, prompt in enumerate(domain_prompts)
}

experiment_yaml = (
    CONFIG_DIR / "e02_domain_prompts.yaml"
)

experiment_yaml.write_text(
    yaml.safe_dump(
        experiment_config,
        sort_keys=False
    )
)

print("\nExperiment YAML:", experiment_yaml)


# --------------------------------------------------
# 5. Verify environment
# --------------------------------------------------

assert MODEL_PATH.exists(), "Model weights missing"
assert DATA_YAML.exists(), "Dataset YAML missing"
assert torch.cuda.is_available(), "GPU unavailable"

print("\nModel:", MODEL_PATH.name)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Training: NO")


# --------------------------------------------------
# 6. Load pretrained YOLO-World
# --------------------------------------------------

model = YOLOWorld(str(MODEL_PATH))

# Ensure the installed Ultralytics version uses
# the YOLO-World-specific validation interface.

validator_name = (
    model.task_map["detect"]["validator"].__name__
)

print("Validator:", validator_name)

if validator_name != "WorldValidator":
    raise RuntimeError(
        "Expected WorldValidator. "
        "Check Ultralytics compatibility before "
        "comparing custom prompt metrics."
    )

model.set_classes(domain_prompts)

print("\nDomain-aware prompts configured.")


# --------------------------------------------------
# 7. Run zero-shot validation
# --------------------------------------------------

metrics = model.val(
    data=str(experiment_yaml),
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
    project=str(RUNS_DIR),
    name="e02_domain_aware",
    exist_ok=True,
    verbose=True,
)


# --------------------------------------------------
# 8. Collect metrics
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
        class_names, metrics.box.maps
    )
}

results = {
    "experiment": EXPERIMENT_ID,
    "description": "Domain-aware zero-shot prompting",
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
    "original_classes": class_names,
    "prompts": domain_prompts,
    "metrics": metric_values,
    "per_class_map50_95": per_class,
}


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

output_file = (
    OUTPUT_DIR / "e02_domain_metrics.json"
)

output_file.write_text(
    json.dumps(results, indent=2) + "\n"
)


# --------------------------------------------------
# 10. Compare E01 against E02
# --------------------------------------------------

baseline_file = (
    OUTPUT_DIR / "e01_baseline_metrics.json"
)

print("\n=== E02 RESULTS ===")

for name, value in metric_values.items():
    print(f"{name}: {value:.4f}")

if baseline_file.exists():

    baseline = json.loads(
        baseline_file.read_text()
    )["metrics"]

    print("\n=== E01 vs E02 COMPARISON ===")

    print(
        f"{'Metric':<15}"
        f"{'E01':>12}"
        f"{'E02':>12}"
        f"{'Change':>12}"
    )

    for key in metric_values:
        old = float(baseline[key])
        new = metric_values[key]
        change = new - old

        print(
            f"{key:<15}"
            f"{old:>12.4f}"
            f"{new:>12.4f}"
            f"{change:>+12.4f}"
        )

else:
    print("\nE01 metrics file not found.")


print("\n=== PER-CLASS mAP@0.50:0.95 ===")

for name, value in per_class.items():
    print(f"{name}: {value:.4f}")


print("\nResults saved:", output_file)
print("Validation plots:", metrics.save_dir)
print("\nE02 completed successfully.")