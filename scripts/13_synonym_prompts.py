"""
Project: YOLO-World Remote-Sensing Object Detection
Course: CS 7267
Experiment: E03 - Synonym and Object-Specific Prompting

Compare E01, E02 and E03 on NWPU VHR-10.

No training or fine-tuning.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import torch
import ultralytics
import yaml
from ultralytics import YOLOWorld


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models/yolov8s-worldv2.pt"

DATA_YAML = (
    ROOT / "data/processed/nwpu_yolo/data.yaml"
)

CONFIG_DIR = ROOT / "configs"
RESULT_DIR = ROOT / "results/zero_shot"
RUN_DIR = ROOT / "runs"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ID = "E03"

IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
NMS_IOU = 0.70


# --------------------------------------------------
# 2. Load original dataset configuration
# --------------------------------------------------

assert MODEL_PATH.exists(), "Model weights missing"
assert DATA_YAML.exists(), "Dataset YAML missing"
assert torch.cuda.is_available(), "CUDA GPU unavailable"

with DATA_YAML.open() as f:
    original_config = yaml.safe_load(f)

names = original_config["names"]

if isinstance(names, dict):
    indexed = {int(k): v for k, v in names.items()}
    class_names = [indexed[i] for i in range(len(indexed))]
else:
    class_names = list(names)

assert len(class_names) == 10


# --------------------------------------------------
# 3. Define E03 alternative prompts
# --------------------------------------------------

prompts = [
    "aircraft",
    "vessel",
    "industrial storage tank",
    "baseball infield",
    "tennis court",
    "basketball court",
    "running track",
    "port",
    "bridge",
    "motor vehicle",
]

assert len(prompts) == len(class_names)

print("\n=== E03 ALTERNATIVE PROMPTS ===")

for i, (name, prompt) in enumerate(
    zip(class_names, prompts)
):
    print(f"{i}: {name} -> {prompt}")


# --------------------------------------------------
# 4. Create experiment-specific dataset YAML
# --------------------------------------------------

experiment_config = original_config.copy()

experiment_config["names"] = {
    i: prompt
    for i, prompt in enumerate(prompts)
}

experiment_yaml = (
    CONFIG_DIR / "e03_synonym_prompts.yaml"
)

experiment_yaml.write_text(
    yaml.safe_dump(
        experiment_config,
        sort_keys=False
    )
)

print("\nExperiment YAML:", experiment_yaml)


# --------------------------------------------------
# 5. Load pretrained YOLO-World
# --------------------------------------------------

print("\n=== ENVIRONMENT ===")

print("Model:", MODEL_PATH.name)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)

model = YOLOWorld(str(MODEL_PATH))

validator = model.task_map["detect"]["validator"].__name__

print("Validator:", validator)

if validator != "WorldValidator":
    raise RuntimeError(
        "YOLO-World validator was not selected."
    )

model.set_classes(prompts)

print("Training: NO")
print("Model weights: Unchanged")


# --------------------------------------------------
# 6. Evaluate 120 validation images
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
    project=str(RUN_DIR),
    name="e03_synonym_prompts",
    exist_ok=True,
    verbose=True,
)


# --------------------------------------------------
# 7. Collect metrics
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
    for name, ap in zip(class_names, metrics.box.maps)
}

results = {
    "experiment": EXPERIMENT_ID,
    "description": "Alternative object-name prompts",
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
    "prompts": prompts,

    "metrics": metric_values,
    "per_class_map50_95": per_class,
}


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

output_file = (
    RESULT_DIR / "e03_synonym_metrics.json"
)

output_file.write_text(
    json.dumps(results, indent=2) + "\n"
)


# --------------------------------------------------
# 9. Compare E01, E02 and E03
# --------------------------------------------------

experiment_files = {
    "E01": RESULT_DIR / "e01_baseline_metrics.json",
    "E02": RESULT_DIR / "e02_domain_metrics.json",
    "E03": output_file,
}

experiments = {}

for exp_id, path in experiment_files.items():
    if path.exists():
        experiments[exp_id] = json.loads(
            path.read_text()
        )["metrics"]
    else:
        print(f"WARNING: Missing {exp_id}: {path}")


print("\n=== E01 vs E02 vs E03 COMPARISON ===")

print(
    f"{'Metric':<16}"
    f"{'E01':>12}"
    f"{'E02':>12}"
    f"{'E03':>12}"
)

for key in metric_values:
    values = []

    for exp_id in ("E01", "E02", "E03"):
        if exp_id in experiments:
            values.append(
                f"{experiments[exp_id][key]:.4f}"
            )
        else:
            values.append("N/A")

    print(
        f"{key:<16}"
        f"{values[0]:>12}"
        f"{values[1]:>12}"
        f"{values[2]:>12}"
    )


# --------------------------------------------------
# 10. Per-class results
# --------------------------------------------------

print("\n=== E03 PER-CLASS mAP@0.50:0.95 ===")

for name, ap in per_class.items():
    print(f"{name}: {ap:.4f}")


# --------------------------------------------------
# 11. Save comparison JSON
# --------------------------------------------------

comparison_file = (
    RESULT_DIR / "e01_e02_e03_comparison.json"
)

comparison_file.write_text(
    json.dumps(experiments, indent=2) + "\n"
)

print("\n=== OUTPUT FILES ===")
print("Metrics:", output_file)
print("Comparison:", comparison_file)
print("Plots:", metrics.save_dir)

print("\nE03 experiment completed successfully.")
