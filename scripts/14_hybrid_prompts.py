"""
Project: YOLO-World Remote-Sensing Object Detection
Course: CS 7267
Experiment: E04 - Hybrid Prompt Evaluation

Objective:
Test whether replacing "airplane" with "aircraft"
improves zero-shot performance while keeping
the remaining nine prompts unchanged.

Model weights are NOT updated.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import torch
import ultralytics
import yaml
from ultralytics import YOLOWorld


# ---------------------------------------------
# 1. Configuration
# ---------------------------------------------

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

EXPERIMENT_ID = "E04"

IMAGE_SIZE = 640
BATCH_SIZE = 8
CONFIDENCE = 0.001
NMS_IOU = 0.70


# ---------------------------------------------
# 2. Load original class names
# ---------------------------------------------

assert MODEL_PATH.exists(), "Model weights missing"
assert DATA_YAML.exists(), "Dataset YAML missing"
assert torch.cuda.is_available(), "CUDA unavailable"

with DATA_YAML.open() as f:
    original_config = yaml.safe_load(f)

names = original_config["names"]

if isinstance(names, dict):
    indexed = {
        int(k): v for k, v in names.items()
    }
    class_names = [
        indexed[i] for i in range(len(indexed))
    ]
else:
    class_names = list(names)

assert len(class_names) == 10
assert class_names[0] == "airplane"


# ---------------------------------------------
# 3. Define hybrid prompts
# ---------------------------------------------

prompts = class_names.copy()

# Change only airplane -> aircraft
prompts[0] = "aircraft"

changes = [
    (old, new)
    for old, new in zip(class_names, prompts)
    if old != new
]

assert changes == [("airplane", "aircraft")]

print("\n=== E04 HYBRID PROMPTS ===")

for i, (old, new) in enumerate(
    zip(class_names, prompts)
):
    print(f"{i}: {old} -> {new}")


# ---------------------------------------------
# 4. Create experiment-specific YAML
# ---------------------------------------------

experiment_config = original_config.copy()

experiment_config["names"] = {
    i: name
    for i, name in enumerate(prompts)
}

experiment_yaml = (
    CONFIG_DIR / "e04_hybrid_prompts.yaml"
)

experiment_yaml.write_text(
    yaml.safe_dump(
        experiment_config,
        sort_keys=False
    )
)

print("\nExperiment YAML:", experiment_yaml)


# ---------------------------------------------
# 5. Load pretrained model
# ---------------------------------------------

model = YOLOWorld(str(MODEL_PATH))

validator = (
    model.task_map["detect"]["validator"].__name__
)

assert validator == "WorldValidator"

model.set_classes(prompts)

print("\n=== ENVIRONMENT ===")
print("Model:", MODEL_PATH.name)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Validator:", validator)
print("Training: NO")


# ---------------------------------------------
# 6. Run zero-shot validation
# ---------------------------------------------

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
    name="e04_hybrid_prompts",
    exist_ok=True,
    verbose=True,
)


# ---------------------------------------------
# 7. Collect metrics
# ---------------------------------------------

metric_values = {
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "map50": float(metrics.box.map50),
    "map75": float(metrics.box.map75),
    "map50_95": float(metrics.box.map),
}

per_class = {
    class_names[i]: float(ap)
    for i, ap in enumerate(metrics.box.maps)
}

results = {
    "experiment": EXPERIMENT_ID,
    "description": "Hybrid zero-shot prompting",
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


# ---------------------------------------------
# 8. Save results
# ---------------------------------------------

output_file = (
    RESULT_DIR / "e04_hybrid_metrics.json"
)

output_file.write_text(
    json.dumps(results, indent=2) + "\n"
)


# ---------------------------------------------
# 9. Compare all four experiments
# ---------------------------------------------

experiment_files = {
    "E01": RESULT_DIR / "e01_baseline_metrics.json",
    "E02": RESULT_DIR / "e02_domain_metrics.json",
    "E03": RESULT_DIR / "e03_synonym_metrics.json",
    "E04": output_file,
}

experiments = {}

for exp_id, path in experiment_files.items():
    if not path.exists():
        raise FileNotFoundError(path)

    experiments[exp_id] = json.loads(
        path.read_text()
    )

print("\n=== E01 vs E02 vs E03 vs E04 ===")

print(
    f"{'Metric':<16}"
    f"{'E01':>10}"
    f"{'E02':>10}"
    f"{'E03':>10}"
    f"{'E04':>10}"
)

for key in metric_values:
    print(
        f"{key:<16}"
        + "".join(
            f"{experiments[e]['metrics'][key]:>10.4f}"
            for e in ("E01", "E02", "E03", "E04")
        )
    )


# ---------------------------------------------
# 10. Compare per-class performance
# ---------------------------------------------

print("\n=== PER-CLASS mAP@0.50:0.95 ===")

print(
    f"{'Class':<24}"
    f"{'E01':>10}"
    f"{'E03':>10}"
    f"{'E04':>10}"
)

for name in class_names:
    def class_ap(exp):
        return experiments[exp][
            "per_class_map50_95"
        ][name]

    print(
        f"{name:<24}"
        f"{class_ap('E01'):>10.4f}"
        f"{class_ap('E03'):>10.4f}"
        f"{class_ap('E04'):>10.4f}"
    )


# ---------------------------------------------
# 11. Save comparison
# ---------------------------------------------

comparison_file = (
    RESULT_DIR / "e01_e02_e03_e04_comparison.json"
)

comparison_file.write_text(
    json.dumps(experiments, indent=2) + "\n"
)

print("\n=== OUTPUT FILES ===")
print("E04 metrics:", output_file)
print("Comparison:", comparison_file)
print("Plots:", metrics.save_dir)

print("\nE04 experiment completed successfully.")
