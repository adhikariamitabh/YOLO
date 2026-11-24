"""
Project: YOLO-World Remote-Sensing Detection
Course: CS 7267

Experiment E05: Image-based one-shot fine-tuning.

Training: 10 images, 50 annotated objects.
Validation: 120 images.
Test: not used.

Uses E04 hybrid class names.
Starts from original pretrained YOLO-World weights.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import torch
import yaml
from ultralytics import YOLOWorld


ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = ROOT / "models/yolov8s-worldv2.pt"

FEWSHOT_YAML = (
    ROOT /
    "data/processed/nwpu_fewshot/shot_01/data.yaml"
)

HYBRID_YAML = (
    ROOT / "configs/e04_hybrid_prompts.yaml"
)

TRAINING_YAML = (
    ROOT / "configs/e05_oneshot_hybrid.yaml"
)

OUTPUT = ROOT / "results/fewshot"
RUNS = ROOT / "runs/fewshot"

OUTPUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Verify prerequisites
# --------------------------------------------------

for path in (WEIGHTS, FEWSHOT_YAML, HYBRID_YAML):
    if not path.exists():
        raise FileNotFoundError(path)

assert torch.cuda.is_available(), "CUDA unavailable"

with FEWSHOT_YAML.open() as f:
    fewshot_config = yaml.safe_load(f)

with HYBRID_YAML.open() as f:
    hybrid_config = yaml.safe_load(f)

shot_root = Path(fewshot_config["path"])

image_dir = shot_root / fewshot_config["train"]
label_dir = shot_root / "labels/train"

images = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

labels = sorted(label_dir.glob("*.txt"))

assert len(images) == 10, (
    f"Expected 10 training images, found {len(images)}"
)

assert len(labels) == 10, (
    f"Expected 10 label files, found {len(labels)}"
)

assert {p.stem for p in images} == {
    p.stem for p in labels
}

boxes = sum(
    len([
        line
        for line in p.read_text().splitlines()
        if line.strip()
    ])
    for p in labels
)

assert boxes == 50, (
    f"Expected 50 annotations, found {boxes}"
)

# --------------------------------------------------
# 2. Create E05 dataset configuration
# --------------------------------------------------

training_config = fewshot_config.copy()

# Use the same prompt vocabulary as E04.
training_config["names"] = hybrid_config["names"]

original_names = fewshot_config["names"]

def ordered_names(names):
    if isinstance(names, dict):
        indexed = {int(k): v for k, v in names.items()}
        return [indexed[i] for i in range(len(indexed))]
    return list(names)

base_names = ordered_names(original_names)
train_names = ordered_names(training_config["names"])

assert len(train_names) == 10
assert base_names[0] == "airplane"
assert train_names[0] == "aircraft"
assert train_names[1:] == base_names[1:]

TRAINING_YAML.write_text(
    yaml.safe_dump(
        training_config,
        sort_keys=False
    )
)

# --------------------------------------------------
# 3. Display experiment details
# --------------------------------------------------

print("\n=== E05 ONE-SHOT TRAINING ===")

print("Model:", WEIGHTS.name)
print("Training images:", len(images))
print("Training objects:", boxes)
print("Validation images:", 120)
print("Classes:", train_names)
print("GPU:", torch.cuda.get_device_name(0))
print("Training YAML:", TRAINING_YAML)
print("Epochs:", 50)
print("Batch size:", 4)
print("Image size:", 640)
print("Optimizer: AdamW")
print("Learning rate: 0.0005")

print("\nOriginal pretrained weights will not be overwritten.")

# --------------------------------------------------
# 4. Load pretrained model
# --------------------------------------------------

model = YOLOWorld(str(WEIGHTS))

assert (
    model.task_map["detect"]["trainer"].__name__
    == "WorldTrainer"
), "Unexpected training interface"

# Do not preload a previous E05 checkpoint.
# This experiment starts from the original weights.

# --------------------------------------------------
# 5. Fine-tune the model
# --------------------------------------------------

model.train(
    data=str(TRAINING_YAML),

    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,

    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=2.0,

    seed=42,
    deterministic=True,

    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    fliplr=0.5,
    flipud=0.5,
    translate=0.1,
    scale=0.2,

    amp=False,

    patience=0,
    val=True,
    split="val",
    conf=0.001,
    iou=0.70,
    max_det=300,

    plots=True,
    save=True,
    compile=False,

    project=str(RUNS),
    name="e05_oneshot",
    exist_ok=False,
    verbose=True,
)

# --------------------------------------------------
# 6. Verify saved model
# --------------------------------------------------

run_dir = Path(model.trainer.save_dir)

best_weights = run_dir / "weights/best.pt"
last_weights = run_dir / "weights/last.pt"

if not best_weights.exists():
    raise FileNotFoundError(
        f"Training finished but best.pt is missing: "
        f"{best_weights}"
    )

if not last_weights.exists():
    raise FileNotFoundError(
        f"Training finished but last.pt is missing: "
        f"{last_weights}"
    )

# --------------------------------------------------
# 7. Save experiment metadata
# --------------------------------------------------

metadata = {
    "experiment": "E05",
    "description": "Image-based one-shot fine-tuning",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "model": WEIGHTS.name,
    "training_images": 10,
    "training_objects": 50,
    "validation_images": 120,

    "epochs": 50,
    "batch": 4,
    "image_size": 640,

    "optimizer": "AdamW",
    "learning_rate": 0.0005,
    "seed": 42,

    "prompt_configuration": "E04 hybrid",
    "training_yaml": str(TRAINING_YAML),

    "run_directory": str(run_dir),
    "best_weights": str(best_weights),
    "last_weights": str(last_weights),

    "test_set_used": False,
}

metadata_file = OUTPUT / "e05_training_metadata.json"

metadata_file.write_text(
    json.dumps(metadata, indent=2) + "\n"
)

# --------------------------------------------------
# 8. Print training summary
# --------------------------------------------------

print("\n=== E05 TRAINING COMPLETED ===")

print("Training images:", 10)
print("Annotated objects:", 50)
print("Epochs:", 50)
print("Training output:", run_dir)
print("Best checkpoint:", best_weights)
print("Last checkpoint:", last_weights)
print("Metadata:", metadata_file)

print("\nNext: standardized validation of best.pt")
