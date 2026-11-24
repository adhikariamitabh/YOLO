"""
Project: YOLO-World Remote-Sensing Object Detection
Course: CS 7267

Experiment E06: Five-Shot Fine-Tuning

Training:
    50 images
    357 annotated objects

Validation:
    120 images
    520 annotated objects

Initial weights:
    Original pretrained YOLOv8s-WorldV2

Prompt configuration:
    E04 hybrid

No test-set evaluation.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
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

WEIGHTS = ROOT / "models/yolov8s-worldv2.pt"

FEWSHOT_YAML = (
    ROOT /
    "data/processed/nwpu_fewshot/shot_05/data.yaml"
)

HYBRID_YAML = (
    ROOT / "configs/e04_hybrid_prompts.yaml"
)

TRAINING_YAML = (
    ROOT / "configs/e06_fiveshot_hybrid.yaml"
)

OUTPUT = ROOT / "results/fewshot"

RUNS = ROOT / "runs/fewshot"

RUN_NAME = "e06_fiveshot"

RUN_DIR = RUNS / RUN_NAME

OUTPUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Verify prerequisites
# --------------------------------------------------

for path in [WEIGHTS, FEWSHOT_YAML, HYBRID_YAML]:
    if not path.exists():
        raise FileNotFoundError(path)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU unavailable")

# Prevent accidental reuse of an earlier run.
if RUN_DIR.exists():
    raise RuntimeError(
        f"E06 output directory already exists:\n"
        f"{RUN_DIR}\n"
        "Review it before starting another run."
    )


# --------------------------------------------------
# 3. Read dataset configurations
# --------------------------------------------------

with FEWSHOT_YAML.open() as f:
    fewshot_config = yaml.safe_load(f)

with HYBRID_YAML.open() as f:
    hybrid_config = yaml.safe_load(f)


def ordered_names(names):
    if isinstance(names, dict):
        indexed = {
            int(k): v for k, v in names.items()
        }
        return [
            indexed[i]
            for i in range(len(indexed))
        ]

    return list(names)


original_names = ordered_names(
    fewshot_config["names"]
)

hybrid_names = ordered_names(
    hybrid_config["names"]
)

assert len(original_names) == 10
assert len(hybrid_names) == 10

assert original_names[0] == "airplane"
assert hybrid_names[0] == "aircraft"

assert hybrid_names[1:] == original_names[1:]


# --------------------------------------------------
# 4. Verify five-shot training dataset
# --------------------------------------------------

shot_root = Path(fewshot_config["path"])

image_dir = shot_root / fewshot_config["train"]

label_dir = shot_root / "labels/train"

images = sorted(
    p
    for p in image_dir.iterdir()
    if p.suffix.lower() in {
        ".jpg", ".jpeg", ".png"
    }
)

labels = sorted(label_dir.glob("*.txt"))

assert len(images) == 50, (
    f"Expected 50 images, found {len(images)}"
)

assert len(labels) == 50, (
    f"Expected 50 labels, found {len(labels)}"
)

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}

assert image_stems == label_stems, (
    "Image and label filenames do not match"
)

box_counts = Counter()

for label_file in labels:

    for line in label_file.read_text().splitlines():

        if not line.strip():
            continue

        values = line.split()

        assert len(values) == 5, (
            f"Invalid label: {label_file}"
        )

        class_id = int(values[0])

        assert 0 <= class_id <= 9

        coordinates = [
            float(v) for v in values[1:]
        ]

        assert all(
            0.0 <= v <= 1.0
            for v in coordinates
        ), f"Invalid coordinates: {label_file}"

        assert coordinates[2] > 0
        assert coordinates[3] > 0

        box_counts[class_id] += 1


total_boxes = sum(box_counts.values())

assert total_boxes == 357, (
    f"Expected 357 boxes, found {total_boxes}"
)

assert all(
    box_counts[i] > 0 for i in range(10)
), "At least one class has no training objects"


# --------------------------------------------------
# 5. Prepare E06 dataset YAML
# --------------------------------------------------

training_config = fewshot_config.copy()

training_config["names"] = {
    i: name
    for i, name in enumerate(hybrid_names)
}

TRAINING_YAML.write_text(
    yaml.safe_dump(
        training_config,
        sort_keys=False
    )
)


# --------------------------------------------------
# 6. Calculate original checkpoint hash
# --------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


original_hash = sha256_file(WEIGHTS)


# --------------------------------------------------
# 7. Print experiment configuration
# --------------------------------------------------

print("\n=== E06 FIVE-SHOT TRAINING ===")

print("Model:", WEIGHTS.name)
print("Training images:", len(images))
print("Training objects:", total_boxes)
print("Validation images:", 120)
print("Validation objects:", 520)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)

print("\n=== CLASSES ===")

for class_id, name in enumerate(hybrid_names):
    print(
        f"{class_id}: {name:<22}"
        f"training boxes: {box_counts[class_id]}"
    )

print("\n=== TRAINING SETTINGS ===")

print("Epochs: 50")
print("Batch size: 4")
print("Image size: 640")
print("Optimizer: AdamW")
print("Learning rate: 0.0005")
print("Seed: 42")
print("Training YAML:", TRAINING_YAML)

print("\nOriginal pretrained weights will not be overwritten.")


# --------------------------------------------------
# 8. Load ORIGINAL pretrained model
# --------------------------------------------------

model = YOLOWorld(str(WEIGHTS))

trainer_name = (
    model.task_map["detect"]["trainer"].__name__
)

assert trainer_name == "WorldTrainer", (
    f"Unexpected trainer: {trainer_name}"
)

print("\nTrainer:", trainer_name)

print("Starting from original pretrained checkpoint.")
print("Not loading E05 weights.")


# --------------------------------------------------
# 9. Train E06
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
    name=RUN_NAME,
    exist_ok=False,
    verbose=True,
)


# --------------------------------------------------
# 10. Verify output checkpoints
# --------------------------------------------------

actual_run_dir = Path(model.trainer.save_dir)

assert actual_run_dir.resolve() == RUN_DIR.resolve(), (
    "Unexpected run directory"
)

best_weights = (
    actual_run_dir / "weights/best.pt"
)

last_weights = (
    actual_run_dir / "weights/last.pt"
)

if not best_weights.exists():
    raise FileNotFoundError(
        f"Missing best checkpoint: {best_weights}"
    )

if not last_weights.exists():
    raise FileNotFoundError(
        f"Missing last checkpoint: {last_weights}"
    )


# --------------------------------------------------
# 11. Verify original weights unchanged
# --------------------------------------------------

final_hash = sha256_file(WEIGHTS)

assert final_hash == original_hash, (
    "Original checkpoint was unexpectedly modified"
)

print("\nOriginal checkpoint verified unchanged.")


# --------------------------------------------------
# 12. Save experiment metadata
# --------------------------------------------------

metadata = {
    "experiment": "E06",
    "description": "Five-shot image-based fine-tuning",

    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "initial_checkpoint": str(WEIGHTS),
    "initial_checkpoint_sha256": original_hash,

    "model": "YOLOv8s-WorldV2",

    "training_images": 50,
    "training_objects": 357,

    "validation_images": 120,
    "validation_objects": 520,

    "epochs": 50,
    "batch_size": 4,
    "image_size": 640,

    "optimizer": "AdamW",
    "learning_rate": 0.0005,

    "seed": 42,

    "prompt_configuration": "E04 hybrid",
    "classes": hybrid_names,

    "objects_per_class": {
        hybrid_names[i]: box_counts[i]
        for i in range(10)
    },

    "training_yaml": str(TRAINING_YAML),

    "run_directory": str(actual_run_dir),

    "best_weights": str(best_weights),
    "last_weights": str(last_weights),

    "test_set_used": False,
}

metadata_file = (
    OUTPUT / "e06_training_metadata.json"
)

metadata_file.write_text(
    json.dumps(metadata, indent=2) + "\n"
)


# --------------------------------------------------
# 13. Final training summary
# --------------------------------------------------

print("\n=== E06 TRAINING COMPLETED ===")

print("Training images:", len(images))
print("Annotated objects:", total_boxes)
print("Epochs:", 50)

print("\nBest checkpoint:", best_weights)
print("Last checkpoint:", last_weights)

print("\nTraining output:", actual_run_dir)
print("Metadata:", metadata_file)

print("\nTest set remains untouched.")

print("\nNext: standardized validation of E06 best.pt")
