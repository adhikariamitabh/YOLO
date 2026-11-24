"""
YOLO-World Remote-Sensing Object Detection
Course: CS 7267

Experiment E08: Full-Dataset Fine-Tuning

Training:
    560 images
    2848 annotated objects
    455 positive images
    105 negative images

Validation:
    120 images
    520 annotated objects

Test:
    Not used

Starts from original pretrained YOLOv8s-WorldV2.
Uses E04 hybrid class names.
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

FULL_DATA = (
    ROOT / "data/processed/nwpu_yolo/data.yaml"
)

HYBRID_YAML = (
    ROOT / "configs/e04_hybrid_prompts.yaml"
)

TRAINING_YAML = (
    ROOT / "configs/e08_full_hybrid.yaml"
)

OUTPUT_DIR = ROOT / "results/full"

RUNS_DIR = ROOT / "runs/full"

RUN_NAME = "e08_full"

RUN_DIR = RUNS_DIR / RUN_NAME

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Verify prerequisites
# --------------------------------------------------

for path in (WEIGHTS, FULL_DATA, HYBRID_YAML):
    if not path.exists():
        raise FileNotFoundError(path)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU unavailable")

if RUN_DIR.exists():
    raise RuntimeError(
        f"E08 directory already exists: {RUN_DIR}\n"
        "Review the existing run before restarting."
    )


# --------------------------------------------------
# 3. Read dataset configurations
# --------------------------------------------------

with FULL_DATA.open() as f:
    full_config = yaml.safe_load(f)

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
    full_config["names"]
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
# 4. Verify dataset paths
# --------------------------------------------------

data_root = Path(full_config["path"]).resolve()


def split_path(value):
    path = Path(value)

    if not path.is_absolute():
        path = data_root / path

    return path.resolve()


image_dir = split_path(full_config["train"])

val_dir = split_path(full_config["val"])

test_dir = split_path(full_config["test"])

label_dir = data_root / "labels/train"

assert image_dir == (
    data_root / "images/train"
).resolve()

assert val_dir == (
    data_root / "images/val"
).resolve()

assert test_dir == (
    data_root / "images/test"
).resolve()


# --------------------------------------------------
# 5. Verify images and labels
# --------------------------------------------------

def list_images(directory):
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    )


train_images = list_images(image_dir)
val_images = list_images(val_dir)
test_images = list_images(test_dir)

labels = sorted(label_dir.glob("*.txt"))

assert len(train_images) == 560, (
    f"Expected 560 images; found {len(train_images)}"
)

assert len(val_images) == 120
assert len(test_images) == 120

assert len(labels) == 560, (
    f"Expected 560 label files; found {len(labels)}"
)

assert {p.stem for p in train_images} == {
    p.stem for p in labels
}, "Training image/label mismatch"

positive_images = [
    p for p in train_images
    if p.name.startswith("positive_")
]

negative_images = [
    p for p in train_images
    if p.name.startswith("negative_")
]

assert len(positive_images) == 455
assert len(negative_images) == 105


# --------------------------------------------------
# 6. Verify all bounding boxes
# --------------------------------------------------

box_counts = Counter()

empty_labels = 0

for label_file in labels:

    lines = [
        line.strip()
        for line in label_file.read_text().splitlines()
        if line.strip()
    ]

    if not lines:
        empty_labels += 1

    for line in lines:

        fields = line.split()

        assert len(fields) == 5, (
            f"Invalid annotation: {label_file}"
        )

        class_id = int(fields[0])

        assert 0 <= class_id < 10

        x, y, w, h = map(float, fields[1:])

        assert 0 <= x <= 1
        assert 0 <= y <= 1
        assert 0 < w <= 1
        assert 0 < h <= 1

        assert x - w / 2 >= -1e-5
        assert y - h / 2 >= -1e-5
        assert x + w / 2 <= 1 + 1e-5
        assert y + h / 2 <= 1 + 1e-5

        box_counts[class_id] += 1


total_boxes = sum(box_counts.values())

assert total_boxes == 2848, (
    f"Expected 2848 boxes; found {total_boxes}"
)

assert empty_labels == 105, (
    f"Expected 105 empty labels; found {empty_labels}"
)

assert all(
    box_counts[i] > 0
    for i in range(10)
)


# --------------------------------------------------
# 7. Create E08 dataset YAML
# --------------------------------------------------

training_config = full_config.copy()

training_config["names"] = {
    i: name
    for i, name in enumerate(hybrid_names)
}

training_config["nc"] = 10

TRAINING_YAML.write_text(
    yaml.safe_dump(
        training_config,
        sort_keys=False
    )
)


# --------------------------------------------------
# 8. Hash original pretrained checkpoint
# --------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


original_hash = sha256_file(WEIGHTS)


# --------------------------------------------------
# 9. Print experiment information
# --------------------------------------------------

print("\n=== E08 FULL-DATA TRAINING ===")

print("Model:", WEIGHTS.name)

print("Training images:", len(train_images))
print("Positive images:", len(positive_images))
print("Negative images:", len(negative_images))

print("Training bounding boxes:", total_boxes)

print("Validation images:", len(val_images))
print("Test images:", len(test_images))

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)

print("\n=== OBJECTS PER CLASS ===")

for i, name in enumerate(hybrid_names):
    print(
        f"{i}: {name:<22} "
        f"{box_counts[i]}"
    )

print("\n=== TRAINING CONFIGURATION ===")

print("Epochs: 50")
print("Batch size: 4")
print("Image size: 640")
print("Optimizer: AdamW")
print("Learning rate: 0.0005")
print("Seed: 42")
print("Training YAML:", TRAINING_YAML)

print("\nStarting from original pretrained weights.")
print("E05, E06 and E07 weights will not be used.")
print("Test set will not be evaluated.")


# --------------------------------------------------
# 10. Load original YOLO-World model
# --------------------------------------------------

model = YOLOWorld(str(WEIGHTS))

trainer_name = (
    model.task_map["detect"]["trainer"].__name__
)

assert trainer_name == "WorldTrainer"

print("\nTrainer:", trainer_name)


# --------------------------------------------------
# 11. Train E08
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

    project=str(RUNS_DIR),
    name=RUN_NAME,
    exist_ok=False,
    verbose=True,
)


# --------------------------------------------------
# 12. Verify saved model checkpoints
# --------------------------------------------------

actual_run_dir = Path(model.trainer.save_dir)

assert actual_run_dir.resolve() == RUN_DIR.resolve()

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
# 13. Verify original weights unchanged
# --------------------------------------------------

assert sha256_file(WEIGHTS) == original_hash, (
    "Original pretrained checkpoint was modified"
)

print("\nOriginal checkpoint unchanged: VERIFIED")


# --------------------------------------------------
# 14. Save experiment metadata
# --------------------------------------------------

metadata = {
    "experiment": "E08",
    "description": "Full-data YOLO-World fine-tuning",

    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "model": "YOLOv8s-WorldV2",

    "initial_checkpoint": str(WEIGHTS),
    "initial_checkpoint_sha256": original_hash,

    "training_images": 560,
    "positive_images": 455,
    "negative_images": 105,
    "training_objects": 2848,

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
    OUTPUT_DIR / "e08_training_metadata.json"
)

metadata_file.write_text(
    json.dumps(metadata, indent=2) + "\n"
)


# --------------------------------------------------
# 15. Final summary
# --------------------------------------------------

print("\n=== E08 TRAINING COMPLETED ===")

print("Training images:", len(train_images))
print("Annotated objects:", total_boxes)
print("Epochs:", 50)

print("\nBest checkpoint:", best_weights)
print("Last checkpoint:", last_weights)

print("\nTraining output:", actual_run_dir)
print("Metadata:", metadata_file)

print("\nOriginal checkpoint unchanged.")
print("Test partition untouched.")

print("\nNext: standardized E08 validation.")
