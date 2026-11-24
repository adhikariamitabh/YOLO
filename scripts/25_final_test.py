"""
CS 7267 — YOLO-World Remote-Sensing Project

Final held-out test evaluation: E04–E08.

Models were trained and selected using validation data.
This script evaluates each predefined model on the
same 120-image / 528-object test partition.

No training.
No checkpoint selection based on test performance.
No test images excluded.
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

OUTPUT = ROOT / "results/final_test"
RUNS = ROOT / "runs/final_test"

SUMMARY_FILE = OUTPUT / "e04_e08_test_comparison.json"

TEST_IMAGES = (
    ROOT / "data/processed/nwpu_yolo/images/test"
)

EXPECTED_CLASSES = [
    "aircraft",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "vehicle",
]

EXPERIMENTS = [
    {
        "id": "E04",
        "label": "Zero-shot hybrid",
        "weights": ROOT / "models/yolov8s-worldv2.pt",
        "data": ROOT / "configs/e04_hybrid_prompts.yaml",
        "training_images": 0,
    },
    {
        "id": "E05",
        "label": "One-shot",
        "weights": ROOT / "runs/fewshot/e05_oneshot/weights/best.pt",
        "data": ROOT / "configs/e05_oneshot_hybrid.yaml",
        "training_images": 10,
    },
    {
        "id": "E06",
        "label": "Five-shot",
        "weights": ROOT / "runs/fewshot/e06_fiveshot/weights/best.pt",
        "data": ROOT / "configs/e06_fiveshot_hybrid.yaml",
        "training_images": 50,
    },
    {
        "id": "E07",
        "label": "Ten-shot",
        "weights": ROOT / "runs/fewshot/e07_tenshot/weights/best.pt",
        "data": ROOT / "configs/e07_tenshot_hybrid.yaml",
        "training_images": 95,
    },
    {
        "id": "E08",
        "label": "Full dataset",
        "weights": ROOT / "runs/full/e08_full/weights/best.pt",
        "data": ROOT / "configs/e08_full_hybrid.yaml",
        "training_images": 560,
    },
]

EVAL = {
    "split": "test",
    "device": 0,
    "imgsz": 640,
    "batch": 8,
    "workers": 2,
    "conf": 0.001,
    "iou": 0.70,
    "max_det": 300,
}

# Two unconfirmed similarity candidates from the
# cross-split audit. We will report this limitation.
AUDIT_NOTE = {
    "potential_train_test_pairs": 2,
    "confirmed_duplicates": None,
    "visual_review_completed": False,
    "test_images_excluded": 0,
    "note": (
        "Two train-test image pairs were flagged by "
        "perceptual hashing but not visually adjudicated. "
        "Results are reported on the complete original "
        "test partition with this limitation disclosed."
    ),
}


# --------------------------------------------------
# 2. Verify prerequisites
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU unavailable")

if SUMMARY_FILE.exists():
    raise RuntimeError(
        "Final test summary already exists. "
        "Do not overwrite test results."
    )

if not TEST_IMAGES.is_dir():
    raise FileNotFoundError(TEST_IMAGES)

images = sorted(
    p for p in TEST_IMAGES.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

assert len(images) == 120, (
    f"Expected 120 test images, found {len(images)}"
)

for exp in EXPERIMENTS:
    for key in ("weights", "data"):
        if not exp[key].is_file():
            raise FileNotFoundError(exp[key])

    run_dir = RUNS / exp["id"].lower()
    if run_dir.exists():
        raise RuntimeError(
            f"Test run directory already exists: {run_dir}"
        )


# --------------------------------------------------
# 3. Verify test dataset configuration
# --------------------------------------------------

def ordered_names(names):
    if isinstance(names, dict):
        mapping = {int(k): v for k, v in names.items()}
        return [mapping[i] for i in range(len(mapping))]
    return list(names)


for exp in EXPERIMENTS:
    with exp["data"].open() as f:
        config = yaml.safe_load(f)

    classes = ordered_names(config["names"])

    assert classes == EXPECTED_CLASSES, (
        f"{exp['id']}: class-name mismatch"
    )

    data_root = Path(config["path"])

    test_path = Path(config["test"])
    if not test_path.is_absolute():
        test_path = data_root / test_path

    assert test_path.resolve() == TEST_IMAGES.resolve(), (
        f"{exp['id']}: unexpected test path"
    )


# --------------------------------------------------
# 4. Print experiment protocol
# --------------------------------------------------

print("\n=== FINAL HELD-OUT TEST EVALUATION ===")

print("GPU:", torch.cuda.get_device_name(0))
print("Ultralytics:", ultralytics.__version__)
print("Dataset: NWPU VHR-10")
print("Split: test")
print("Test images:", len(images))
print("Ground-truth objects: 528")
print("Model count:", len(EXPERIMENTS))
print("Image size:", EVAL["imgsz"])
print("Batch size:", EVAL["batch"])
print("Confidence:", EVAL["conf"])
print("NMS IoU:", EVAL["iou"])
print("Training: NO")
print("Test images excluded: 0")

print(
    "\nDataset limitation: "
    "two unresolved train-test similarity candidates."
)

print("\nModels:")

for exp in EXPERIMENTS:
    print(
        f"  {exp['id']}: {exp['label']}, "
        f"{exp['training_images']} training images"
    )


# --------------------------------------------------
# 5. Evaluate all predefined models
# --------------------------------------------------

OUTPUT.mkdir(parents=True, exist_ok=True)

all_results = {}

for exp in EXPERIMENTS:

    experiment_id = exp["id"]

    print("\n" + "=" * 55)
    print(f"FINAL TEST: {experiment_id}")
    print("=" * 55)

    print("Model:", exp["weights"])
    print("Dataset YAML:", exp["data"])

    model = YOLOWorld(str(exp["weights"]))

    validator_name = (
        model.task_map["detect"]["validator"].__name__
    )

    assert validator_name == "WorldValidator", (
        f"Unexpected validator: {validator_name}"
    )

    # The pretrained zero-shot model needs the
    # predefined E04 hybrid prompt vocabulary.
    # WorldValidator also reads classes from YAML.
    if experiment_id == "E04":
        model.set_classes(EXPECTED_CLASSES)

    metrics = model.val(
        data=str(exp["data"]),
        split="test",
        device=0,
        imgsz=640,
        batch=8,
        workers=2,
        conf=0.001,
        iou=0.70,
        max_det=300,
        plots=True,
        save_txt=False,
        project=str(RUNS),
        name=experiment_id.lower(),
        exist_ok=False,
        verbose=True,
    )

    values = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
    }

    class_ap = list(metrics.box.maps)

    assert len(class_ap) == 10, (
        f"{experiment_id}: unexpected AP count"
    )

    per_class = {
        name: float(class_ap[i])
        for i, name in enumerate(EXPECTED_CLASSES)
    }

    record = {
        "experiment": experiment_id,
        "description": exp["label"],
        "checkpoint": str(exp["weights"]),
        "dataset_yaml": str(exp["data"]),
        "training_images": exp["training_images"],
        "test_images": 120,
        "test_objects": 528,
        "metrics": values,
        "per_class_map50_95": per_class,
        "run_directory": str(metrics.save_dir),
    }

    all_results[experiment_id] = record

    individual_file = (
        OUTPUT / f"{experiment_id.lower()}_test_metrics.json"
    )

    individual_file.write_text(
        json.dumps(record, indent=2) + "\n"
    )

    print(
        f"\n{experiment_id} TEST mAP50-95: "
        f"{values['map50_95']:.4f}"
    )

    print("Saved:", individual_file)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------
# 6. Print comparison
# --------------------------------------------------

print("\n=== FINAL TEST COMPARISON ===")

print(
    f"{'Metric':<14}"
    + "".join(
        f"{exp['id']:>10}"
        for exp in EXPERIMENTS
    )
)

metric_names = [
    "precision",
    "recall",
    "map50",
    "map75",
    "map50_95",
]

for metric_name in metric_names:

    row = f"{metric_name:<14}"

    for exp in EXPERIMENTS:
        value = all_results[
            exp["id"]
        ]["metrics"][metric_name]

        row += f"{value:>10.4f}"

    print(row)


# --------------------------------------------------
# 7. Print class-level comparison
# --------------------------------------------------

print("\n=== PER-CLASS TEST mAP50-95 ===")

print(
    f"{'Class':<24}"
    + "".join(
        f"{exp['id']:>10}"
        for exp in EXPERIMENTS
    )
)

for class_name in EXPECTED_CLASSES:

    row = f"{class_name:<24}"

    for exp in EXPERIMENTS:

        value = all_results[
            exp["id"]
        ]["per_class_map50_95"][class_name]

        row += f"{value:>10.4f}"

    print(row)


# --------------------------------------------------
# 8. Save final comparison
# --------------------------------------------------

summary = {
    "project": (
        "Adapting YOLO-World for Remote-Sensing "
        "Object Detection"
    ),
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": "NWPU VHR-10",
    "split": "test",
    "test_images": 120,
    "test_objects": 528,
    "evaluation": EVAL,
    "classes": EXPECTED_CLASSES,
    "dataset_similarity_audit": AUDIT_NOTE,
    "experiments": all_results,
    "no_training_performed": True,
    "no_checkpoint_selection_using_test": True,
}

SUMMARY_FILE.write_text(
    json.dumps(summary, indent=2) + "\n"
)


# --------------------------------------------------
# 9. Completion
# --------------------------------------------------

print("\n=== FINAL TEST EVALUATION COMPLETE ===")

print("Models evaluated:", len(all_results))
print("Training performed: NO")
print("Test images excluded: 0")

print("\nResults:", SUMMARY_FILE)
print("Plots:", RUNS)

print(
    "\nDo not change models or training settings "
    "in response to these test results."
)
