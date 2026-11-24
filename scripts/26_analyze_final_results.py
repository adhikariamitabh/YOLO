"""
CS 7267 — YOLO-World Remote-Sensing Project

Final results analysis and report figures.

Reads completed validation and test metrics.
Does not train, validate, or modify models.
"""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "results/report"
OUTPUT.mkdir(parents=True, exist_ok=True)

TEST_FILE = (
    ROOT / "results/final_test/e04_e08_test_comparison.json"
)

VAL_FILES = {
    "E04": ROOT / "results/zero_shot/e04_hybrid_metrics.json",
    "E05": ROOT / "results/fewshot/e05_validation_metrics.json",
    "E06": ROOT / "results/fewshot/e06_validation_metrics.json",
    "E07": ROOT / "results/fewshot/e07_validation_metrics.json",
    "E08": ROOT / "results/full/e08_validation_metrics.json",
}

EXPERIMENTS = ["E04", "E05", "E06", "E07", "E08"]

TRAIN_IMAGES = {
    "E04": 0,
    "E05": 10,
    "E06": 50,
    "E07": 95,
    "E08": 560,
}

for file in [TEST_FILE, *VAL_FILES.values()]:
    if not file.is_file():
        raise FileNotFoundError(file)


# --------------------------------------------------
# 1. Load results
# --------------------------------------------------

test_data = json.loads(TEST_FILE.read_text())

validation = {
    name: json.loads(path.read_text())
    for name, path in VAL_FILES.items()
}

assert test_data["split"] == "test"
assert test_data["test_images"] == 120
assert test_data["test_objects"] == 528

test = test_data["experiments"]

METRICS = [
    "precision",
    "recall",
    "map50",
    "map75",
    "map50_95",
]

for name in EXPERIMENTS:
    assert name in test

    for metric in METRICS:
        assert metric in validation[name]["metrics"]
        assert metric in test[name]["metrics"]


# --------------------------------------------------
# 2. Save overall comparison CSV
# --------------------------------------------------

comparison_csv = OUTPUT / "overall_comparison.csv"

with comparison_csv.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "experiment",
        "training_images",
        "metric",
        "validation",
        "test",
        "test_minus_validation",
    ])

    for name in EXPERIMENTS:
        for metric in METRICS:
            val = validation[name]["metrics"][metric]
            tst = test[name]["metrics"][metric]

            writer.writerow([
                name,
                TRAIN_IMAGES[name],
                metric,
                round(val, 6),
                round(tst, 6),
                round(tst - val, 6),
            ])


# --------------------------------------------------
# 3. Figure 1 — Learning curve
# --------------------------------------------------

val_map = [
    validation[name]["metrics"]["map50_95"] * 100
    for name in EXPERIMENTS
]

test_map = [
    test[name]["metrics"]["map50_95"] * 100
    for name in EXPERIMENTS
]

x = np.arange(len(EXPERIMENTS))

labels = [
    f"{name}\n({TRAIN_IMAGES[name]} images)"
    for name in EXPERIMENTS
]

plt.figure(figsize=(9, 5))

plt.plot(
    x, val_map,
    marker="o",
    linewidth=2,
    label="Validation"
)

plt.plot(
    x, test_map,
    marker="s",
    linewidth=2,
    label="Test"
)

plt.xticks(x, labels)
plt.ylabel("mAP@0.50:0.95 (%)")
plt.xlabel("Experiment / training images")
plt.title("YOLO-World: Accuracy Across Training Set Sizes")
plt.ylim(0, 75)
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()

learning_file = OUTPUT / "fig01_learning_curve.png"
plt.savefig(learning_file, dpi=300)
plt.close()


# --------------------------------------------------
# 4. Figure 2 — E08 per-class generalization
# --------------------------------------------------

classes = test_data["classes"]

e08_val = validation["E08"]["per_class_map50_95"]
e08_test = test["E08"]["per_class_map50_95"]

val_scores = [
    e08_val[name] * 100 for name in classes
]

test_scores = [
    e08_test[name] * 100 for name in classes
]

positions = np.arange(len(classes))
height = 0.36

plt.figure(figsize=(10, 7))

plt.barh(
    positions - height / 2,
    val_scores,
    height,
    label="Validation"
)

plt.barh(
    positions + height / 2,
    test_scores,
    height,
    label="Test"
)

plt.yticks(positions, classes)
plt.gca().invert_yaxis()
plt.xlim(0, 100)
plt.xlabel("mAP@0.50:0.95 (%)")
plt.title("E08: Per-Class Validation vs Test")
plt.legend()
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()

class_figure = OUTPUT / "fig02_e08_per_class.png"
plt.savefig(class_figure, dpi=300)
plt.close()


# --------------------------------------------------
# 5. Save E08 class-level comparison
# --------------------------------------------------

class_csv = OUTPUT / "e08_per_class_comparison.csv"

with class_csv.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "class",
        "validation_map50_95",
        "test_map50_95",
        "test_minus_validation",
    ])

    for name in classes:
        val = e08_val[name]
        tst = e08_test[name]

        writer.writerow([
            name,
            round(val, 6),
            round(tst, 6),
            round(tst - val, 6),
        ])


# --------------------------------------------------
# 6. Print report-ready results
# --------------------------------------------------

print("\n=== FINAL TEST RESULTS ANALYSIS ===")

print(
    f"{'Experiment':<12}"
    f"{'Images':>8}"
    f"{'Val mAP':>12}"
    f"{'Test mAP':>12}"
    f"{'Gap':>12}"
)

for name in EXPERIMENTS:
    val = validation[name]["metrics"]["map50_95"]
    tst = test[name]["metrics"]["map50_95"]

    print(
        f"{name:<12}"
        f"{TRAIN_IMAGES[name]:>8}"
        f"{val:>12.4f}"
        f"{tst:>12.4f}"
        f"{tst-val:>+12.4f}"
    )


print("\n=== E08 CLASS GENERALIZATION ===")

class_gaps = sorted(
    [
        (
            name,
            e08_test[name] - e08_val[name]
        )
        for name in classes
    ],
    key=lambda item: item[1]
)

for name, gap in class_gaps:
    print(f"{name:<24} {gap:+.4f}")


# --------------------------------------------------
# 7. Final checks
# --------------------------------------------------

assert len(class_gaps) == 10

assert abs(
    test["E08"]["metrics"]["map50_95"] - 0.6260
) < 0.0001

print("\n=== GENERATED FILES ===")

for file in [
    comparison_csv,
    learning_file,
    class_figure,
    class_csv,
]:
    print(file)

print("\nTraining performed: NO")
print("Inference performed: NO")
print("Test metrics recomputed: NO")

print("\nFINAL RESULTS ANALYSIS: COMPLETE")
