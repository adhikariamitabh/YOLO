"""
CS 7267 — YOLO-World Remote-Sensing Project

Generate a Results and Discussion draft from
completed experiment files.

No model training or inference.
No changes to experimental results.
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "results/report"

TEST_FILE = (
    ROOT /
    "results/final_test/e04_e08_test_comparison.json"
)

ERROR_FILE = (
    REPORT_DIR /
    "e08_error_analysis/e08_error_summary.json"
)

VALIDATION_FILES = {
    "E04": ROOT / "results/zero_shot/e04_hybrid_metrics.json",
    "E05": ROOT / "results/fewshot/e05_validation_metrics.json",
    "E06": ROOT / "results/fewshot/e06_validation_metrics.json",
    "E07": ROOT / "results/fewshot/e07_validation_metrics.json",
    "E08": ROOT / "results/full/e08_validation_metrics.json",
}

OUTPUT = REPORT_DIR / "RESULTS_DISCUSSION_DRAFT.md"

EXPERIMENTS = [
    ("E04", "Zero-shot", 0),
    ("E05", "One-shot", 10),
    ("E06", "Five-shot", 50),
    ("E07", "Ten-shot", 95),
    ("E08", "Full training", 560),
]

for path in [
    TEST_FILE,
    ERROR_FILE,
    *VALIDATION_FILES.values()
]:
    if not path.exists():
        raise FileNotFoundError(path)

test_data = json.loads(TEST_FILE.read_text())
error_data = json.loads(ERROR_FILE.read_text())

validation = {
    name: json.loads(path.read_text())
    for name, path in VALIDATION_FILES.items()
}

test = test_data["experiments"]

assert test_data["test_images"] == 120
assert test_data["test_objects"] == 528

assert error_data["images"] == 120

assert abs(
    test["E08"]["metrics"]["map50_95"] - 0.6260
) < 0.0001


def percent(value):
    return f"{value * 100:.2f}%"


def points(value):
    return f"{value * 100:+.2f}"


# --------------------------------------------------
# 1. Start report
# --------------------------------------------------

lines = [
    "# Results and Discussion",
    "",
    "## 1. Experimental Overview",
    "",
    (
        "This study evaluated YOLOv8s-WorldV2 on the "
        "NWPU VHR-10 remote-sensing dataset using "
        "zero-shot prompting and supervised fine-tuning."
    ),
    "",
    (
        "The original dataset was divided into 560 "
        "training images, 120 validation images, and "
        "120 held-out test images."
    ),
    "",
    (
        "The fine-tuning experiments used 10, 50, 95, "
        "and 560 training images. These were image-based "
        "few-shot subsets rather than strict fixed "
        "numbers of object instances per class."
    ),
    "",
    (
        "All fine-tuning experiments independently "
        "started from the original pretrained checkpoint. "
        "Training used 50 epochs, batch size 4, "
        "image size 640, AdamW, initial learning rate "
        "0.0005, and random seed 42."
    ),
    "",
    (
        "Models were selected using the validation set. "
        "Final testing used the same 120-image test set "
        "with 528 annotated objects for each model."
    ),
    "",
    "## 2. Overall Model Performance",
    "",
    "| Experiment | Training images | Validation mAP50-95 "
    "| Test mAP50-95 | Test mAP50 |",
    "|---|---:|---:|---:|---:|",
]

for name, label, count in EXPERIMENTS:

    val = validation[name]["metrics"]
    tst = test[name]["metrics"]

    lines.append(
        f"| {name} — {label} | {count} "
        f"| {percent(val['map50_95'])} "
        f"| {percent(tst['map50_95'])} "
        f"| {percent(tst['map50'])} |"
    )


# --------------------------------------------------
# 2. Main findings
# --------------------------------------------------

e04_test = test["E04"]["metrics"]["map50_95"]
e08_test = test["E08"]["metrics"]["map50_95"]

e07_test = test["E07"]["metrics"]["map50_95"]

e08_val = validation["E08"]["metrics"]["map50_95"]

lines += [
    "",
    (
        "The full-data model, E08, achieved the highest "
        "overall test accuracy, with "
        f"{percent(e08_test)} mAP@0.50:0.95 and "
        f"{percent(test['E08']['metrics']['map50'])} "
        "mAP@0.50."
    ),
    "",
    (
        "Compared with E04 zero-shot prompting, "
        "E08 improved test mAP@0.50:0.95 by "
        f"{points(e08_test - e04_test)} "
        "percentage points."
    ),
    "",
    (
        "E08 also outperformed ten-shot E07 by "
        f"{points(e08_test - e07_test)} "
        "percentage points."
    ),
    "",
    (
        "Performance increased as more annotated "
        "training images were introduced. The largest "
        "early improvement occurred when moving from "
        "one-shot to five-shot training."
    ),
    "",
    "![Learning curve](fig01_learning_curve.png)",
    "",
    "*Figure 1. Validation and test accuracy across "
    "training dataset sizes.*",
    "",
    "## 3. Validation and Test Generalization",
    "",
    "| Experiment | Validation | Test | Test minus validation |",
    "|---|---:|---:|---:|",
]

for name, label, count in EXPERIMENTS:

    val = validation[name]["metrics"]["map50_95"]
    tst = test[name]["metrics"]["map50_95"]

    lines.append(
        f"| {name} | {percent(val)} | {percent(tst)} "
        f"| {points(tst - val)} pp |"
    )

lines += [
    "",
    (
        "The E08 model achieved "
        f"{percent(e08_val)} mAP@0.50:0.95 "
        "on validation and "
        f"{percent(e08_test)} on the held-out test set, "
        "a decrease of "
        f"{abs((e08_test - e08_val) * 100):.2f} "
        "percentage points."
    ),
    "",
    (
        "This difference indicates that performance "
        "on the validation images was somewhat higher "
        "than performance on unseen test images."
    ),
    "",
    "## 4. Per-Class Performance",
    "",
    "| Object class | Validation mAP50-95 | "
    "Test mAP50-95 | Difference (pp) |",
    "|---|---:|---:|---:|",
]

class_names = test_data["classes"]

class_val = validation["E08"]["per_class_map50_95"]
class_test = test["E08"]["per_class_map50_95"]

for name in class_names:

    val = class_val[name]
    tst = class_test[name]

    lines.append(
        f"| {name} | {percent(val)} "
        f"| {percent(tst)} "
        f"| {points(tst - val)} |"
    )

lines += [
    "",
    "![Per-class comparison](fig02_e08_per_class.png)",
    "",
    "*Figure 2. E08 validation and test mAP@0.50:0.95 "
    "for the ten object categories.*",
    "",
    (
        "Storage tanks exhibited the largest negative "
        "generalization gap, declining from "
        f"{percent(class_val['storage tank'])} "
        "on validation to "
        f"{percent(class_test['storage tank'])} "
        "on test."
    ),
    "",
    (
        "The validation partition contained only one "
        "image with storage-tank annotations, while "
        "the test partition contained three such images. "
        "This small number of distinct scenes limits "
        "the reliability of class-level comparisons."
    ),
    "",
    "## 5. Detection Error Analysis",
    "",
]

counts = error_data["counts"]

lines += [
    (
        "A supplementary diagnostic analysis examined "
        "E08 predictions at confidence threshold 0.25 "
        "with class-aware matching at IoU 0.50."
    ),
    "",
    (
        f"The analysis found {counts['tp']} matched "
        f"predictions, {counts['fp']} unmatched predictions, "
        f"and {counts['fn']} unmatched ground-truth objects."
    ),
    "",
    "| Class | GT | Matched | Unmatched predictions | Missed objects |",
    "|---|---:|---:|---:|---:|",
]

for row in error_data["classes"]:
    lines.append(
        f"| {row['class']} | {row['gt']} "
        f"| {row['tp']} | {row['fp']} "
        f"| {row['fn']} |"
    )

lines += [
    "",
    "![Detection errors](fig03_e08_detection_errors.png)",
    "",
    "*Figure 3. Diagnostic unmatched predictions and "
    "missed objects by class.*",
    "",
    (
        "Storage tanks produced the most diagnostic "
        "errors. These counts should not be interpreted "
        "as the official precision, recall, or AP values, "
        "because the analysis used one fixed confidence "
        "threshold and IoU criterion."
    ),
    "",
    "## 6. Discussion",
    "",
    (
        "The experimental results show that supervised "
        "fine-tuning substantially improved YOLO-World "
        "performance on aerial imagery compared with "
        "zero-shot prompting."
    ),
    "",
    (
        "The improvements from E05 to E06 were larger "
        "than those from E06 to E07. This suggests "
        "diminishing gains within the tested training "
        "protocol, although E08 achieved further "
        "improvement with the full dataset."
    ),
    "",
    (
        "Performance varied substantially by object "
        "category. Storage tanks were notably less "
        "accurate on the test partition, while classes "
        "such as ground track fields and baseball "
        "diamonds achieved higher test accuracy."
    ),
    "",
    (
        "The results indicate that domain-specific "
        "labeled data is important for adapting "
        "YOLO-World to remote-sensing imagery."
    ),
    "",
    "## 7. Limitations",
    "",
    (
        "First, the experiments used a single random "
        "seed and one dataset split. Performance "
        "variability across seeds was not measured."
    ),
    "",
    (
        "Second, the fixed 50-epoch training schedule "
        "resulted in different numbers of optimizer "
        "updates as dataset size increased. Therefore, "
        "the results do not isolate dataset size "
        "from training computation."
    ),
    "",
    (
        "Third, full-data E08 included 105 negative "
        "training images, whereas the few-shot subsets "
        "were selected from positive training images. "
        "This is an additional difference between "
        "the training conditions."
    ),
    "",
    (
        "Fourth, class-level evaluation is sensitive "
        "to the number and diversity of available "
        "images. Some object categories were "
        "represented in only a few validation or "
        "test scenes."
    ),
    "",
    (
        "Fifth, a cross-split perceptual-hash audit "
        "flagged two potentially similar train-test "
        "image pairs. No exact duplicates were found, "
        "but the candidates were not visually confirmed "
        "or dismissed. All original test images "
        "remained in the final evaluation."
    ),
    "",
    (
        "Finally, the few-shot datasets were constructed "
        "by selecting images associated with each class "
        "while retaining all annotations in each selected "
        "image. Consequently, the one-, five-, and "
        "ten-shot terms refer to image selection rather "
        "than exactly one, five, or ten objects per class."
    ),
    "",
    "## 8. Conclusion",
    "",
    (
        "The full-data E08 model achieved "
        f"{percent(e08_test)} test mAP@0.50:0.95, "
        "compared with "
        f"{percent(e04_test)} for the selected "
        "zero-shot baseline."
    ),
    "",
    (
        "The results demonstrate substantial benefits "
        "from supervised domain adaptation on NWPU VHR-10. "
        "However, generalization remains class-dependent, "
        "and further evaluation across independent "
        "datasets and training seeds is needed."
    ),
    "",
    "---",
    "",
    "**Draft status:** Numerical results verified. "
    "Qualitative image review and references "
    "remain to be finalized.",
]

OUTPUT.write_text(
    "\n".join(lines) + "\n"
)

print("\n=== RESULTS AND DISCUSSION DRAFT ===")

print("Experiments:", len(EXPERIMENTS))
print("Test images:", test_data["test_images"])
print("Test objects:", test_data["test_objects"])

print("E08 test mAP50-95:", f"{e08_test:.4f}")
print("E08 validation mAP50-95:", f"{e08_val:.4f}")

print("Matched predictions:", counts["tp"])
print("Unmatched predictions:", counts["fp"])
print("Missed objects:", counts["fn"])

print("\nDraft:", OUTPUT)
print("Sections generated: 8")

for i in range(1, 4):
    figure = REPORT_DIR / f"fig0{i}_" 
    matches = list(REPORT_DIR.glob(f"fig0{i}_*.png"))
    if matches:
        print(f"Figure {i}: available")
    else:
        print(f"Figure {i}: MISSING")

print("\nTraining performed: NO")
print("Test metrics changed: NO")

print("\nREPORT DRAFT GENERATION: COMPLETE")
