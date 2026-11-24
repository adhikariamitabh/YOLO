"""
CS 7267 — YOLO-World
Figure 3: E08 Detection Error Distribution

Reads completed error-analysis results.
No inference or model changes.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT /
    "results/report/e08_error_analysis/e08_error_summary.json"
)

OUTPUT = (
    ROOT /
    "results/report/fig03_e08_detection_errors.png"
)

assert INPUT.exists()

data = json.loads(INPUT.read_text())

assert data["experiment"] == "E08"
assert data["split"] == "test"
assert data["images"] == 120

rows = data["classes"]

assert len(rows) == 10

assert sum(r["tp"] for r in rows) == 487
assert sum(r["fp"] for r in rows) == 58
assert sum(r["fn"] for r in rows) == 41

# Sort categories by total diagnostic errors.
rows = sorted(
    rows,
    key=lambda r: r["fp"] + r["fn"],
    reverse=True
)

names = [r["class"] for r in rows]
fp = [r["fp"] for r in rows]
fn = [r["fn"] for r in rows]

y = np.arange(len(names))
height = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

bars_fp = ax.barh(
    y - height / 2,
    fp,
    height,
    label="Unmatched predictions (FP)"
)

bars_fn = ax.barh(
    y + height / 2,
    fn,
    height,
    label="Missed objects (FN)"
)

ax.set_yticks(y, names)
ax.invert_yaxis()

ax.set_xlabel("Number of objects")
ax.set_title(
    "E08 — Detection Errors by Object Class",
    fontweight="bold"
)

ax.bar_label(bars_fp, padding=3, fontsize=8)
ax.bar_label(bars_fn, padding=3, fontsize=8)

ax.set_xlim(0, max(fp + fn) + 7)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

ax.legend(loc="lower right")

fig.text(
    0.5,
    0.01,
    "Diagnostic confidence threshold = 0.25; "
    "matching IoU = 0.50. Not official AP metrics.",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\n=== FIGURE 3 COMPLETED ===")

print("Classes:", len(names))
print("Total unmatched predictions:", sum(fp))
print("Total missed objects:", sum(fn))
print("Output:", OUTPUT)

print("\nTraining: NO")
print("Inference: NO")
print("Model changed: NO")

print("\nERROR FIGURE GENERATION: COMPLETE")
