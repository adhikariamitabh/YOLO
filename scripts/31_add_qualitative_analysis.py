"""
CS 7267 — YOLO-World

Add reviewed qualitative observations to the
Results and Discussion draft.

Preserves the original draft.
No training or inference.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

REPORT = ROOT / "results/report"

SOURCE = REPORT / "RESULTS_DISCUSSION_DRAFT.md"

OUTPUT = REPORT / "RESULTS_DISCUSSION_REVIEWED.md"

GALLERY = REPORT / "e08_error_analysis/gallery"

assert SOURCE.is_file(), SOURCE

if OUTPUT.exists():
    raise FileExistsError(
        f"Reviewed draft already exists: {OUTPUT}"
    )

examples = {
    "storage": "positive_324",
    "harbor": "positive_282",
    "sports": "positive_333",
}

paths = {}

for key, stem in examples.items():

    matches = sorted(
        GALLERY.glob(f"*_{stem}.png")
    )

    assert len(matches) == 1, (
        f"Expected one gallery image for {stem}; "
        f"found {len(matches)}"
    )

    paths[key] = matches[0].relative_to(REPORT)


text = SOURCE.read_text()

# Renumber the existing final sections.
for old, new in [(8, 9), (7, 8), (6, 7)]:
    text = re.sub(
        rf"(?m)^## {old}\. ",
        f"## {new}. ",
        text
    )

marker = "## 7. Discussion"

assert text.count(marker) == 1


section = f"""## 6. Qualitative Detection Review

A qualitative review examined 12 selected test images
containing ground-truth annotations and E08 predictions.
These images were selected to inspect detection errors
rather than to represent a random sample.

Predictions were analyzed using confidence >= 0.25
and class-aware matching at IoU >= 0.50.
Green boxes indicate matched objects, while red
boxes indicate unmatched predictions or annotations.

### 6.1 Storage Tanks

In `positive_324.png`, densely grouped circular
storage tanks include both correctly matched objects
and unmatched annotations or predictions.

This observation is consistent with the diagnostic
error counts: 19 missed storage tanks and 22
unmatched storage-tank predictions across the
full test partition.

![Storage-tank detection example]({paths['storage'].as_posix()})

*Figure 4. Example of storage-tank detections.
Green boxes indicate matches; red boxes indicate
unmatched annotations or predictions.*

### 6.2 Harbors and Docking Structures

The marina scenes show numerous correctly identified
regions alongside unmatched boxes near closely spaced
boats and docking structures.

These examples suggest that the spatial arrangement
and visual complexity of harbor scenes may contribute
to localization and detection errors.

![Harbor detection example]({paths['harbor'].as_posix()})

*Figure 5. Example of harbor-region detections
in a visually complex marina scene.*

### 6.3 Sports Facilities

The model correctly matched several annotated sports
facilities, including tennis courts and baseball diamonds.
However, some annotated regions were missed, and other
predictions did not match ground-truth objects.

![Sports facility detection example]({paths['sports'].as_posix()})

*Figure 6. Example of sports-facility detections
and unmatched annotations.*

### 6.4 Interpretation

The examples demonstrate both successful detection
and remaining difficulties in complex aerial scenes.

The visual inspection is consistent with the overall
quantitative results, but it does not independently
establish the causes of individual errors.

Because these 12 images were selected for diagnostic
inspection, they should not be treated as a representative
sample of all 120 test images.

"""

text = text.replace(
    marker,
    section + marker,
    1
)

OUTPUT.write_text(text)

print("\n=== QUALITATIVE ANALYSIS ADDED ===")

print("Source:", SOURCE)
print("Reviewed draft:", OUTPUT)
print("Examples added:", len(paths))

for key, path in paths.items():
    print(f"{key}: {path}")

print("Report sections: 9")
print("Original draft preserved: YES")
print("Training performed: NO")
print("Test metrics modified: NO")

print("\nQUALITATIVE REPORT UPDATE: COMPLETE")
