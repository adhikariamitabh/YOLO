
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
ANNOTATIONS = BASE / "data/coco2017/annotations"
SPLITS_FILE = BASE / "splits/coco2017_seed42.json"
OUTPUT = BASE / "splits/coco2017_class_distribution.csv"

MIN_IMAGES = 10  # Flag classes with fewer than 10 images

# --------------------------------------------------
# Load saved dataset splits
# --------------------------------------------------

with open(SPLITS_FILE) as f:
    splits = json.load(f)

selected = {
    name: {int(img["id"]) for img in splits[name]}
    for name in ("train", "val", "test")
}

assert selected["train"].isdisjoint(selected["val"])
assert selected["train"].isdisjoint(selected["test"])
assert selected["val"].isdisjoint(selected["test"])

print("Dataset split sizes:")
for name, ids in selected.items():
    print(f"  {name}: {len(ids)} images")

# --------------------------------------------------
# Initialize class statistics
# --------------------------------------------------

box_counts = {
    name: Counter() for name in selected
}

image_counts = {
    name: defaultdict(set) for name in selected
}

annotated_images = {
    name: set() for name in selected
}

categories = {}

# --------------------------------------------------
# Process annotations
# --------------------------------------------------

def analyze_annotations(filename, split_names):
    global categories

    path = ANNOTATIONS / filename
    print(f"\nReading {filename} ...")

    with open(path) as f:
        coco = json.load(f)

    current_categories = {
        cat["id"]: cat["name"]
        for cat in coco["categories"]
    }

    if categories and categories != current_categories:
        raise ValueError("COCO category definitions differ")

    categories = current_categories

    # Map each selected image ID to its split
    image_to_split = {
        img_id: split_name
        for split_name in split_names
        for img_id in selected[split_name]
    }

    # Verify selected images exist in the annotations
    available_ids = {
        img["id"] for img in coco["images"]
    }

    missing = set(image_to_split) - available_ids

    if missing:
        raise ValueError(
            f"{len(missing)} selected images are missing"
        )

    # Count valid, non-crowd bounding boxes
    for ann in coco["annotations"]:
        img_id = ann["image_id"]

        if img_id not in image_to_split:
            continue

        if ann.get("iscrowd", 0) == 1:
            continue

        bbox = ann.get("bbox", [])

        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        split_name = image_to_split[img_id]
        class_id = ann["category_id"]

        box_counts[split_name][class_id] += 1
        image_counts[split_name][class_id].add(img_id)
        annotated_images[split_name].add(img_id)

    print("Annotation analysis complete")


analyze_annotations(
    "instances_train2017.json",
    ["train"]
)

analyze_annotations(
    "instances_val2017.json",
    ["val", "test"]
)

# --------------------------------------------------
# Save class distribution to CSV
# --------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Class ID",
        "Class Name",
        "Train Boxes",
        "Train Images",
        "Val Boxes",
        "Val Images",
        "Test Boxes",
        "Test Images",
    ])

    for class_id, class_name in sorted(categories.items()):
        writer.writerow([
            class_id,
            class_name,
            box_counts["train"][class_id],
            len(image_counts["train"][class_id]),
            box_counts["val"][class_id],
            len(image_counts["val"][class_id]),
            box_counts["test"][class_id],
            len(image_counts["test"][class_id]),
        ])

# --------------------------------------------------
# Print dataset summary
# --------------------------------------------------

print("\nCOCO 2017 SUBSET ANALYSIS")
print("-" * 50)
print("Total COCO classes:", len(categories))

for split_name, ids in selected.items():

    represented = sum(
        len(image_counts[split_name][class_id]) > 0
        for class_id in categories
    )

    total_boxes = sum(box_counts[split_name].values())

    images_without_boxes = (
        len(ids) - len(annotated_images[split_name])
    )

    print(f"\n{split_name.upper()}")
    print("Images:", len(ids))
    print("Classes represented:", represented)
    print("Total bounding boxes:", total_boxes)
    print("Images without eligible boxes:", images_without_boxes)

    rare_classes = [
        categories[class_id]
        for class_id in categories
        if len(image_counts[split_name][class_id]) < MIN_IMAGES
    ]

    print(
        f"Classes with fewer than {MIN_IMAGES} images:",
        len(rare_classes)
    )

    if rare_classes:
        print("Classes:", ", ".join(rare_classes))

print("\nDistribution saved to:")
print(OUTPUT)

