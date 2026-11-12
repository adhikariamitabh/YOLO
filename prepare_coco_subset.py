import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
ANN = BASE / "data/coco2017/annotations"
OUT = BASE / "splits"

OUT.mkdir(exist_ok=True)

SEED = 42
rng = random.Random(SEED)

# Load COCO image metadata
with open(ANN / "instances_train2017.json") as f:
    train_data = json.load(f)

with open(ANN / "instances_val2017.json") as f:
    val_data = json.load(f)

train_images = sorted(
    train_data["images"], key=lambda x: x["id"]
)

val_images = sorted(
    val_data["images"], key=lambda x: x["id"]
)

# Shuffle deterministically
rng.shuffle(train_images)
rng.shuffle(val_images)

# Create non-overlapping splits
train_subset = train_images[:4000]
validation_subset = val_images[:500]
test_subset = val_images[500:1500]

# Keep identifiers and filenames
def image_info(images):
    return [
        {"id": im["id"], "file_name": im["file_name"]}
        for im in images
    ]

splits = {
    "seed": SEED,
    "train": image_info(train_subset),
    "val": image_info(validation_subset),
    "test": image_info(test_subset),
}

# Verify no overlapping image IDs
train_ids = {im["id"] for im in train_subset}
val_ids = {im["id"] for im in validation_subset}
test_ids = {im["id"] for im in test_subset}

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

output_file = OUT / "coco2017_seed42.json"

with open(output_file, "w") as f:
    json.dump(splits, f, indent=2)

print("COCO subset preparation complete")
print("Training images:", len(train_subset))
print("Validation images:", len(validation_subset))
print("Testing images:", len(test_subset))
print("Dataset split overlap: None")
print("Saved:", output_file)
