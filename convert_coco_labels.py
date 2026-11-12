from pathlib import Path
from ultralytics.data.converter import convert_coco

BASE = Path(__file__).resolve().parent
DATASET = BASE / "data/coco2017"

ANNOTATIONS = DATASET / "annotations"
TEMP = DATASET / "converted_yolo"
LABELS = DATASET / "labels"

# Prevent accidental duplicate conversion
if TEMP.exists() or LABELS.exists():
    raise SystemExit(
        "Conversion output already exists. "
        "Inspect it before rerunning."
    )

print("Converting COCO 2017 annotations...")

convert_coco(
    labels_dir=str(ANNOTATIONS),
    save_dir=str(TEMP),
    cls91to80=True,
    use_segments=False
)

# Move labels into YOLO's expected directory structure
LABELS.mkdir()

for split in ("train2017", "val2017"):
    source = TEMP / "labels" / split
    destination = LABELS / split

    if not source.is_dir():
        raise FileNotFoundError(source)

    source.rename(destination)

    count = len(list(destination.glob("*.txt")))
    print(f"{split}: {count} label files")

print("COCO annotation conversion complete.")
