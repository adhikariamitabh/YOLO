from pathlib import Path
from shutil import copy2

base = Path("data/lateral_flow")
output = base / "combined"

(output / "images").mkdir(parents=True, exist_ok=True)
(output / "labels").mkdir(parents=True, exist_ok=True)

datasets = [
    ("Pos_YOLO_OBB", "pos"),
    ("Neg_YOLO_OBB", "neg")
]

for folder, prefix in datasets:
    image_dir = base / folder / "images"
    label_dir = base / folder / "labels"

    for image in image_dir.iterdir():
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            raise FileNotFoundError(label)

        copy2(image, output / "images" / f"{prefix}_{image.name}")
        copy2(label, output / "labels" / f"{prefix}_{image.stem}.txt")

print("Total images:", len(list((output / "images").iterdir())))
print("Total labels:", len(list((output / "labels").iterdir())))
