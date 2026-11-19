from pathlib import Path
from collections import defaultdict
import hashlib

base = Path("data/lateral_flow/combined")
images_dir = base / "images"
labels_dir = base / "labels"

images = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

hashes = defaultdict(list)
invalid = []

for img in images:
    label = labels_dir / f"{img.stem}.txt"

    if not label.exists():
        invalid.append(f"Missing label: {img.name}")
        continue

    # Check YOLO OBB format
    for line in label.read_text().splitlines():
        fields = line.split()

        if len(fields) != 9:
            invalid.append(f"Invalid OBB format: {label.name}")
            continue

        try:
            values = [float(x) for x in fields]
            if values[0] != 0 or not all(
                0 <= x <= 1 for x in values[1:]
            ):
                invalid.append(f"Invalid values: {label.name}")
        except ValueError:
            invalid.append(f"Non-numeric values: {label.name}")

    # Check exact duplicate images
    digest = hashlib.sha256(img.read_bytes()).hexdigest()
    hashes[digest].append(img.name)

duplicates = [
    names for names in hashes.values()
    if len(names) > 1
]

print("Images:", len(images))
print("Label files:", len(list(labels_dir.glob("*.txt"))))
print("Exact duplicate groups:", len(duplicates))
print("Annotation problems:", len(invalid))

for group in duplicates:
    print("Duplicate:", group)

for problem in invalid[:20]:
    print("Problem:", problem)
