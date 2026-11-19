from pathlib import Path
from itertools import combinations
import cv2
import numpy as np
import csv

BASE = Path("data/lateral_flow/dataset/images")
SPLITS = ["train", "val", "test"]

def phash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {path}")

    img = cv2.resize(img, (32, 32)).astype(np.float32)
    dct = cv2.dct(img)
    low = dct[:8, :8]

    median = np.median(low.flatten()[1:])
    bits = low > median

    return bits.flatten()

images = []

for split in SPLITS:
    for path in sorted((BASE / split).iterdir()):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            images.append((split, path, phash(path)))

results = []

for (s1, p1, h1), (s2, p2, h2) in combinations(images, 2):
    if s1 == s2:
        continue

    distance = int(np.count_nonzero(h1 != h2))
    results.append((distance, s1, p1.name, s2, p2.name))

results.sort()

print(f"Total images: {len(images)}")
print(f"Cross-split comparisons: {len(results)}")
print(f"Pairs with pHash distance <= 8: "
      f"{sum(r[0] <= 8 for r in results)}")

print("\n20 MOST SIMILAR CROSS-SPLIT PAIRS")
for row in results[:20]:
    print(row)

output = Path("runs/lateral_flow/cross_split_similarity.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["distance", "split1", "image1",
                     "split2", "image2"])
    writer.writerows(results)

print(f"\nSaved: {output}")
