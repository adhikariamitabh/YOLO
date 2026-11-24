"""
YOLO-World / NWPU VHR-10
Cross-Split Image Similarity Audit

Checks:
- Training vs validation
- Training vs test
- Validation vs test

Methods:
- SHA-256 file hash
- Decoded pixel hash
- Perceptual hash (pHash)
- Difference hash (dHash)

Does not modify datasets, train models,
read labels, or calculate detection metrics.
"""

from pathlib import Path
from itertools import combinations
from collections import Counter
import csv
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    ROOT / "data/processed/nwpu_yolo/images"
)

OUTPUT = ROOT / "results/data_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": DATA_ROOT / "train",
    "val": DATA_ROOT / "val",
    "test": DATA_ROOT / "test",
}

EXPECTED_COUNTS = {
    "train": 560,
    "val": 120,
    "test": 120,
}

PHASH_THRESHOLD = 10
DHASH_THRESHOLD = 8

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png"
}


# --------------------------------------------------
# 2. Hash functions
# --------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def bits_to_int(bits):
    value = 0

    for bit in bits.flatten():
        value = (value << 1) | int(bit)

    return value


def perceptual_hash(gray):
    resized = cv2.resize(
        gray, (32, 32),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    dct = cv2.dct(resized)

    low_frequency = dct[:8, :8].flatten()

    median = np.median(
        low_frequency[1:]
    )

    bits = low_frequency > median

    # Exclude the DC coefficient.
    bits[0] = False

    return bits_to_int(bits)


def difference_hash(gray):
    resized = cv2.resize(
        gray, (9, 8),
        interpolation=cv2.INTER_AREA
    )

    differences = (
        resized[:, 1:] > resized[:, :-1]
    )

    return bits_to_int(differences)


def hamming_distance(a, b):
    return (a ^ b).bit_count()


# --------------------------------------------------
# 3. Load and fingerprint images
# --------------------------------------------------

fingerprints = {}

print("\n=== DATASET SIMILARITY AUDIT ===")

for split, directory in SPLITS.items():

    assert directory.exists(), (
        f"Missing directory: {directory}"
    )

    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    assert len(paths) == EXPECTED_COUNTS[split], (
        f"{split}: expected "
        f"{EXPECTED_COUNTS[split]} images, "
        f"found {len(paths)}"
    )

    fingerprints[split] = []

    for path in paths:

        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Unreadable image: {path}"
            )

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        pixel_digest = hashlib.sha256()

        pixel_digest.update(
            str(image.shape).encode()
        )

        pixel_digest.update(
            image.tobytes()
        )

        fingerprints[split].append({
            "path": path,
            "file_sha256": sha256_file(path),
            "pixel_sha256": pixel_digest.hexdigest(),
            "phash": perceptual_hash(gray),
            "dhash": difference_hash(gray),
        })

    print(
        f"{split:<8}: "
        f"{len(fingerprints[split])} images"
    )


# --------------------------------------------------
# 4. Compare images across splits
# --------------------------------------------------

candidates = []
comparison_counts = Counter()
candidate_counts = Counter()

exact_file_count = 0
exact_pixel_count = 0

print("\n=== CROSS-SPLIT COMPARISONS ===")

for split_a, split_b in combinations(
    SPLITS.keys(), 2
):

    group = f"{split_a}_vs_{split_b}"

    for image_a in fingerprints[split_a]:

        for image_b in fingerprints[split_b]:

            comparison_counts[group] += 1

            p_distance = hamming_distance(
                image_a["phash"],
                image_b["phash"]
            )

            d_distance = hamming_distance(
                image_a["dhash"],
                image_b["dhash"]
            )

            exact_file = (
                image_a["file_sha256"]
                == image_b["file_sha256"]
            )

            exact_pixel = (
                image_a["pixel_sha256"]
                == image_b["pixel_sha256"]
            )

            if exact_file:
                exact_file_count += 1

            if exact_pixel:
                exact_pixel_count += 1

            possible_match = (
                exact_file
                or exact_pixel
                or p_distance <= PHASH_THRESHOLD
                or d_distance <= DHASH_THRESHOLD
            )

            if not possible_match:
                continue

            if exact_file or exact_pixel:
                risk = "EXACT"
            elif (
                p_distance <= 4
                or d_distance <= 4
            ):
                risk = "HIGH_SIMILARITY"
            else:
                risk = "REVIEW"

            record = {
                "group": group,
                "split_a": split_a,
                "image_a": str(
                    image_a["path"].relative_to(ROOT)
                ),
                "split_b": split_b,
                "image_b": str(
                    image_b["path"].relative_to(ROOT)
                ),
                "phash_distance": p_distance,
                "dhash_distance": d_distance,
                "exact_file": exact_file,
                "exact_pixels": exact_pixel,
                "risk": risk,
            }

            candidates.append(record)
            candidate_counts[group] += 1

    print(
        f"{group:<16} "
        f"comparisons={comparison_counts[group]:>7} "
        f"candidates={candidate_counts[group]}"
    )


# --------------------------------------------------
# 5. Rank candidate pairs
# --------------------------------------------------

risk_priority = {
    "EXACT": 0,
    "HIGH_SIMILARITY": 1,
    "REVIEW": 2,
}


def candidate_sort_key(record):
    return (
        risk_priority[record["risk"]],
        min(
            record["phash_distance"] / PHASH_THRESHOLD,
            record["dhash_distance"] / DHASH_THRESHOLD
        ),
        record["phash_distance"]
        + record["dhash_distance"]
    )


candidates.sort(key=candidate_sort_key)


# --------------------------------------------------
# 6. Save candidate CSV
# --------------------------------------------------

CSV_FILE = (
    OUTPUT / "cross_split_similarity_candidates.csv"
)

fields = [
    "group",
    "split_a",
    "image_a",
    "split_b",
    "image_b",
    "phash_distance",
    "dhash_distance",
    "exact_file",
    "exact_pixels",
    "risk",
]

with CSV_FILE.open("w", newline="") as f:

    writer = csv.DictWriter(
        f, fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(candidates)


# --------------------------------------------------
# 7. Generate contact sheet
# --------------------------------------------------

PREVIEW_FILE = (
    OUTPUT / "cross_split_top24.jpg"
)

if candidates:

    selected = candidates[:24]

    tile_width = 420
    tile_height = 200
    row_height = 245

    sheet = Image.new(
        "RGB",
        (tile_width * 2, row_height * len(selected)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, record in enumerate(selected):

        for side, key in enumerate(
            ("image_a", "image_b")
        ):

            path = ROOT / record[key]

            with Image.open(path) as source:
                photo = source.convert("RGB")

                thumbnail = ImageOps.contain(
                    photo,
                    (400, 185)
                )

            x = side * tile_width + 10
            y = i * row_height + 5

            sheet.paste(thumbnail, (x, y))

            draw.text(
                (x, y + 188),
                f"{record['split_a'] if side == 0 else record['split_b']}: "
                f"{path.name[:43]}",
                fill="black"
            )

        draw.text(
            (10, i * row_height + 225),
            (
                f"{record['risk']} | "
                f"pHash={record['phash_distance']} | "
                f"dHash={record['dhash_distance']}"
            ),
            fill="black"
        )

    sheet.save(
        PREVIEW_FILE,
        quality=90
    )


# --------------------------------------------------
# 8. Save audit summary
# --------------------------------------------------

risk_counts = Counter(
    record["risk"] for record in candidates
)

summary = {
    "dataset": "NWPU VHR-10",
    "images_per_split": EXPECTED_COUNTS,
    "total_images": 800,
    "methods": [
        "SHA256",
        "decoded_pixel_SHA256",
        "pHash",
        "dHash"
    ],
    "phash_threshold": PHASH_THRESHOLD,
    "dhash_threshold": DHASH_THRESHOLD,
    "cross_split_comparisons": dict(
        comparison_counts
    ),
    "candidate_pairs_by_split": dict(
        candidate_counts
    ),
    "total_candidate_pairs": len(candidates),
    "exact_file_pairs": exact_file_count,
    "exact_pixel_pairs": exact_pixel_count,
    "risk_counts": dict(risk_counts),
    "candidate_csv": str(CSV_FILE),
    "preview": (
        str(PREVIEW_FILE)
        if candidates else None
    ),
    "test_labels_used": False,
    "test_metrics_calculated": False,
    "dataset_modified": False,
    "note": (
        "Perceptual hash matches are candidates, "
        "not confirmed duplicates. Hashing may miss "
        "cropped, rotated or differently framed scenes."
    ),
}

SUMMARY_FILE = (
    OUTPUT / "split_similarity_summary.json"
)

SUMMARY_FILE.write_text(
    json.dumps(summary, indent=2) + "\n"
)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n=== AUDIT SUMMARY ===")

print(
    "Total image-pair comparisons:",
    sum(comparison_counts.values())
)

print("Exact file matches:", exact_file_count)
print("Exact pixel matches:", exact_pixel_count)

print(
    "Potentially similar pairs:",
    len(candidates)
)

for risk in (
    "EXACT",
    "HIGH_SIMILARITY",
    "REVIEW"
):
    print(
        f"{risk:<18}: "
        f"{risk_counts[risk]}"
    )

print("\n=== TOP CANDIDATE PAIRS ===")

for record in candidates[:15]:

    print(
        f"{record['group']:<16} "
        f"p={record['phash_distance']:>2} "
        f"d={record['dhash_distance']:>2} "
        f"{record['risk']}"
    )

    print("  A:", record["image_a"])
    print("  B:", record["image_b"])


print("\n=== OUTPUT FILES ===")

print("Summary:", SUMMARY_FILE)
print("Candidates:", CSV_FILE)

if candidates:
    print("Contact sheet:", PREVIEW_FILE)
else:
    print("No candidate pairs met the thresholds.")

print("\nTest labels used: NO")
print("Test metrics calculated: NO")
print("Dataset modified: NO")

print("\nAUDIT COMPLETED")
