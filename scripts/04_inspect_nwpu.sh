#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/projects/yolo-world"

ARCHIVE="data/raw/NWPU-VHR-10.zip"
DATASET="data/raw/NWPU-VHR-10"

echo "=== NWPU VHR-10 Dataset Inspection ==="

# Extract without modifying the original ZIP
if [ ! -d "$DATASET" ]; then
    echo "Extracting dataset..."
    unzip -q "$ARCHIVE" -d data/raw/
else
    echo "Dataset directory already exists."
fi

echo ""
echo "=== Directory Structure ==="
find "$DATASET" -maxdepth 2 -type d | sort

echo ""
echo "=== README ==="
sed -n '1,100p' "$DATASET/README.md"

echo ""
echo "=== Image Count ==="
find "$DATASET" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' \
    -o -iname '*.png' -o -iname '*.tif' \) \
    | wc -l

echo ""
echo "=== Annotation Count ==="
find "$DATASET/ground_truth" -type f \
    -name '*.txt' | wc -l

echo ""
echo "=== Sample Annotation: 008.txt ==="
sed -n '1,10p' "$DATASET/ground_truth/008.txt"

echo ""
echo "Inspection completed."
