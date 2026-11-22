#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/projects/yolo-world"
DATA_DIR="$PROJECT/data/raw"

URL="https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip"
ZIP="$DATA_DIR/NWPU-VHR-10.zip"

mkdir -p "$DATA_DIR"

echo "Downloading NWPU VHR-10 dataset..."

if [ -f "$ZIP" ]; then
    echo "Archive already exists: $ZIP"
else
    curl -fL --retry 3 "$URL" -o "$ZIP"
fi

echo "Checking archive integrity..."
unzip -tq "$ZIP"

echo "Dataset archive:"
ls -lh "$ZIP"

echo "First 30 archive entries:"
unzip -Z -1 "$ZIP" | head -n 30 || true

echo "Download and archive verification completed."
