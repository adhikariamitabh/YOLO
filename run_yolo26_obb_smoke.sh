#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/YOLO
source ~/venvs/yolo-basic/bin/activate

echo "Starting YOLO26n-OBB smoke test..."

yolo obb train \
  model=yolo26n-obb.pt \
  data=lateral_flow.yaml \
  epochs=1 \
  imgsz=640 \
  batch=8 \
  device=0 \
  workers=4 \
  seed=42 \
  plots=True \
  project="$PWD/runs/lateral_flow" \
  name=smoke_test

echo "Smoke test completed."
