#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/YOLO
source ~/venvs/yolo-basic/bin/activate

# Evaluate YOLO26n-OBB on the original 45 test images
yolo obb val \
  model=runs/lateral_flow/yolo26n_full/weights/best.pt \
  data=lateral_flow.yaml \
  split=test \
  imgsz=640 \
  batch=8 \
  device=0 \
  plots=True \
  project="$PWD/runs/lateral_flow" \
  name=yolo26n_test
