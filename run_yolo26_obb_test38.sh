#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/YOLO
source ~/venvs/yolo-basic/bin/activate

# Evaluate YOLO26n-OBB on 38 filtered test images
# Seven identified duplicate images were excluded.
# This is a filtered evaluation, not a fully independent test.

yolo obb val \
  model=runs/lateral_flow/yolo26n_full/weights/best.pt \
  data=lateral_flow_test38.yaml \
  split=test \
  imgsz=640 \
  batch=8 \
  device=0 \
  plots=True \
  project="$PWD/runs/lateral_flow" \
  name=yolo26n_test38 \
  2>&1 | tee runs/lateral_flow/test38_metrics.log
