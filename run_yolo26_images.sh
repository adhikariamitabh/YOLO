#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/YOLO
source ~/venvs/yolo-basic/bin/activate

echo "Running YOLO26n image detection..."

yolo predict \
  model=yolo26n.pt \
  source=data/videos \
  imgsz=640 \
  device=0 \
  save=True \
  project="$PWD/runs/image_detection" \
  name=yolo26n \
  exist_ok=True

echo "Detection complete."
echo "Results saved in runs/image_detection/yolo26n/"
