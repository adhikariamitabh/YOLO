#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
source "$HOME/venvs/yolo-basic/bin/activate"

mkdir -p runs/coco_baseline/logs
log="runs/coco_baseline/logs/yolo11n_$(date +%Y%m%d_%H%M%S).log"

yolo val \
  model=yolo11n.pt \
  data=coco_full.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=4 \
  plots=True \
  project=runs/coco_baseline \
  name=yolo11n 2>&1 | tee "$log"
