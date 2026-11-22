"""
Project: YOLO-World Remote-Sensing Object Detection
Course: CS 7267
Experiment: Setup-01
Purpose: Load pretrained YOLO-World and verify configuration.
"""

from pathlib import Path
import torch
from ultralytics import YOLOWorld

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "yolov8s-worldv2.pt"

print("Project directory:", PROJECT_DIR)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Load pretrained weights.
# The first run downloads weights if needed.
if MODEL_PATH.exists():
    model = YOLOWorld(str(MODEL_PATH))
else:
    model = YOLOWorld("yolov8s-worldv2.pt")

    # Save a copy inside the project's models directory.
    import shutil
    shutil.copy2(model.ckpt_path, MODEL_PATH)

print("Model loaded successfully")
print("Model task:", model.task)
print("Number of classes:", len(model.names))
print("Model weights:", MODEL_PATH)
