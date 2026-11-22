"""
Project: YOLO-World Remote-Sensing Detection
Experiment: Setup-02
Purpose: Verify GPU inference with custom class prompts.
"""

from pathlib import Path
import torch
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/yolov8s-worldv2.pt"
IMAGE = ROOT / "data/samples/bus.jpg"

# Load pretrained model
model = YOLOWorld(str(MODEL))

# Test custom text prompts
model.set_classes(["person", "bus"])

# Run inference on GPU 0
results = model.predict(
    source=str(IMAGE),
    device=0,
    imgsz=640,
    conf=0.25,
    save=True,
    project=str(ROOT / "runs"),
    name="gpu_smoke_test",
    exist_ok=True
)

# Display results
print("\nGPU:", torch.cuda.get_device_name(0))
print("Classes:", model.names)

for result in results:
    print("Detections:", len(result.boxes))
    print("Inference timing (ms):", result.speed)

print("GPU inference completed successfully.")
