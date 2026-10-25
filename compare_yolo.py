from itertools import product
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt
# Load pretrained YOLO models
model11 = YOLO("yolo11n.pt")
model26 = YOLO("yolo26n.pt")

print("YOLO11 and YOLO26 models loaded successfully.")

from pathlib import Path

# Project paths
BASE = Path(__file__).resolve().parent
# Find all JPG images in the input directory
IMAGES = sorted((BASE / "data/images").glob("*.jpg"))

print(f"Total images: {len(IMAGES)}")

#IMAGE = BASE / "data/images/bus.jpg"
# Store comparison results
comparison = []
# Run both models on the same original image

#for name, model in [("yolo11", model11), ("yolo26", model26)]:

for image, (name, model) in product(
    IMAGES, [("yolo11", model11), ("yolo26", model26)]
):
    print(f"\nRunning {name}...")

    results = model.predict(
        source=str(image),
        imgsz=640,
        device="cpu",
        project=str(BASE / "runs/smoke"),
        name=name,
        exist_ok=True,
        save=True
    )
    # Extract prediction results
    r = results[0]

    count = len(r.boxes)
    avg_conf = float(r.boxes.conf.mean()) if count else 0.0
    inference_ms = r.speed["inference"]

    print(f"Objects detected: {count}")
    print(f"Average confidence: {avg_conf:.3f}")
    print(f"Inference time: {inference_ms:.2f} ms")
    # Store results for CSV and graph
    comparison.append([
        name,
        count,
        round(avg_conf, 3),
        round(inference_ms, 2),
        image.name
    ])
# Save comparison results
csv_file = BASE / "runs/smoke/comparison.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Objects", "Avg Confidence", "Inference (ms)", "Image"])
    writer.writerows(comparison)

print(f"\nComparison saved to: {csv_file}")

# Compare inference times
models = [f"{row[0]}\n{row[4]}" for row in comparison]
times = [row[3] for row in comparison]

plt.bar(models, times)
plt.ylabel("Inference Time (ms)")
plt.title("YOLO11 vs YOLO26 - CPU Inference Time")

plt.savefig(BASE / "runs/smoke/inference_comparison.png")
plt.show()
