# YOLO-World Remote-Sensing Object Detection
## Project Logbook

**Course:** CS 7267 – Machine Learning
**Server:** KSU DGX (dgxone)
**Project Directory:** ~/projects/yolo-world
**Python Environment:** ~/venvs/yolo-world
**Started:** November 22, 2026

---

## 1. Project Objective

Investigate how YOLO-World adapts to remote-sensing imagery
using zero-shot prompting, domain-aware prompting,
one-shot/few-shot adaptation, and full fine-tuning.

### Research Questions

1. Can domain-aware prompts improve zero-shot detection?
2. Can limited labeled examples improve performance?
3. How much improvement comes from full fine-tuning?

### Dataset
- Primary: NWPU VHR-10
- Fallback: VisDrone

### Evaluation Metrics
- Precision
- Recall
- IoU
- mAP@0.50
- mAP@0.50:0.95

---

## 2. Environment Configuration

| Component | Configuration |
|-----------|---------------|
| Server | KSU DGX |
| GPU | NVIDIA Tesla V100-SXM2-32GB |
| Python | 3.12.3 |
| Environment | Python venv |
| Environment Name | yolo-world |
| PyTorch | 2.7.1+cu126 |
| CUDA Build | 12.6 |
| GPU Available | True |
| Ultralytics | Pending installation |

The existing YOLO project and environment remain unchanged.

---

## 3. Experiment Register

| ID | Experiment | Status |
|----|------------|--------|
| E01 | Zero-shot baseline | Not started |
| E02 | Domain-aware prompts | Not started |
| E03 | Detailed prompt comparison | Not started |
| E04 | One-shot adaptation | Not started |
| E05 | Five-shot adaptation | Not started |
| E06 | Ten-shot adaptation | Not started |
| E07 | Full fine-tuning | Not started |
| E08 | YOLO model comparison | Not started |

---

## 4. Chronological Project Log

### Entry 001 – November 22, 2026
**Activity:** Project initialization

**Completed:**
- Reviewed project proposal.
- Selected KSU DGX for GPU experiments.
- Selected Python venv instead of Conda.
- Created ~/projects/yolo-world.
- Created ~/venvs/yolo-world.
- Activated the new environment.
- Verified Python 3.12.3.

**Decision:**
Keep the existing YOLO environment and projects intact.

**Status:** Completed

---

### Entry 002 – November 22, 2026
**Activity:** PyTorch installation and GPU verification

**Verified Results:**

- PyTorch: 2.7.1+cu126
- CUDA: 12.6
- GPU Available: True
- GPU: Tesla V100-SXM2-32GB

**Observation:**
PyTorch successfully recognizes the KSU NVIDIA GPU.

**Status:** Completed

---

### Entry 003 – Pending
**Activity:** Install Ultralytics

**Planned Commands:**

    python -m pip install ultralytics
    python -m pip check

**Status:** Not yet verified

---

## 5. Issues and Resolutions

Record errors, causes, solutions, and verification results here.

---

## 6. Research Observations

Record findings about:
- Prompt effectiveness
- Small-object detection
- False positives and missed objects
- Class-specific performance
- Domain shift
- Model adaptation

---

## 7. Next Steps

1. Install and verify Ultralytics.
2. Load pretrained YOLO-World weights.
3. Run a GPU inference smoke test.
4. Prepare the NWPU VHR-10 dataset.
5. Establish the zero-shot baseline.

---

## 8. Reproducibility Notes

For every experiment record:
- Experiment ID and date
- Model and package versions
- Dataset and split
- Prompts and training settings
- Random seed
- GPU configuration
- Evaluation metrics
- Output file locations
- Observations and limitations


### Entry 004 – Ultralytics Verification
**Activity:** YOLO-World installation
**Status:** Pending


### Entry 003 – November 22, 2026
**Activity:** Ultralytics installation and verification

**Verified Results:**
- Ultralytics: 8.4.216
- PyTorch: 2.7.1+cu126
- CUDA: 12.6
- GPU available: True
- GPU: Tesla V100-SXM2-32GB
- pip check: No broken requirements found

**Observation:**
Ultralytics imports successfully, and PyTorch
continues to recognize the NVIDIA GPU.

**Status:** Completed


### Entry 004 – YOLO-World Model Initialization

**Script:** scripts/01_load_yolo_world.py

**Model:** YOLOv8s-WorldV2

**Objective:** Verify pretrained model loading.

**Status:** Completed


### Entry 004 – November 22, 2026
**Activity:** YOLO-World pretrained model verification

**Script:** scripts/01_load_yolo_world.py
**Model:** YOLOv8s-WorldV2

**Verified Results:**
- PyTorch: 2.7.1+cu126
- CUDA available: True
- Model loaded successfully
- Model task: detect
- Default classes: 80
- Weights: models/yolov8s-worldv2.pt

**Observation:**
The pretrained YOLO-World model loads successfully.
GPU inference and custom text prompts have not yet
been tested.

**Status:** Completed


### Entry 005 – November 22, 2026
**Activity:** First YOLO-World GPU inference test

**Script:** scripts/02_gpu_smoke_test.py
**Model:** YOLOv8s-WorldV2
**Dataset:** Ultralytics sample image (bus.jpg)
**GPU:** NVIDIA Tesla V100-SXM2-32GB

**Configuration:**
- Device: GPU 0
- Input size: 640
- Confidence threshold: 0.25
- Custom prompts: person, bus

**Results:**
- Persons detected: 4
- Buses detected: 1
- Total detections: 5
- Preprocessing: 92.8 ms
- Inference: 12.7 ms
- Postprocessing: 4.3 ms

**Observations:**
- GPU inference completed successfully.
- Custom text prompts worked.
- CLIP dependencies were installed automatically.
- Model weights and experiment outputs remain on DGX.
- This is an environment smoke test, not a research baseline.

**Output:** runs/gpu_smoke_test/

**Status:** Completed


### Entry 006 – November 22, 2026
**Activity:** Repeat GPU inference verification

**Script:** scripts/02_gpu_smoke_test.py
**Model:** YOLOv8s-WorldV2
**GPU:** Tesla V100-SXM2-32GB

**Second Run Results:**
- Persons detected: 4
- Buses detected: 1
- Total detections: 5
- Preprocessing: 24.2 ms
- Inference: 12.3 ms
- Postprocessing: 1.9 ms
- Total processing: 38.4 ms

**Dependency Verification:**
- pip check: No broken requirements found.

**Observation:**
Detection counts remained consistent across both runs.
The second run had lower preprocessing latency.
Additional repetitions are required for reliable benchmarks.

**Status:** Completed


### Entry 006 – November 22, 2026
**Activity:** Repeat GPU inference verification

**Script:** scripts/02_gpu_smoke_test.py
**Model:** YOLOv8s-WorldV2
**GPU:** Tesla V100-SXM2-32GB

**Second Run Results:**
- Persons detected: 4
- Buses detected: 1
- Total detections: 5
- Preprocessing: 24.2 ms
- Inference: 12.3 ms
- Postprocessing: 1.9 ms
- Total processing: 38.4 ms

**Dependency Verification:**
- pip check: No broken requirements found.

**Observation:**
Detection counts remained consistent across both runs.
The second run had lower preprocessing latency.
Additional repetitions are required for reliable benchmarks.

**Status:** Completed

