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


---

### Entry 007 – November 22, 2026
**Activity:** NWPU VHR-10 dataset acquisition

**Script:** scripts/03_download_nwpu.sh

**Dataset:** NWPU VHR-10
**Source:** https://gcheng-nwpu.github.io/#Datasets
**Download:** https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip

**Verified Results:**
- Archive: data/raw/NWPU-VHR-10.zip
- Download size: approximately 74 MB
- Archive integrity check: Passed
- Ground-truth annotation directory identified

**Observation:**
Dataset downloaded successfully to the KSU DGX server.
Original ZIP retained for reproducibility.
Dataset files excluded from GitHub using .gitignore.

**Status:** Completed

---

### Entry 008 – November 22, 2026
**Activity:** Dataset extraction and preliminary inspection

**Script:** scripts/04_inspect_nwpu.sh

**Verified Results:**
- Total images: 800
- Ground-truth annotation files: 650
- Expected negative images: 150
- Dataset extraction: Successful

**Sample Annotation:**
(208,361),(272,418),1
(234,429),(289,481),1
(322,473),(376,540),1
(313,631),(384,697),1

**Initial Interpretation:**
- First coordinate pair: bounding-box corner
- Second coordinate pair: opposite bounding-box corner
- Final integer: object class ID

**Observations:**
- Annotations are not in standard YOLO format.
- Class-ID mapping must be confirmed from dataset documentation.
- Complete bounding-box validation is pending.
- The 150 expected negative images still require verification.

**Status:** Preliminary inspection completed

---

### Entry 009 – Pending
**Activity:** Dataset quality audit

**Script:** scripts/05_analyze_nwpu.py

**Planned Checks:**
- Verify image and annotation correspondence
- Count objects per class
- Validate bounding-box coordinates
- Identify missing or malformed annotations
- Verify negative images
- Confirm all class definitions

**Status:** Not yet executed


### Entry 010 – November 22, 2026
**Activity:** Visual quality control of NWPU VHR-10

**Script:** scripts/07_visualize_nwpu.py

**Completed:**
- Inspected 5 positive images.
- Inspected 2 negative images.
- Verified visible alignment of sampled bounding boxes.
- Confirmed multiple classes within individual images.
- Observed different annotation conventions across categories.

**Observations:**
- Sampled positive bounding boxes appear correctly positioned.
- Baseball diamond annotations may cover the infield
  rather than the complete playing field.
- Some small vehicles appear unannotated.
- The negative images have no annotation files.
- One negative image contains possible small vehicles
  or construction equipment.

**Research Limitation:**
Negative images are unannotated but may contain
small instances of target categories. Predictions
on these images require careful error interpretation.

**Decision:**
Preserve original annotations.
Review negative-image detections during error analysis.
Do not automatically interpret all detections on negative
images as visually incorrect.

**Status:** Initial visual quality control completed.


### Dataset Audit Correction – November 23, 2026

**Script:** scripts/05_analyze_nwpu.py

**Issue:** The initial audit used image filename stems
as dictionary keys. Positive and negative image folders
contained 150 duplicate filename stems.

**Initial result:** 210 bounding-box errors.

**Correction:** Match ground-truth annotations only
against images in positive_image_set.

**Verified results:**
- Positive images: 650
- Negative images: 150
- Total images: 800
- Annotation files: 650
- Bounding boxes: 3,896
- Coordinate validation errors: 0

**Status:** Resolved.


### Entry 011 – November 23, 2026
**Activity:** Reproducible dataset splitting

**Script:** scripts/08_split_nwpu.py
**Dataset:** NWPU VHR-10
**Random Seed:** 42
**Split:** 70% train / 15% validation / 15% test

**Results:**

Training:
- Total images: 560
- Positive: 455
- Negative: 105
- Annotated objects: 2848

Validation:
- Total images: 120
- Positive: 98
- Negative: 22
- Annotated objects: 520

Testing:
- Total images: 120
- Positive: 97
- Negative: 23
- Annotated objects: 528

**Verification:**
- All 10 classes present in every split: PASS
- No image assigned to multiple splits: PASS
- Exact duplicate image check: PASS
- All 3896 annotated objects accounted for: PASS
- Random seed reproducibility: 42

**Files Created:**
- configs/nwpu_split_manifest.csv
- configs/nwpu_split_summary.json

**Research Decision:**
Use the same fixed partitions for all experiments.
Use validation data for prompt selection and tuning.
Reserve the test set for final model evaluation.
Original images and annotations remain unchanged.

**Limitation:**
Potential spatial overlap between aerial images
has not yet been evaluated.

**Status:** Completed


### Entry 012 – November 23, 2026
**Activity:** NWPU VHR-10 annotation conversion and validation

**Scripts:**
- scripts/09_convert_nwpu_to_yolo.py
- scripts/10_validate_yolo_conversion.py

**Results:**
- Training: 560 images, 2848 objects
- Validation: 120 images, 520 objects
- Testing: 120 images, 528 objects
- Total: 800 images, 3896 objects

**Validation:**
- Image verification: PASS
- Bounding-box verification: PASS
- Class ID conversion: PASS
- Coordinate normalization: PASS
- Detected conversion errors: 0

**Dataset Configuration:**
data/processed/nwpu_yolo/data.yaml

**Research Decision:**
Use the converted dataset for zero-shot evaluation,
few-shot adaptation, and full fine-tuning.
Keep the test partition reserved for final evaluation.

**Status:** Completed


### Entry 013 – November 23, 2026
**Experiment:** E01 – Zero-Shot Baseline

**Script:** scripts/11_zero_shot_baseline.py
**Model:** YOLOv8s-WorldV2
**Dataset:** NWPU VHR-10
**Split:** Validation
**Images:** 120
**Ground-truth objects:** 520
**GPU:** Tesla V100-SXM2-32GB

**Configuration:**
- Image size: 640
- Batch size: 8
- Confidence threshold: 0.001
- NMS IoU threshold: 0.70
- Classes: 10
- Training: None
- Model weights: Unchanged

**Evaluation Results:**
- Precision: 0.6066
- Recall: 0.0331
- mAP@0.50: 0.0471
- mAP@0.75: 0.0181
- mAP@0.50:0.95: 0.0217

**Per-Class mAP@0.50:0.95:**
- Airplane: 0.1282
- Ship: 0.0153
- Storage tank: 0.0000
- Baseball diamond: 0.0000
- Tennis court: 0.0000
- Basketball court: 0.0000
- Ground track field: 0.0000
- Harbor: 0.0000
- Bridge: 0.0000
- Vehicle: 0.0738

**Observations:**
- First formal zero-shot baseline completed.
- Recall is very low.
- Airplane achieved the highest class mAP.
- Vehicle and ship achieved nonzero mAP.
- Seven categories achieved zero mAP.
- No target-domain fine-tuning was performed.
- Predictions require visual error analysis.

**Outputs:**
- results/zero_shot/e01_baseline_metrics.json
- runs/e01_zero_shot_baseline/

**Status:** Completed

### Entry 014 – November 23, 2026
**Experiment:** E02 – Domain-Aware Zero-Shot Prompting

**Script:** scripts/12_domain_aware_prompts.py
**Model:** YOLOv8s-WorldV2
**Dataset:** NWPU VHR-10
**Split:** Validation
**Images:** 120
**Objects:** 520

**Experiment Design:**
Compared simple class-name prompts with descriptions
containing "viewed from above".

**E01 Baseline:**
- Precision: 0.6066
- Recall: 0.0331
- mAP50: 0.0471
- mAP50-95: 0.0217

**E02 Results:**
- Precision: 0.5328
- Recall: 0.0239
- mAP50: 0.0208
- mAP50-95: 0.0080

**Key Findings:**
- Domain-aware prompts reduced overall performance.
- mAP50 decreased by approximately 56%.
- Airplane performance decreased.
- Vehicle performance decreased substantially.
- Ground track field showed a small improvement.
- Seven classes had near-zero or zero performance.
- No model weights were updated.

**Interpretation:**
Adding aerial-view language did not improve
zero-shot performance in this experiment.
Prompt effectiveness appears category-dependent.
Further testing and visual error analysis are needed.

**Outputs:**
- configs/e02_domain_prompts.yaml
- results/zero_shot/e02_domain_metrics.json
- runs/e02_domain_aware/

**Status:** Completed

### Entry 015 – November 23, 2026
**Experiment:** E03 – Synonym and Object-Specific Prompting

**Script:** scripts/13_synonym_prompts.py
**Model:** YOLOv8s-WorldV2
**Dataset:** NWPU VHR-10
**Split:** Validation
**Images:** 120
**Ground-truth objects:** 520
**Training:** None

**Prompts:**
- aircraft
- vessel
- industrial storage tank
- baseball infield
- tennis court
- basketball court
- running track
- port
- bridge
- motor vehicle

**E03 Results:**
- Precision: 0.7445
- Recall: 0.0257
- mAP50: 0.0336
- mAP75: 0.0136
- mAP50-95: 0.0155

**Comparison (mAP50-95):**
- E01 simple prompts: 0.0217
- E02 aerial prompts: 0.0080
- E03 alternative prompts: 0.0155

**Important Observations:**
- E01 remains the best overall configuration.
- Aircraft improved over airplane for the tested setups.
- Vessel performed worse than ship.
- Motor vehicle performed worse than vehicle.
- Overall recall remains very low.
- No model weights were updated.

**Research Limitation:**
Multiple prompts changed simultaneously in E03.
Per-class changes cannot be attributed exclusively
to individual prompt terms without controlled tests.

**Outputs:**
- configs/e03_synonym_prompts.yaml
- results/zero_shot/e03_synonym_metrics.json
- results/zero_shot/e01_e02_e03_comparison.json
- runs/e03_synonym_prompts/

**Status:** Completed


### Entry 016 – November 23, 2026
**Experiment:** E04 – Hybrid Zero-Shot Prompting

**Script:** scripts/14_hybrid_prompts.py
**Model:** YOLOv8s-WorldV2
**Dataset:** NWPU VHR-10
**Split:** Validation
**Images:** 120
**Ground-truth objects:** 520
**Training:** None

**Prompt Configuration:**
- Changed airplane to aircraft
- Retained the other nine original class names

**E04 Results:**
- Precision: 0.6209
- Recall: 0.0430
- mAP50: 0.0513
- mAP75: 0.0195
- mAP50-95: 0.0232

**Per-Class mAP50-95:**
- Airplane: 0.1447
- Ship: 0.0134
- Vehicle: 0.0742
- Other seven categories: 0.0000

**Comparison:**
- E01 mAP50-95: 0.0217
- E02 mAP50-95: 0.0080
- E03 mAP50-95: 0.0155
- E04 mAP50-95: 0.0232

**Research Findings:**
- E04 achieved the highest overall validation mAP.
- Replacing airplane with aircraft improved
  airplane detection compared with E01.
- Recall improved relative to E01.
- Seven categories still had zero mAP50-95.
- No model weights were updated.

**Decision:**
Retain E04 as the strongest observed zero-shot
validation configuration.
Keep the test partition untouched.
Proceed to limited-data model adaptation.

**Status:** Completed


### Entry 017 – November 23, 2026
**Experiment:** E05 – One-Shot Fine-Tuning

**Training script:** scripts/16_train_oneshot.py
**Validation script:** scripts/17_validate_oneshot.py

**Model:** YOLOv8s-WorldV2
**Initial weights:** Original pretrained checkpoint
**Prompt configuration:** E04 hybrid
**GPU:** Tesla V100-SXM2-32GB

**Training configuration:**
- Training images: 10
- Training bounding boxes: 50
- Validation images: 120
- Validation bounding boxes: 520
- Epochs: 50
- Batch size: 4
- Image size: 640
- Optimizer: AdamW
- Initial learning rate: 0.0005
- Seed: 42

**Standardized Validation Results:**
- Precision: 0.7023
- Recall: 0.3494
- mAP50: 0.4071
- mAP75: 0.2987
- mAP50-95: 0.2631

**Comparison with E04:**
- E04 mAP50: 0.0513
- E05 mAP50: 0.4071
- E04 mAP50-95: 0.0232
- E05 mAP50-95: 0.2631

**Per-Class E05 mAP50-95:**
- Aircraft: 0.1660
- Ship: 0.5268
- Storage tank: 0.5264
- Baseball diamond: 0.5841
- Tennis court: 0.0234
- Basketball court: 0.0214
- Ground track field: 0.6034
- Harbor: 0.0058
- Bridge: 0.0008
- Vehicle: 0.1731

**Findings:**
- Training completed all 50 epochs.
- Both best.pt and last.pt were saved.
- Independent validation reproduced training results.
- Fine-tuning substantially improved detection.
- Ground track field and baseball diamond performed well.
- Bridge, harbor, and basketball court remained difficult.
- Test set was not used.
- Results require confirmation across seeds and held-out test data.

**Checkpoint:**
runs/fewshot/e05_oneshot/weights/best.pt

**Metrics:**
results/fewshot/e05_validation_metrics.json

**Status:** Completed and validated

