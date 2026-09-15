# 🎗️ Breast Lesion Detection in Digital Mammograms with YOLO11

A reproducible, publication-grade medical computer vision pipeline for lesion detection in mammography:
**data auditing → YOLO11 training → statistical and clinical evaluation (FROC + bootstrap CIs)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moftahykh/Breast-Lesion-Detection-YOLO11/blob/main/Breast_Lesion_Detection_YOLO11_.ipynb)

---

## Project Summary

This repository contains an end-to-end notebook that trains **YOLO11s** on mammogram lesion annotations and evaluates performance using both computer-vision and clinically relevant metrics.

The workflow is designed for **reproducibility and reporting quality**:
- Fixed random seed and environment/version logging
- Pre-training dataset quality audits (format, distributions, geometry, leakage)
- Transfer learning with YOLO11s
- Overfitting diagnosis from training dynamics
- Test-set evaluation with COCO metrics + **FROC analysis**
- **Bootstrap 95% confidence intervals** for operating-point metrics
- Error analysis and qualitative visualizations

> Note: final numeric results are intentionally computed live in the notebook and may differ across runs/hardware.

---

## Notebook Scope (Aligned with Implementation)

The notebook `Breast_Lesion_Detection_YOLO11_.ipynb` is organized as:

1. **Setup & Reproducibility**
   - Install dependencies (`ultralytics`, `opencv-python`, `matplotlib`, `pandas`, `pyyaml`)
   - Set global seed (`42`) and log runtime versions

2. **Dataset Acquisition & Configuration**
   - Uses Roboflow Universe dataset export
   - Validates and loads `data.yaml`
   - Renames ambiguous class label `'1'` to `'lesion'` for reporting clarity (ID unchanged)

3. **Data Quality Audit (Preprocessing)**
   - Structural integrity checks of labels and images
   - Split and class distribution analysis
   - Bounding-box geometry analysis
   - Rotation-aware data leakage detection across splits
   - Optional CLAHE enhancement demo

4. **Training & Diagnostics**
   - YOLO11s transfer learning with tuned settings for grayscale mammograms
   - Automated memorization/overfitting diagnosis using `results.csv`

5. **Evaluation (Clinical + Statistical)**
   - Standard test metrics (mAP@50, mAP@50:95)
   - Confidence-threshold analysis (PR/F1/confusion outputs)
   - **FROC curve** (sensitivity vs false positives per image)
   - **Bootstrap (1,000 resamples) 95% CIs** for key operating-point metrics

6. **Interpretability & Discussion**
   - False-negative / false-positive case analysis
   - Qualitative prediction gallery
   - Literature positioning, limitations, model card, references

---

## Dataset

- **Source:** Roboflow Universe — `breast-cancer-detection-jixc0` (v2)
- **License:** CC BY 4.0
- **Task:** Object detection on digital mammograms
- **Classes:** `lesion`, `tumor`
- **Documented split size:** 5,406 images total
  - Train: 4,675
  - Valid: 530
  - Test: 201

The notebook also documents Roboflow preprocessing (resize to 640×640) and rotation augmentation implications for leakage checks.

---

## Model Configuration (Notebook Default)

- **Backbone:** `yolo11s.pt` (COCO-pretrained)
- **Epochs:** 70
- **Batch size:** 8
- **Image size:** 640
- **Patience:** 15
- **Scheduler:** cosine LR (`cos_lr=True`)
- **Seed:** 42
- **Grayscale-aware augmentation choice:** `hsv_h=0.0`, `hsv_s=0.0`, keep brightness jitter (`hsv_v=0.3`)

---

## How to Run

### Option 1: Google Colab (recommended)
Open and run the notebook directly:
- `Breast_Lesion_Detection_YOLO11_.ipynb`

### Option 2: Local Jupyter
1. Create a Python environment (3.10+ recommended)
2. Install dependencies:
   ```bash
   pip install ultralytics pyyaml opencv-python matplotlib pandas torch
   ```
3. Open the notebook in Jupyter and update dataset paths as needed.

---

## Outputs You Can Expect

- Data audit tables and plots
- Training curves and confusion matrix
- PR/F1 confidence curves
- FROC curve with sensitivity at target FP/image levels
- Bootstrap confidence intervals
- Error-analysis visualizations and final qualitative predictions

---

## Intended Use

This project is for **research and education**.
It is **not** a certified medical device and must not be used as a standalone clinical decision system.

---

## Citation

If you use this repository, cite:
- The Roboflow dataset source (as documented in the notebook)
- Ultralytics YOLO framework
- Clinical CAD/FROC and bootstrap references listed in the notebook
