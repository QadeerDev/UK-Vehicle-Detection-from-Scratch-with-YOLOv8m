# 🇬🇧 UK Vehicle Detection — YOLOv8m Trained from Scratch

> **30-Day Computer Vision Portfolio · Day 2**
> *Muhammad Qadeer — AI/ML Engineer · [GitHub: QadeerDev](https://github.com/QadeerDev)*

---

## Overview

End-to-end object detection pipeline trained on UK traffic footage — covering data ingestion, exploratory analysis, domain-specific augmentation, model training, evaluation, and video inference.

**Classes detected:** `Car` · `Bus` · `Truck` · `Bicycle`

| Detail | Value |
|--------|-------|
| Model | YOLOv8m (fine-tuned from COCO) |
| Dataset | Roboflow Universe — UK Traffic (YOLO format) |
| Training Platform | Kaggle Notebooks (NVIDIA P100 16GB) |
| Epochs | 50 |
| Input Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | AdamW |

---

## Project Structure

```
Day-2-UK-Vehicle-Detection/
├── Input/                         # Raw dataset (not tracked by git)
│   └── united-kingdom-dataset/
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
├── Model Weights/                 # best.pt saved here after training
├── notebooks/
│   └── day2-uk-vehicle-fromscratch.ipynb
└── Output/
    ├── eda_class_distribution.png
    ├── eda_image_stats.png
    ├── eda_bbox_analysis.png
    ├── eda_sample_images.png
    ├── training_curves.png
    ├── evaluation_plots.png
    ├── val_predictions.png
    ├── inference_stats.png
    ├── inference_log.csv
    └── uk_inference_output.mp4
```

---

## Pipeline Phases

### Phase 1 — Environment & Dataset Setup
- Installs `ultralytics`, `roboflow`, and `supervision`
- Downloads dataset via Roboflow API in YOLOv8 format
- Verifies folder structure and counts images/labels per split
- GPU check: confirms CUDA availability and VRAM

### Phase 2 — Exploratory Data Analysis
- Parses all YOLO `.txt` label files into a unified DataFrame
- **Class distribution** bar charts per split (train / valid / test)
- **Image resolution** histogram — width, height, aspect ratio
- **Bounding box analysis** — size distribution, position heatmap, co-occurrence matrix
- Class imbalance ratio calculation with warning threshold

### Phase 3 — Preprocessing & Augmentation Config

UK-specific augmentation decisions are documented in the notebook:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `fliplr` | `0.0` | UK is left-hand drive — horizontal flip corrupts lane geometry |
| `degrees` | `0.0` | Fixed CCTV angles — rotation adds noise, not generalisation |
| `hsv_v` | `0.4` | UK weather ranges from bright sun to heavy overcast |
| `mosaic` | `1.0` | Strong context augmentation for multi-vehicle scenes |
| `mixup` | `0.1` | Mild blending for boundary robustness |
| `copy_paste` | `0.1` | Adds rare-class samples (Bicycle, Bus) into dense scenes |

### Phase 4 — Training
- Loads `yolov8m.pt` (COCO pretrained backbone)
- Trains with AdamW, `lr0=0.001`, cosine LR schedule (`lrf=0.01`)
- 3-epoch warmup, 50 total epochs
- Saves best checkpoint by `mAP@50`
- Plots training curves: box loss, class loss, mAP@50, mAP@50:95, precision, recall

### Phase 5 — Evaluation
- Runs `model.val()` on the held-out test split
- Metrics reported: `mAP@50`, `mAP@50:95`, Precision, Recall
- Per-class `AP@50` breakdown
- Renders confusion matrix (normalised), PR curve, F1 curve
- Displays 8 validation sample predictions with detection counts

### Phase 6 — Inference on UK Road Footage
- Runs best weights on 4K London road video (30 fps)
- Annotates each frame using `supervision` BoxAnnotator + LabelAnnotator
- Logs every detection (frame, class, confidence) to `inference_log.csv`
- Plots post-inference stats:
  - Detections per class (bar chart)
  - Confidence score distribution per class
  - Detection timeline across all frames

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/QadeerDev/30-day-cv-portfolio.git
cd 30-day-cv-portfolio/Day-2-UK-Vehicle-Detection
pip install ultralytics roboflow supervision
```

### 2. Download Dataset

Get a free API key from [roboflow.com](https://roboflow.com) and run:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("uk-traffic-vehicle")
dataset = project.version(1).download("yolov8")
```

### 3. Run Notebook

Open `notebooks/day2-uk-vehicle-fromscratch.ipynb` in Kaggle or Jupyter and execute cells top-to-bottom.

### 4. Inference on Your Own Video

```python
from ultralytics import YOLO
model = YOLO("Model Weights/best.pt")
model.predict("your_video.mp4", conf=0.35, save=True)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| mAP@50 | _(see Output/evaluation_plots.png)_ |
| mAP@50:95 | _(see Output/evaluation_plots.png)_ |
| Best Epoch | _(logged in training_curves.png)_ |

> Results depend on the specific Roboflow dataset version used. Refer to the output plots for exact numbers from your training run.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-ff6600)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Supervision](https://img.shields.io/badge/Supervision-0.x-blueviolet)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle_P100-20BEFF)

---

## 30-Day CV Portfolio

| Day | Project | Model |
|-----|---------|-------|
| Day 1 | ConstructionGuard-AI — Helmet Detection | YOLOv8n |
| **Day 2** | **UK Vehicle Detection & Tracking** | **YOLOv8m** |
| Day 3 | Face Mask Detection | MobileNetV2 |
| Day 4 | Gun Detection for Surveillance | YOLOv8s |
| Day 5 | Multi-Class Grocery Counter | YOLOv8m |
| ... | ... | ... |

---

## Author

**Muhammad Qadeer**
AI/ML Engineer · Computer Vision Research Assistant · GIFT University

[![GitHub](https://img.shields.io/badge/GitHub-QadeerDev-black?logo=github)](https://github.com/QadeerDev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-qadeerjutt-blue?logo=linkedin)](https://linkedin.com/in/qadeerjutt)

---

*Part of the 30-Day Computer Vision Portfolio Challenge — building and shipping one real CV project every day.*
