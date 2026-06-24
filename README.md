# Power Line and Transmission Tower Segmentation in Aerial Images

> Comparing Class-Imbalance-Aware Loss Functions for Thin-Structure Semantic Segmentation using the TTPLA Dataset

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Question](#research-question)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Setup and Installation](#setup-and-installation)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Visualizations](#visualizations)
9. [Team](#team)

---

## Project Overview

This project addresses the problem of automatically detecting power lines and transmission towers in aerial drone imagery using semantic segmentation. Manual inspection of aerial footage is slow, costly, and error-prone. An automated segmentation system could flag cable and tower locations directly from images, reducing the need for human review.

The core technical challenge is **class imbalance** — cable and tower pixels make up only **2.18%** of all pixels in the dataset, with a background-to-foreground ratio of **44.9:1**. Standard loss functions treat every pixel equally, causing models to ignore the rare foreground class entirely.

This project trains a fixed model architecture (**U-Net with a pretrained ResNet-34 encoder**) five separate times — each time with a different loss function — and compares the results to identify which loss function best handles this severe imbalance.

---

## Research Question

> *Which loss function — Binary Cross-Entropy (BCE), Weighted BCE, Focal Loss, Dice Loss, or Tversky Loss — produces the most accurate segmentation of power lines and transmission towers under a 45:1 class imbalance?*

---

## Dataset

- **Name:** TTPLA — Transmission Towers and Power Lines Aerial Dataset
- **Source:** [github.com/r3ab/ttpla_dataset](https://github.com/r3ab/ttpla_dataset)
- **Size:** 1,242 aerial images with LabelMe-style polygon annotations
- **Labels:** `cable`, `tower` (both treated as foreground in binary segmentation)
- **Class imbalance:**
  - Foreground (cable/tower): **2.18%** of pixels
  - Background: **97.82%** of pixels
  - Ratio: **44.9 : 1**
- **Data split** (random seed = 42):
  - Train: **869 images (70%)**
  - Validation: **186 images (15%)**
  - Test: **186 images (15%)**

> **Note:** The raw images and masks are not included in this repository due to file size constraints. Please download the dataset directly from the TTPLA GitHub link above.

---

## Project Structure

```
├── figures/
│   ├── 01_training_curve_bce.png
│   ├── 01_training_curve_weighted_bce.png
│   ├── 01_training_curve_focal.png
│   ├── 01_training_curve_dice.png
│   ├── 01_training_curve_tversky.png
│   ├── 02_combined_training_curves.png
│   ├── 03_test_metrics_bar_chart.png
│   ├── 04_radar_chart.png
│   ├── 05_precision_recall_scatter.png
│   ├── 06_metric_heatmap.png
│   └── 07_qualitative_predictions.png
│
├── python_files/
│   ├── convert_masks.py       # Step 1: Convert JSON annotations to binary masks
│   ├── explore_dataset.py     # Step 2: Visualise data and measure class imbalance
│   └── split_dataset.py       # Step 3: Split dataset into train/val/test
│
├── ttpla_training/
│   ├── dataset.py             # PyTorch Dataset class with augmentation
│   ├── model.py               # U-Net + ResNet-34 architecture
│   ├── losses.py              # All 5 loss functions
│   ├── metrics.py             # IoU, F1, Precision, Recall
│   ├── train.py               # Step 4: Training script
│   ├── evaluate.py            # Step 5: Test set evaluation
│   └── visualize_results.py   # Step 6: Generate all figures
│
├── results/
│   ├── run_bce/               # training_log.csv (best_model.pth not included)
│   ├── run_weighted_bce/      # training_log.csv
│   ├── run_focal/             # training_log.csv
│   ├── run_dice/              # training_log.csv
│   ├── run_tversky/           # training_log.csv
│   └── test_set_comparison.csv
│
├── splits/
│   ├── train.txt              # 869 image names
│   ├── val.txt                # 186 image names
│   └── test.txt               # 186 image names
│
├── Power_Line_Segmentation_Presentation.ipynb
├── requirements.txt
└── README.md
```

---

## Setup and Installation

**Prerequisites:** Python 3.10

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/Freddyx94/Power-Line-and-Transmission-Tower-Segmentation-in-Aerial-Images.git
cd [Power-Line-and-Transmission-Tower-Segmentation-in-Aerial-Images]
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Download the dataset:**

Download the TTPLA dataset from [github.com/r3ab/ttpla_dataset](https://github.com/r3ab/ttpla_dataset) and place the images and `.json` annotation files into a local folder (e.g. `data_original_size/`).

---

## How to Run

Run the following scripts in order to reproduce the full pipeline from scratch.

### Step 1 — Convert annotations to masks
```bash
py -3.10 python_files/convert_masks.py
```
Reads each `.json` annotation file and produces a binary `.png` mask per image.

---

### Step 2 — Explore the dataset
```bash
py -3.10 python_files/explore_dataset.py
```
Generates sample overlays and a class distribution chart. Confirms the 44.9:1 class imbalance.

---

### Step 3 — Split the dataset
```bash
py -3.10 python_files/split_dataset.py
```
Splits 1,242 images into train/val/test sets using random seed 42. Produces `splits/train.txt`, `val.txt`, `test.txt`.

---

### Step 4 — Train the model (one per loss function)
```bash
py -3.10 ttpla_training/train.py --loss bce
py -3.10 ttpla_training/train.py --loss weighted_bce
py -3.10 ttpla_training/train.py --loss focal
py -3.10 ttpla_training/train.py --loss dice
py -3.10 ttpla_training/train.py --loss tversky
```
Each run trains U-Net + ResNet-34 for 30 epochs and saves `best_model.pth` and `training_log.csv` to `results/run_{loss}/`.

> **Note:** Each run takes approximately 24 hours on CPU. GPU is strongly recommended.

To run all 5 sequentially in one command (PowerShell):
```bash
py -3.10 ttpla_training/train.py --loss bce; py -3.10 ttpla_training/train.py --loss weighted_bce; py -3.10 ttpla_training/train.py --loss focal; py -3.10 ttpla_training/train.py --loss dice; py -3.10 ttpla_training/train.py --loss tversky
```

---

### Step 5 — Evaluate on the test set
```bash
py -3.10 ttpla_training/evaluate.py
```
Loads all 5 saved models and evaluates them on the 186 held-out test images. Saves `results/test_set_comparison.csv`.

---

### Step 6 — Generate visualizations
```bash
py -3.10 ttpla_training/visualize_results.py
```
Generates all 11 figures and saves them to `figures/`.

---

### Step 7 — Run the presentation notebook
Open `Power_Line_Segmentation_Presentation.ipynb` in Jupyter Notebook or VS Code and run all cells from top to bottom.

---

## Results

Final test set results (186 held-out images never seen during training), ranked by IoU:

| Rank | Loss Function | IoU | F1 | Precision | Recall |
|------|---------------|-----|----|-----------|--------|
| 🥇 1 | **Dice** | **0.674** | **0.798** | 0.807 | 0.803 |
| 🥈 2 | **Tversky** | 0.659 | 0.786 | 0.738 | 0.857 |
| 🥉 3 | **BCE** (baseline) | 0.644 | 0.774 | 0.827 | 0.743 |
| 4 | **Focal** | 0.611 | 0.746 | 0.884 | 0.662 |
| 5 | **Weighted BCE** | 0.431 | 0.585 | 0.435 | 0.971 |

**Key findings:**
- **Dice Loss** achieved the best overall performance — highest IoU (0.674) and F1 (0.798) with a well-balanced precision/recall
- **Tversky Loss** achieved the highest recall (0.857) — consistent with its design penalising missed cables more than false detections
- **Weighted BCE performed worst** — a single fixed class weight overcorrected, causing the model to predict cable almost everywhere (precision = 0.435)
- Dice Loss outperforms the plain BCE baseline by **+4.7% IoU** and **+3.1% F1**

**Conclusion:** Segmentation-specific, imbalance-aware loss functions (Dice, Tversky) meaningfully outperform both standard and naively-weighted cross-entropy on thin-structure aerial segmentation tasks.

---

## Visualizations

### Combined Training Curves
![Combined Training Curves](figures/02_combined_training_curves.png)

### Final Test Metrics
![Test Metrics Bar Chart](figures/03_test_metrics_bar_chart.png)

### Performance Heatmap
![Heatmap](figures/06_metric_heatmap.png)

### Qualitative Predictions
![Qualitative Predictions](figures/07_qualitative_predictions.png)

---

## Team

| Name | Matriculation Number |
|------|----------------------|
| DAVID JOYSON AKKARAPATY | [90159624.] |
| OPEYEMI SAMUEL KOLADE | [Matric No.] |
| SAHEED YAKUBU | [59117739.] |

**Course:** MACHINE LEARNING  
**Institution:** UNIVERSITY OF EUROPE FOR APPLIED SCIENCES, POTSDAM 
**Date:** 29/07/2026

---

*Dataset: TTPLA — [github.com/r3ab/ttpla_dataset](https://github.com/r3ab/ttpla_dataset)*
