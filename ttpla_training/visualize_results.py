"""
TTPLA Thesis Visualization Suite
Generates every figure needed for the Results chapter:

  1. Training curves (loss, IoU) per loss function
  2. Combined training curves (all 5 losses overlaid)
  3. Final test metrics - grouped bar chart
  4. Radar / spider chart comparing all metrics
  5. Precision-Recall trade-off scatter plot
  6. Ranking heatmap (metric x loss function)
  7. Qualitative predictions: image | ground truth | prediction overlay

Usage:
    python visualize_results.py
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from dataset import TTPLADataset
from model   import get_model

# ─── PATHS ────────────────────────────────────────────────────────────────────
IMAGES_DIR   = r"C:\Users\lilfr\Downloads\data_original_size"
MASKS_DIR    = r"C:\Users\lilfr\Downloads\masks"
SPLITS_DIR   = r"C:\Users\lilfr\Downloads\splits"
RESULTS_DIR  = r"C:\Users\lilfr\Downloads\results"
OUTPUT_DIR   = r"C:\Users\lilfr\Downloads\thesis_figures"
TEST_CSV     = os.path.join(RESULTS_DIR, "test_set_comparison.csv")
# ──────────────────────────────────────────────────────────────────────────────

LOSS_NAMES = ["bce", "weighted_bce", "focal", "dice", "tversky"]
COLORS = {
    "bce":          "#73726c",
    "weighted_bce": "#378ADD",
    "focal":        "#D85A30",
    "dice":         "#1D9E75",
    "tversky":      "#7F77DD",
}
LABELS = {
    "bce":          "BCE",
    "weighted_bce": "Weighted BCE",
    "focal":        "Focal",
    "dice":         "Dice",
    "tversky":      "Tversky",
}

IMAGE_SIZE = (512, 512)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_training_log(loss_name):
    """Load training_log.csv for a given loss function."""
    path = os.path.join(RESULTS_DIR, f"run_{loss_name}", "training_log.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Missing training log for {loss_name}: {path}")
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "epoch":         [int(r["epoch"]) for r in rows],
        "train_loss":    [float(r["train_loss"]) for r in rows],
        "train_iou":     [float(r["train_iou"]) for r in rows],
        "val_loss":      [float(r["val_loss"]) for r in rows],
        "val_iou":       [float(r["val_iou"]) for r in rows],
        "val_f1":        [float(r["val_f1"]) for r in rows],
        "val_precision": [float(r["val_precision"]) for r in rows],
        "val_recall":    [float(r["val_recall"]) for r in rows],
    }


def load_test_results():
    """Load the final test_set_comparison.csv."""
    if not os.path.exists(TEST_CSV):
        print(f"  [WARN] Missing test results: {TEST_CSV}")
        return {}
    with open(TEST_CSV) as f:
        rows = list(csv.DictReader(f))
    return {r["loss_function"]: r for r in rows}


# ═══════════════════════════════════════════════════════════════════════════
# 1. INDIVIDUAL TRAINING CURVES (one figure per loss function)
# ═══════════════════════════════════════════════════════════════════════════

def plot_individual_training_curves(logs):
    print("\n[1/7] Plotting individual training curves...")
    for loss_name, log in logs.items():
        if log is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{LABELS[loss_name]} Loss — Training Curves", fontsize=14, fontweight="bold")

        axes[0].plot(log["epoch"], log["train_loss"], "o-", color=COLORS[loss_name], label="Train Loss")
        axes[0].plot(log["epoch"], log["val_loss"], "o--", color=COLORS[loss_name], alpha=0.6, label="Val Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(log["epoch"], log["train_iou"], "o-", color=COLORS[loss_name], label="Train IoU")
        axes[1].plot(log["epoch"], log["val_iou"], "o--", color=COLORS[loss_name], alpha=0.6, label="Val IoU")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("IoU")
        axes[1].set_title("IoU"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"01_training_curve_{loss_name}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. COMBINED TRAINING CURVES (all 5 overlaid)
# ═══════════════════════════════════════════════════════════════════════════

def plot_combined_training_curves(logs):
    print("\n[2/7] Plotting combined training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Loss Function Comparison — Validation Metrics over Training", fontsize=15, fontweight="bold")

    metric_pairs = [("val_iou", "Validation IoU"), ("val_f1", "Validation F1"),
                    ("val_precision", "Validation Precision"), ("val_recall", "Validation Recall")]

    for ax, (metric, title) in zip(axes.flat, metric_pairs):
        for loss_name, log in logs.items():
            if log is None:
                continue
            ax.plot(log["epoch"], log[metric], label=LABELS[loss_name],
                    color=COLORS[loss_name], linewidth=2)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(title.split()[-1])
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "02_combined_training_curves.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. FINAL TEST METRICS - GROUPED BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_test_bar_chart(test_results):
    print("\n[3/7] Plotting final test metrics bar chart...")
    if not test_results:
        return

    ranked = sorted(test_results.items(), key=lambda x: float(x[1]["test_iou"]), reverse=True)
    names  = [LABELS[n] for n, _ in ranked]
    iou       = [float(r["test_iou"]) for _, r in ranked]
    f1        = [float(r["test_f1"]) for _, r in ranked]
    precision = [float(r["test_precision"]) for _, r in ranked]
    recall    = [float(r["test_recall"]) for _, r in ranked]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = range(len(names)); width = 0.2

    ax.bar([i - 1.5*width for i in x], iou,       width, label="IoU",       color="#1D9E75")
    ax.bar([i - 0.5*width for i in x], f1,        width, label="F1 Score",  color="#378ADD")
    ax.bar([i + 0.5*width for i in x], precision, width, label="Precision", color="#D85A30")
    ax.bar([i + 1.5*width for i in x], recall,    width, label="Recall",    color="#BA7517")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Final Test Set Results — Loss Function Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "03_test_metrics_bar_chart.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. RADAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def radar_factory(num_vars):
    """Create a radar chart projection (matplotlib recipe)."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            return super()._gen_axes_spines()

    register_projection(RadarAxes)
    return theta


def plot_radar_chart(test_results):
    print("\n[4/7] Plotting radar chart...")
    if not test_results:
        return

    categories = ["IoU", "F1", "Precision", "Recall"]
    theta = radar_factory(len(categories))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="radar"))

    for loss_name in LOSS_NAMES:
        if loss_name not in test_results:
            continue
        r = test_results[loss_name]
        values = [float(r["test_iou"]), float(r["test_f1"]),
                  float(r["test_precision"]), float(r["test_recall"])]
        ax.plot(theta, values, color=COLORS[loss_name], linewidth=2, label=LABELS[loss_name])
        ax.fill(theta, values, color=COLORS[loss_name], alpha=0.08)

    ax.set_varlabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Comparison Across Loss Functions", fontsize=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "04_radar_chart.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PRECISION-RECALL TRADE-OFF SCATTER
# ═══════════════════════════════════════════════════════════════════════════

def plot_precision_recall_scatter(test_results):
    print("\n[5/7] Plotting precision-recall trade-off...")
    if not test_results:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for loss_name in LOSS_NAMES:
        if loss_name not in test_results:
            continue
        r = test_results[loss_name]
        p = float(r["test_precision"]); rec = float(r["test_recall"])
        iou = float(r["test_iou"])
        ax.scatter(rec, p, s=iou * 1500, color=COLORS[loss_name],
                  alpha=0.7, edgecolors="black", linewidth=1.2, label=LABELS[loss_name])
        ax.annotate(LABELS[loss_name], (rec, p), textcoords="offset points",
                   xytext=(10, 8), fontsize=10, fontweight="bold")

    # Diagonal F1 reference line
    x = np.linspace(0.01, 1, 100)
    for f1_target in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y = (f1_target * x) / (2*x - f1_target)
        y = np.clip(y, 0, 1)
        valid = (y > 0) & (y <= 1)
        ax.plot(x[valid], y[valid], "--", color="gray", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Trade-off\n(bubble size = IoU)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "05_precision_recall_scatter.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. RANKING HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_heatmap(test_results):
    print("\n[6/7] Plotting metric heatmap...")
    if not test_results:
        return

    metrics = ["test_iou", "test_f1", "test_precision", "test_recall"]
    metric_labels = ["IoU", "F1", "Precision", "Recall"]

    ranked = sorted(test_results.items(), key=lambda x: float(x[1]["test_iou"]), reverse=True)
    names = [LABELS[n] for n, _ in ranked]

    data = np.array([[float(r[m]) for m in metrics] for _, r in ranked])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_labels))); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=11)

    for i in range(len(names)):
        for j in range(len(metric_labels)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                   color="black", fontsize=10, fontweight="bold")

    ax.set_title("Performance Heatmap — Test Set Metrics", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Score")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "06_metric_heatmap.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. QUALITATIVE PREDICTIONS (image | ground truth | predictions from all models)
# ═══════════════════════════════════════════════════════════════════════════

def plot_qualitative_predictions(num_samples=4):
    print("\n[7/7] Generating qualitative prediction comparisons...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TTPLADataset(
        split_file=os.path.join(SPLITS_DIR, "test.txt"),
        images_dir=IMAGES_DIR, masks_dir=MASKS_DIR,
        image_size=IMAGE_SIZE, augment=False,
    )

    # Load all available models
    models = {}
    for loss_name in LOSS_NAMES:
        model_path = os.path.join(RESULTS_DIR, f"run_{loss_name}", "best_model.pth")
        if not os.path.exists(model_path):
            continue
        m = get_model(encoder="resnet34", pretrained=False).to(device)
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.eval()
        models[loss_name] = m

    if not models:
        print("  [WARN] No models found, skipping qualitative comparison.")
        return

    # ImageNet normalization constants for un-normalizing the image for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    n_cols = 2 + len(models)  # original + GT + one per model
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(3.2 * n_cols, 3.2 * num_samples))

    # Pick evenly spaced sample indices
    indices = np.linspace(0, len(test_dataset) - 1, num_samples, dtype=int)

    col_titles = ["Image", "Ground Truth"] + [LABELS[n] for n in models.keys()]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask, name = test_dataset[idx]

            # Un-normalize image for display
            img_disp = image.permute(1, 2, 0).numpy()
            img_disp = (img_disp * std + mean).clip(0, 1)

            axes[row, 0].imshow(img_disp)
            axes[row, 0].set_ylabel(name, fontsize=8)
            axes[row, 1].imshow(mask.squeeze(0).numpy(), cmap="gray")

            input_tensor = image.unsqueeze(0).to(device)
            for col, (loss_name, m) in enumerate(models.items(), start=2):
                output = m(input_tensor)
                pred = (torch.sigmoid(output) > 0.5).float().cpu().squeeze().numpy()
                axes[row, col].imshow(pred, cmap="gray")

            for col in range(n_cols):
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(col_titles[col], fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "07_qualitative_predictions.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*70}")
    print("  TTPLA Thesis Visualization Suite")
    print(f"  Output folder: {OUTPUT_DIR}")
    print(f"{'='*70}")

    # Load all training logs
    logs = {name: load_training_log(name) for name in LOSS_NAMES}
    test_results = load_test_results()

    plot_individual_training_curves(logs)
    plot_combined_training_curves(logs)
    plot_test_bar_chart(test_results)
    plot_radar_chart(test_results)
    plot_precision_recall_scatter(test_results)
    plot_heatmap(test_results)
    plot_qualitative_predictions(num_samples=4)

    print(f"\n{'='*70}")
    print(f"  All visualizations saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
