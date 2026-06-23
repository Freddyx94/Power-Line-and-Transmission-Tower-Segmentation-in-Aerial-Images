"""
TTPLA Test Set Evaluation Script
Loads each trained model and evaluates it on the held-out TEST set
(186 images the models have never seen during training or validation).

This produces the final, unbiased results table for your thesis.

Usage:
    python evaluate.py
"""

import os
import csv
import torch
from torch.utils.data import DataLoader

from dataset import TTPLADataset
from model   import get_model
from metrics import compute_metrics

# ─── PATHS ────────────────────────────────────────────────────────────────────
IMAGES_DIR  = r"C:\Users\lilfr\Downloads\data_original_size"
MASKS_DIR   = r"C:\Users\lilfr\Downloads\masks"
SPLITS_DIR  = r"C:\Users\lilfr\Downloads\splits"
RESULTS_DIR = r"C:\Users\lilfr\Downloads\results"
OUTPUT_CSV  = r"C:\Users\lilfr\Downloads\results\test_set_comparison.csv"
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE  = (512, 512)
BATCH_SIZE  = 4
NUM_WORKERS = 0

# Map: loss function name -> path to its best_model.pth
# Update these paths if your folder names are different
MODEL_PATHS = {
    "bce":          os.path.join(RESULTS_DIR, "run_bce",          "best_model.pth"),
    "weighted_bce": os.path.join(RESULTS_DIR, "run_weighted_bce", "best_model.pth"),
    "focal":        os.path.join(RESULTS_DIR, "run_focal",        "best_model.pth"),
    "dice":         os.path.join(RESULTS_DIR, "run_dice",         "best_model.pth"),
    "tversky":      os.path.join(RESULTS_DIR, "run_tversky",      "best_model.pth"),
}


def evaluate_model(model, loader, device):
    """Run model on the test set and return averaged metrics."""
    model.eval()
    total_iou       = 0.0
    total_f1        = 0.0
    total_precision = 0.0
    total_recall    = 0.0
    n_batches       = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            metrics = compute_metrics(outputs, masks)

            total_iou       += metrics["iou"]
            total_f1        += metrics["f1"]
            total_precision += metrics["precision"]
            total_recall    += metrics["recall"]
            n_batches       += 1

    return {
        "iou"       : total_iou       / n_batches,
        "f1"        : total_f1        / n_batches,
        "precision" : total_precision / n_batches,
        "recall"    : total_recall    / n_batches,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  TTPLA Test Set Evaluation")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    # ── Test dataset (loaded once, reused for every model) ───────────────────
    test_dataset = TTPLADataset(
        split_file = os.path.join(SPLITS_DIR, "test.txt"),
        images_dir = IMAGES_DIR,
        masks_dir  = MASKS_DIR,
        image_size = IMAGE_SIZE,
        augment    = False,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    print(f"  Test samples: {len(test_dataset)}\n")

    # ── Evaluate each model ───────────────────────────────────────────────────
    results = {}

    for loss_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"  [SKIP] {loss_name}: model file not found at {model_path}")
            continue

        print(f"  Evaluating: {loss_name.upper()} ...")

        # Build model and load trained weights
        model = get_model(encoder="resnet34", pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Run on test set
        metrics = evaluate_model(model, test_loader, device)
        results[loss_name] = metrics

        print(f"    IoU: {metrics['iou']:.4f}  "
              f"F1: {metrics['f1']:.4f}  "
              f"Precision: {metrics['precision']:.4f}  "
              f"Recall: {metrics['recall']:.4f}\n")

        # Free memory before loading next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not results:
        print("  No models were found to evaluate. Check MODEL_PATHS.")
        return

    # ── Save comparison table ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss_function", "test_iou", "test_f1",
                         "test_precision", "test_recall"])
        for loss_name, m in results.items():
            writer.writerow([loss_name, f"{m['iou']:.6f}", f"{m['f1']:.6f}",
                             f"{m['precision']:.6f}", f"{m['recall']:.6f}"])

    # ── Print final ranked summary ────────────────────────────────────────────
    print(f"{'='*70}")
    print("  FINAL TEST SET RESULTS (ranked by IoU)")
    print(f"{'='*70}")
    ranked = sorted(results.items(), key=lambda x: x[1]["iou"], reverse=True)
    print(f"  {'Rank':<6}{'Loss':<16}{'IoU':<10}{'F1':<10}{'Precision':<12}{'Recall':<10}")
    print(f"  {'-'*64}")
    for i, (name, m) in enumerate(ranked, 1):
        print(f"  {i:<6}{name:<16}{m['iou']:<10.4f}{m['f1']:<10.4f}"
              f"{m['precision']:<12.4f}{m['recall']:<10.4f}")

    print(f"\n  Results saved to: {OUTPUT_CSV}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
