"""
TTPLA Training Script
Trains a U-Net segmentation model with a chosen loss function.

Usage:
    python train.py --loss focal
    python train.py --loss dice
    python train.py --loss bce
    python train.py --loss weighted_bce
    python train.py --loss tversky
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TTPLADataset
from model   import get_model, count_parameters
from losses  import get_loss
from metrics import compute_metrics

# ─── PATHS ────────────────────────────────────────────────────────────────────
IMAGES_DIR  = r"C:\Users\lilfr\Downloads\data_original_size"
MASKS_DIR   = r"C:\Users\lilfr\Downloads\masks"
SPLITS_DIR  = r"C:\Users\lilfr\Downloads\splits"
RESULTS_DIR = r"C:\Users\lilfr\Downloads\results"
# ──────────────────────────────────────────────────────────────────────────────

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
IMAGE_SIZE   = (512, 512)   # resize all images to this
BATCH_SIZE   = 4            # reduce to 2 if you run out of memory
NUM_EPOCHS   = 30
LEARNING_RATE = 1e-4
NUM_WORKERS  = 0            # set to 0 on Windows to avoid multiprocessing errors
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0

    for batch_idx, (images, masks, _) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        metrics     = compute_metrics(outputs.detach(), masks)
        total_loss += loss.item()
        total_iou  += metrics["iou"]

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}]  "
                  f"Loss: {loss.item():.4f}  IoU: {metrics['iou']:.4f}")

    avg_loss = total_loss / len(loader)
    avg_iou  = total_iou  / len(loader)
    return avg_loss, avg_iou


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss      = 0.0
    total_iou       = 0.0
    total_f1        = 0.0
    total_precision = 0.0
    total_recall    = 0.0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss    = criterion(outputs, masks)
            metrics = compute_metrics(outputs, masks)

            total_loss      += loss.item()
            total_iou       += metrics["iou"]
            total_f1        += metrics["f1"]
            total_precision += metrics["precision"]
            total_recall    += metrics["recall"]

    n = len(loader)
    return {
        "loss"      : total_loss      / n,
        "iou"       : total_iou       / n,
        "f1"        : total_f1        / n,
        "precision" : total_precision / n,
        "recall"    : total_recall    / n,
    }


def main(args):
    # ── Setup ──────────────────────────────────────────────────────────────
    loss_name  = args.loss
    run_dir    = os.path.join(RESULTS_DIR, f"run_{loss_name}")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training with loss : {loss_name.upper()}")
    print(f"  Device             : {device}")
    print(f"  Results saved to   : {run_dir}")
    print(f"{'='*60}\n")

    # ── Datasets & Loaders ─────────────────────────────────────────────────
    train_dataset = TTPLADataset(
        split_file = os.path.join(SPLITS_DIR, "train.txt"),
        images_dir = IMAGES_DIR,
        masks_dir  = MASKS_DIR,
        image_size = IMAGE_SIZE,
        augment    = True,
    )
    val_dataset = TTPLADataset(
        split_file = os.path.join(SPLITS_DIR, "val.txt"),
        images_dir = IMAGES_DIR,
        masks_dir  = MASKS_DIR,
        image_size = IMAGE_SIZE,
        augment    = False,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Epochs        : {NUM_EPOCHS}\n")

    # ── Model, Loss, Optimizer ─────────────────────────────────────────────
    model     = get_model(encoder="resnet34", pretrained=True).to(device)
    criterion = get_loss(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    print("  Model: U-Net with ResNet-34 encoder")
    count_parameters(model)
    print()

    # ── Training Loop ──────────────────────────────────────────────────────
    best_iou    = 0.0
    log_path    = os.path.join(run_dir, "training_log.csv")
    best_path   = os.path.join(run_dir, "best_model.pth")

    # Write CSV header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_iou,val_loss,val_iou,"
                "val_f1,val_precision,val_recall\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        print(f"  Train  -> Loss: {train_loss:.4f}  IoU: {train_iou:.4f}")
        print(f"  Val    -> Loss: {val_metrics['loss']:.4f}  "
              f"IoU: {val_metrics['iou']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}  "
              f"Precision: {val_metrics['precision']:.4f}  "
              f"Recall: {val_metrics['recall']:.4f}")
        print(f"  Time: {elapsed:.1f}s\n")

        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), best_path)
            print(f"  *** New best IoU: {best_iou:.4f} - model saved ***\n")

        # Learning rate scheduler step
        scheduler.step(val_metrics["iou"])

        # Log to CSV
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_iou:.6f},"
                    f"{val_metrics['loss']:.6f},{val_metrics['iou']:.6f},"
                    f"{val_metrics['f1']:.6f},{val_metrics['precision']:.6f},"
                    f"{val_metrics['recall']:.6f}\n")

    print(f"{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val IoU : {best_iou:.4f}")
    print(f"  Best model   : {best_path}")
    print(f"  Training log : {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TTPLA segmentation model")
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["bce", "weighted_bce", "focal", "dice", "tversky"],
                        help="Loss function to use")
    args = parser.parse_args()
    main(args)
