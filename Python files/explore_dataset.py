"""
TTPLA Dataset - Data Exploration Script
This script:
  1. Counts images and masks
  2. Calculates class imbalance (cable pixels vs background pixels)
  3. Saves overlay visualizations to verify masks look correct
"""

import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATASET_DIR  = r"C:\Users\lilfr\Downloads\data_original_size"
MASKS_DIR    = r"C:\Users\lilfr\Downloads\masks"
OUTPUT_DIR   = r"C:\Users\lilfr\Downloads\exploration_results"
# ──────────────────────────────────────────────────────────────────────────────

# How many sample images to visualize
NUM_SAMPLES = 6


def get_paired_files():
    """Return list of (image_path, mask_path) pairs that exist for both."""
    image_files = {os.path.splitext(f)[0]: f 
                   for f in os.listdir(DATASET_DIR) 
                   if f.lower().endswith((".jpg", ".png"))}
    
    mask_files  = {os.path.splitext(f)[0]: f 
                   for f in os.listdir(MASKS_DIR) 
                   if f.endswith(".png")}

    common = sorted(set(image_files.keys()) & set(mask_files.keys()))

    pairs = []
    for name in common:
        img_path  = os.path.join(DATASET_DIR, image_files[name])
        mask_path = os.path.join(MASKS_DIR,   mask_files[name])
        pairs.append((name, img_path, mask_path))

    return pairs


def calculate_class_imbalance(pairs):
    """Calculate overall foreground vs background pixel ratio."""
    print("\n Calculating class imbalance across all masks...")
    print("-" * 60)

    total_pixels      = 0
    total_foreground  = 0

    for name, img_path, mask_path in pairs:
        mask = np.array(Image.open(mask_path).convert("L"))
        fg   = np.sum(mask > 0)
        tot  = mask.size
        total_foreground += fg
        total_pixels     += tot

    total_background = total_pixels - total_foreground
    fg_pct = (total_foreground / total_pixels) * 100
    bg_pct = (total_background / total_pixels) * 100
    ratio  = total_background / total_foreground if total_foreground > 0 else float("inf")

    print(f"  Total images analysed  : {len(pairs)}")
    print(f"  Total pixels           : {total_pixels:,}")
    print(f"  Foreground pixels      : {total_foreground:,}  ({fg_pct:.2f}%)")
    print(f"  Background pixels      : {total_background:,}  ({bg_pct:.2f}%)")
    print(f"  Background : Foreground ratio  = {ratio:.1f} : 1")
    print("-" * 60)

    if ratio > 50:
        print("  SEVERE imbalance detected.")
        print("  Recommendation: Use Focal Loss or Dice Loss instead of BCE.")
    elif ratio > 10:
        print("  MODERATE imbalance detected.")
        print("  Recommendation: Use weighted BCE or Focal Loss.")
    else:
        print("  Mild imbalance. Weighted BCE may be sufficient.")

    print("-" * 60)
    return fg_pct, bg_pct, ratio


def save_overlay_samples(pairs, num_samples):
    """Save side-by-side visualizations: original | mask | overlay."""
    print(f"\n Saving {num_samples} sample visualizations...")

    samples = pairs[:num_samples]
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))

    for i, (name, img_path, mask_path) in enumerate(samples):
        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        # Create overlay: red tint on foreground pixels
        overlay = image.copy()
        fg_pixels = mask > 0
        overlay[fg_pixels, 0] = 255   # red channel
        overlay[fg_pixels, 1] = (overlay[fg_pixels, 1] * 0.3).astype(np.uint8)
        overlay[fg_pixels, 2] = (overlay[fg_pixels, 2] * 0.3).astype(np.uint8)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Original: {name}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Binary Mask", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay (red = cable/tower)", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sample_overlays.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_imbalance_chart(fg_pct, bg_pct):
    """Save a pie chart showing class distribution."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes  = [fg_pct, bg_pct]
    labels = [f"Foreground\n(cable/tower)\n{fg_pct:.2f}%",
              f"Background\n{bg_pct:.2f}%"]
    colors = ["#e74c3c", "#2ecc71"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.2f%%",
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("Pixel Class Distribution", fontsize=14, fontweight="bold")

    out_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Find paired files
    pairs = get_paired_files()
    print(f"\n Dataset Summary")
    print("=" * 60)
    print(f"  Images with matching masks: {len(pairs)}")

    if not pairs:
        print("No matching image/mask pairs found. Check your paths.")
        return

    # Step 2: Class imbalance
    fg_pct, bg_pct, ratio = calculate_class_imbalance(pairs)

    # Step 3: Visualizations
    save_overlay_samples(pairs, min(NUM_SAMPLES, len(pairs)))
    save_imbalance_chart(fg_pct, bg_pct)

    print(f"\n All results saved to: {OUTPUT_DIR}")
    print("  - sample_overlays.png   : verify your masks look correct")
    print("  - class_distribution.png: class imbalance pie chart")
    print("=" * 60)


if __name__ == "__main__":
    main()
