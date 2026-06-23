"""
TTPLA Dataset - Split into Train / Validation / Test sets
Produces three text files listing the image names for each split.
Split ratio: 70% train, 15% validation, 15% test
"""

import os
import random

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\lilfr\Downloads\data_original_size"
MASKS_DIR   = r"C:\Users\lilfr\Downloads\masks"
OUTPUT_DIR  = r"C:\Users\lilfr\Downloads\splits"
# ──────────────────────────────────────────────────────────────────────────────

# Split ratios (must add up to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Random seed for reproducibility (keep this fixed so splits are always the same)
RANDOM_SEED = 42


def get_paired_names():
    """Return sorted list of base names that have both an image and a mask."""
    image_names = {os.path.splitext(f)[0]
                   for f in os.listdir(DATASET_DIR)
                   if f.lower().endswith((".jpg", ".png"))}

    mask_names  = {os.path.splitext(f)[0]
                   for f in os.listdir(MASKS_DIR)
                   if f.endswith(".png")}

    paired = sorted(image_names & mask_names)
    return paired


def split_names(names):
    """Shuffle and split names into train/val/test."""
    random.seed(RANDOM_SEED)
    shuffled = names.copy()
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    return train, val, test


def save_split(names, filename):
    """Save a list of names to a text file, one per line."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w") as f:
        for name in sorted(names):
            f.write(name + "\n")
    print(f"  Saved: {out_path}  ({len(names)} images)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Get all valid paired names
    names = get_paired_names()
    print(f"\n Dataset Split")
    print("=" * 60)
    print(f"  Total paired images: {len(names)}")

    if not names:
        print("  No paired images found. Check your paths.")
        return

    # Step 2: Split
    train, val, test = split_names(names)

    print(f"\n  Split ratio: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print(f"  Train : {len(train)} images")
    print(f"  Val   : {len(val)}   images")
    print(f"  Test  : {len(test)}  images")
    print("-" * 60)

    # Step 3: Save to text files
    print("\n  Saving split files...")
    save_split(train, "train.txt")
    save_split(val,   "val.txt")
    save_split(test,  "test.txt")

    print("\n  Done! Split files saved to:", OUTPUT_DIR)
    print("  These files will be used by the training script to")
    print("  load the correct images for each phase.")
    print("=" * 60)


if __name__ == "__main__":
    main()
