"""
TTPLA Dataset - Convert LabelMe JSON annotations to binary segmentation masks.
For each image, this script creates a corresponding .png mask where:
  - Pixel = 255 (white) → cable (foreground)
  - Pixel = 0   (black) → background
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw

# ─── CONFIGURE THESE PATHS ────────────────────────────────────────────────────
# Point these to your actual dataset folder
DATASET_DIR  = "./data_original_size"        # folder containing all .jpg and .json files
OUTPUT_DIR   = "./masks"          # folder where masks will be saved
# ──────────────────────────────────────────────────────────────────────────────

# Labels we want to segment
TARGET_LABELS = {"cable"}


def convert_json_to_mask(json_path: str, output_dir: str) -> None:
    """Read one LabelMe .json file and save a binary mask as a .png."""

    with open(json_path, "r") as f:
        data = json.load(f)

    img_h = data["imageHeight"]
    img_w = data["imageWidth"]

    # Create a blank (all-black) mask
    mask = Image.new("L", (img_w, img_h), color=0)
    draw = ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        label = shape["label"].lower()
        if label not in TARGET_LABELS:
            continue  # skip labels we don't care about

        # points is a list of [x, y] pairs → convert to list of (x, y) tuples
        polygon = [tuple(pt) for pt in shape["points"]]

        if len(polygon) < 3:
            continue  # a valid polygon needs at least 3 points

        # Fill the polygon region with white (255 = foreground)
        draw.polygon(polygon, fill=255)

    # Save the mask with the same base name as the json, but as .png
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    out_path  = os.path.join(output_dir, base_name + ".png")
    mask.save(out_path)

    return base_name, img_h, img_w, len(data["shapes"])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".json")]
    json_files.sort()

    if not json_files:
        print(f"No .json files found in: {DATASET_DIR}")
        return

    print(f"Found {len(json_files)} annotation files. Converting...")
    print("-" * 60)

    total_shapes = 0
    for fname in json_files:
        json_path = os.path.join(DATASET_DIR, fname)
        try:
            base, h, w, n_shapes = convert_json_to_mask(json_path, OUTPUT_DIR)
            total_shapes += n_shapes
            print(f"  ✓ {base}.png  [{w}×{h}]  shapes: {n_shapes}")
        except Exception as e:
            print(f"  ✗ {fname}  ERROR: {e}")

    print("-" * 60)
    print(f"Done! Masks saved to: {OUTPUT_DIR}")
    print(f"Total annotations converted: {total_shapes}")


if __name__ == "__main__":
    main()
