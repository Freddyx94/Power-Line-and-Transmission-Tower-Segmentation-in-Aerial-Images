"""
TTPLA Dataset - Convert LabelMe JSON annotations to binary segmentation masks.
For each image, this script creates a corresponding .png mask where:
  - Pixel = 255 (white) → cable or tower (foreground)
  - Pixel = 0   (black) → background
"""

import os
import json
from PIL import Image, ImageDraw

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\lilfr\Downloads\data_original_size"
OUTPUT_DIR  = r"C:\Users\lilfr\Downloads\masks"
# ──────────────────────────────────────────────────────────────────────────────

# Labels we want to segment
TARGET_LABELS = {"cable", "tower"}


def convert_json_to_mask(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    img_h = data["imageHeight"]
    img_w = data["imageWidth"]

    # Create a blank black mask
    mask = Image.new("L", (img_w, img_h), color=0)
    draw = ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        label = shape["label"].lower()
        if label not in TARGET_LABELS:
            continue

        # Convert [x, y] pairs to (x, y) tuples
        polygon = [tuple(pt) for pt in shape["points"]]

        if len(polygon) < 3:
            continue

        # Fill the polygon with white (255 = foreground)
        draw.polygon(polygon, fill=255)

    # Save mask as .png with same base name as the json file
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    out_path  = os.path.join(OUTPUT_DIR, base_name + ".png")
    mask.save(out_path)

    return base_name, img_h, img_w, len(data["shapes"])


def main():
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all json files in the dataset folder
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
            base, h, w, n_shapes = convert_json_to_mask(json_path)
            total_shapes += n_shapes
            print(f"  OK  {base}.png  [{w}x{h}]  shapes: {n_shapes}")
        except Exception as e:
            print(f"  ERROR  {fname}  :  {e}")

    print("-" * 60)
    print(f"Done! Masks saved to: {OUTPUT_DIR}")
    print(f"Total annotations converted: {total_shapes}")


if __name__ == "__main__":
    main()
