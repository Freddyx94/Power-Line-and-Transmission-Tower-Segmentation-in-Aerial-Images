"""
TTPLA Dataset class - loads images and masks for training.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class TTPLADataset(Dataset):
    """
    Loads paired images and masks from the TTPLA dataset.
    Each item returns a (image_tensor, mask_tensor) pair.
    """

    def __init__(self, split_file, images_dir, masks_dir,
                 image_size=(512, 512), augment=False):
        """
        Args:
            split_file  : path to train.txt / val.txt / test.txt
            images_dir  : folder containing original images
            masks_dir   : folder containing binary mask .png files
            image_size  : resize all images to this (H, W)
            augment     : if True, apply data augmentation (train only)
        """
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.image_size = image_size
        self.augment    = augment

        # Read image names from the split file
        with open(split_file, "r") as f:
            self.names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.names)

    def _find_image(self, name):
        """Find the image file - tries .jpg and .png extensions."""
        for ext in [".jpg", ".JPG", ".png", ".PNG"]:
            path = os.path.join(self.images_dir, name + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No image found for: {name}")

    def _augment(self, image, mask):
        """Apply random augmentations to image and mask together."""

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask, angle)

        # Random brightness and contrast (image only, not mask)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))

        return image, mask

    def __getitem__(self, idx):
        name = self.names[idx]

        # Load image and mask
        img_path  = self._find_image(name)
        mask_path = os.path.join(self.masks_dir, name + ".png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")   # grayscale

        # Resize to fixed size
        image = TF.resize(image, self.image_size)
        mask  = TF.resize(mask,  self.image_size,
                          interpolation=TF.InterpolationMode.NEAREST)

        # Apply augmentation (training only)
        if self.augment:
            image, mask = self._augment(image, mask)

        # Convert to tensors
        image = TF.to_tensor(image)          # shape: [3, H, W], range [0, 1]
        mask  = torch.from_numpy(
                    np.array(mask)
                ).float() / 255.0            # shape: [H, W], range [0, 1]
        mask  = mask.unsqueeze(0)            # shape: [1, H, W]

        # Normalize image with ImageNet mean and std (for pretrained backbone)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        return image, mask, name
