"""
U-Net with pretrained ResNet-34 encoder backbone.
Uses segmentation-models-pytorch library.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(encoder="resnet34", pretrained=True):
    """
    Returns a U-Net with a pretrained ResNet-34 encoder.

    Args:
        encoder   : backbone name (resnet34 is a good default)
        pretrained: if True, load ImageNet pretrained weights

    Returns:
        model: nn.Module ready for training
    """
    weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = weights,
        in_channels     = 3,       # RGB images
        classes         = 1,       # binary: cable or not
        activation      = None,    # raw logits (loss handles sigmoid)
    )

    return model


def count_parameters(model):
    """Print number of trainable parameters."""
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
