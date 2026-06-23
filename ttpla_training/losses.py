"""
Loss functions for segmentation with class imbalance.
Implements: BCE, Weighted BCE, Focal Loss, Dice Loss, Tversky Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Standard Binary Cross-Entropy Loss.
    Baseline - does NOT handle class imbalance.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    Assigns higher penalty to foreground (cable) pixels.
    pos_weight = background_ratio / foreground_ratio = 44.9
    """
    def __init__(self, pos_weight=44.9):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, predictions, targets):
        pw = self.pos_weight.to(predictions.device)
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)
        return loss(predictions, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017).
    Down-weights easy background examples so the model
    focuses on hard foreground pixels (thin cables).

    alpha : weight for foreground class (0.25 is common)
    gamma : focusing parameter (2.0 is most common)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # Standard BCE per pixel (no reduction)
        bce = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )

        # Focal weight: (1 - p_t)^gamma
        p_t          = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss - directly optimises the overlap between
    prediction and ground truth mask.
    Works well for imbalanced segmentation tasks.

    smooth : small value to avoid division by zero
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)

        # Flatten spatial dimensions
        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)

        return 1.0 - dice   # loss = 1 - Dice score


class TverskyLoss(nn.Module):
    """
    Tversky Loss (Salehi et al. 2017).
    Generalisation of Dice Loss with separate weights for
    False Positives (FP) and False Negatives (FN).

    For thin structures like cables, we want to penalise
    False Negatives more (missing a cable is worse than a
    small false detection), so set beta > alpha.

    alpha : weight for False Positives  (default 0.3)
    beta  : weight for False Negatives  (default 0.7)
    smooth: small value to avoid division by zero
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)

        # Flatten spatial dimensions
        probs   = probs.view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        tversky = (TP + self.smooth) / \
                  (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1.0 - tversky


def get_loss(name):
    """
    Factory function - returns the loss function by name.
    Usage: criterion = get_loss("focal")
    """
    losses = {
        "bce"          : BCELoss(),
        "weighted_bce" : WeightedBCELoss(pos_weight=44.9),
        "focal"        : FocalLoss(alpha=0.25, gamma=2.0),
        "dice"         : DiceLoss(smooth=1.0),
        "tversky"      : TverskyLoss(alpha=0.3, beta=0.7),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    return losses[name]
