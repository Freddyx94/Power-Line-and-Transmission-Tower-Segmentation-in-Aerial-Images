"""
Evaluation metrics for binary segmentation.
Implements: IoU, F1 (Dice), Precision, Recall.
"""

import torch


def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute segmentation metrics from raw logits.

    Args:
        predictions : model output logits, shape [B, 1, H, W]
        targets     : ground truth masks,  shape [B, 1, H, W]
        threshold   : pixel is foreground if sigmoid(logit) > threshold

    Returns:
        dict with iou, f1, precision, recall
    """
    # Convert logits to binary predictions
    probs  = torch.sigmoid(predictions)
    preds  = (probs > threshold).float()

    # Flatten everything
    preds   = preds.view(-1)
    targets = targets.view(-1)

    TP = (preds * targets).sum().item()
    FP = (preds * (1 - targets)).sum().item()
    FN = ((1 - preds) * targets).sum().item()
    TN = ((1 - preds) * (1 - targets)).sum().item()

    # IoU (Intersection over Union) - main metric for segmentation
    iou = TP / (TP + FP + FN + 1e-8)

    # F1 / Dice Score
    f1 = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    # Precision - of all predicted cables, how many are correct?
    precision = TP / (TP + FP + 1e-8)

    # Recall - of all actual cables, how many did we detect?
    recall = TP / (TP + FN + 1e-8)

    return {
        "iou"       : iou,
        "f1"        : f1,
        "precision" : precision,
        "recall"    : recall,
    }
