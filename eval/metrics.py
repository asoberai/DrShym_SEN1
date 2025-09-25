"""
Evaluation metrics for flood segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model outputs (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + BCE loss with stability"""

    def __init__(self, dice_weight: float = 0.3, bce_weight: float = 0.7):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(predictions, targets)

        # Only add dice loss if BCE is stable
        if not torch.isnan(bce) and bce.item() < 10.0:
            dice = self.dice_loss(predictions, targets)
            if not torch.isnan(dice):
                return self.dice_weight * dice + self.bce_weight * bce

        return bce


class StableBCELoss(nn.Module):
    """Stable BCE loss for flood segmentation with NaN protection"""

    def __init__(self, pos_weight: float = 2.0, epsilon: float = 1e-7):
        super(StableBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure targets are in correct format and range
        targets = targets.float()
        targets = torch.clamp(targets, 0.0, 1.0)

        # Check for NaN or infinite values
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("Warning: NaN or Inf in predictions")
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=10.0, neginf=-10.0)

        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN or Inf in targets")
            targets = torch.nan_to_num(targets, nan=0.0)

        # Calculate positive weight based on class imbalance
        pos_count = targets.sum()
        neg_count = targets.numel() - pos_count

        if pos_count > 0:
            weight = neg_count / pos_count
            weight = torch.clamp(weight, 0.1, 10.0)  # Clamp extreme weights
        else:
            weight = torch.tensor(1.0, device=targets.device)

        # Use stable BCE computation
        try:
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='mean')
            loss = criterion(predictions, targets)

            # Final NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN loss detected, using fallback")
                # Fallback to simple MSE loss
                probs = torch.sigmoid(predictions)
                loss = nn.functional.mse_loss(probs, targets)

        except Exception as e:
            print(f"Error in loss computation: {e}")
            # Emergency fallback
            probs = torch.sigmoid(predictions)
            loss = nn.functional.mse_loss(probs, targets)

        return loss


class FloodMetrics:
    """Comprehensive metrics for flood segmentation evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.total_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """
        Update metrics with batch predictions

        Args:
            predictions: Model logits (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
            threshold: Classification threshold
        """
        # Apply sigmoid and threshold
        probs = torch.sigmoid(predictions)
        pred_masks = (probs > threshold).float()

        # Flatten for metric calculation
        pred_flat = pred_masks.view(-1)
        target_flat = targets.view(-1)

        # Calculate confusion matrix components
        tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
        tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
        fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        self.total_samples += predictions.shape[0]

    def compute_iou(self) -> float:
        """Intersection over Union"""
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        return self.tp / (self.tp + self.fp + self.fn)

    def compute_f1(self) -> float:
        """F1 Score"""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_precision(self) -> float:
        """Precision"""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def compute_recall(self) -> float:
        """Recall"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def compute_accuracy(self) -> float:
        """Accuracy"""
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'iou': self.compute_iou(),
            'f1': self.compute_f1(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'accuracy': self.compute_accuracy()
        }


def compute_metrics(predictions: torch.Tensor,
                   targets: torch.Tensor,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute metrics for a single batch

    Args:
        predictions: Model logits (B, 1, H, W)
        targets: Ground truth (B, 1, H, W)
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    metrics = FloodMetrics()
    metrics.update(predictions, targets, threshold)
    return metrics.compute_all()


def compute_dice_coefficient(predictions: torch.Tensor,
                           targets: torch.Tensor,
                           smooth: float = 1.0) -> float:
    """
    Compute Dice coefficient

    Args:
        predictions: Model outputs (B, 1, H, W)
        targets: Ground truth (B, 1, H, W)
        smooth: Smoothing factor

    Returns:
        Dice coefficient
    """
    # Apply sigmoid to get probabilities
    predictions = torch.sigmoid(predictions)

    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate Dice coefficient
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

    return dice.item()


def compute_expected_calibration_error(predictions: torch.Tensor,
                                      targets: torch.Tensor,
                                      n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE)

    ECE measures how well the predicted probabilities align with actual outcomes.
    Lower ECE indicates better calibrated probabilities.

    Args:
        predictions: Model logits (B, 1, H, W)
        targets: Ground truth (B, 1, H, W)
        n_bins: Number of bins for calibration

    Returns:
        ECE value (0 = perfectly calibrated)
    """
    # Convert to probabilities
    probs = torch.sigmoid(predictions).flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(probs)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Average confidence in bin
            confidence_in_bin = probs[in_bin].mean()

            # Average accuracy in bin
            accuracy_in_bin = targets_flat[in_bin].mean()

            # Add to ECE
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin

    return float(ece)


def compute_brier_score(predictions: torch.Tensor,
                       targets: torch.Tensor) -> float:
    """
    Compute Brier Score

    Brier score measures the accuracy of probabilistic predictions.
    Lower Brier score indicates better probabilistic predictions.

    Args:
        predictions: Model logits (B, 1, H, W)
        targets: Ground truth (B, 1, H, W)

    Returns:
        Brier score (0 = perfect, 1 = worst)
    """
    # Convert to probabilities
    probs = torch.sigmoid(predictions).flatten()
    targets_flat = targets.flatten()

    # Brier score = mean((prob - target)^2)
    brier = torch.mean((probs - targets_flat) ** 2)

    return float(brier.item())


def compute_reliability_diagram_data(predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Compute data for reliability diagram

    Args:
        predictions: Model logits (B, 1, H, W)
        targets: Ground truth (B, 1, H, W)
        n_bins: Number of bins

    Returns:
        Dictionary with bin_confidence, bin_accuracy, bin_counts
    """
    probs = torch.sigmoid(predictions).flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confidence = []
    bin_accuracy = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)

        if in_bin.sum() > 0:
            confidence = probs[in_bin].mean()
            accuracy = targets_flat[in_bin].mean()
            count = in_bin.sum()
        else:
            confidence = (bin_lower + bin_upper) / 2
            accuracy = 0.0
            count = 0

        bin_confidence.append(confidence)
        bin_accuracy.append(accuracy)
        bin_counts.append(count)

    return {
        'bin_confidence': np.array(bin_confidence),
        'bin_accuracy': np.array(bin_accuracy),
        'bin_counts': np.array(bin_counts)
    }


class AdvancedFloodMetrics:
    """Extended flood metrics including calibration measures"""

    def __init__(self):
        self.predictions = []
        self.targets = []
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.total_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """Update metrics with batch predictions"""

        # Store for calibration metrics
        self.predictions.append(predictions.cpu().detach())
        self.targets.append(targets.cpu().detach())

        # Standard binary classification metrics
        probs = torch.sigmoid(predictions)
        pred_masks = (probs > threshold).float()

        # Flatten for metric calculation
        pred_flat = pred_masks.view(-1)
        target_flat = targets.view(-1)

        # Calculate confusion matrix components
        tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
        tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
        fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        self.total_samples += predictions.shape[0]

    def compute_iou(self) -> float:
        """Intersection over Union"""
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        return self.tp / (self.tp + self.fp + self.fn)

    def compute_f1(self) -> float:
        """F1 Score"""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_precision(self) -> float:
        """Precision"""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def compute_recall(self) -> float:
        """Recall"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def compute_accuracy(self) -> float:
        """Accuracy"""
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def compute_ece(self, n_bins: int = 15) -> float:
        """Expected Calibration Error"""
        if not self.predictions:
            return 0.0

        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        return compute_expected_calibration_error(all_preds, all_targets, n_bins)

    def compute_brier(self) -> float:
        """Brier Score"""
        if not self.predictions:
            return 0.0

        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        return compute_brier_score(all_preds, all_targets)

    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics including calibration"""
        return {
            'iou': self.compute_iou(),
            'f1': self.compute_f1(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'accuracy': self.compute_accuracy(),
            'ece': self.compute_ece(),
            'brier': self.compute_brier()
        }

    def get_reliability_data(self, n_bins: int = 10) -> Dict[str, np.ndarray]:
        """Get reliability diagram data"""
        if not self.predictions:
            return {'bin_confidence': np.array([]), 'bin_accuracy': np.array([]), 'bin_counts': np.array([])}

        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        return compute_reliability_diagram_data(all_preds, all_targets, n_bins)