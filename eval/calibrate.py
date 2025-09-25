"""
Temperature calibration for flood segmentation models
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar


class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration"""

    def __init__(self, initial_temperature: float = 1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return logits / self.temperature

    def set_temperature(self, temperature: float):
        """Set temperature parameter"""
        self.temperature.data.fill_(temperature)


def temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits"""
    return logits / temperature


def compute_ece_for_temperature(temperature: float,
                               logits: torch.Tensor,
                               targets: torch.Tensor,
                               n_bins: int = 15) -> float:
    """Compute ECE for given temperature (for optimization)"""
    # Apply temperature scaling
    scaled_logits = temperature_scale_logits(logits, temperature)
    probs = torch.sigmoid(scaled_logits).flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

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


def find_optimal_temperature(validation_loader: DataLoader,
                           model: nn.Module,
                           device: torch.device,
                           temperature_range: Tuple[float, float] = (0.1, 10.0)) -> float:
    """
    Find optimal temperature using validation set

    Args:
        validation_loader: Validation data loader
        model: Trained model
        device: Device to run on
        temperature_range: Range to search for temperature

    Returns:
        Optimal temperature value
    """
    print("Finding optimal temperature for calibration...")

    model.eval()

    # Collect all logits and targets
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(validation_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)

            # Store for calibration
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all data
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print(f"Collected {all_logits.shape[0]} samples for calibration")

    # Define objective function for optimization
    def objective(temperature):
        return compute_ece_for_temperature(temperature, all_logits, all_targets)

    # Find optimal temperature
    result = minimize_scalar(objective, bounds=temperature_range, method='bounded')

    if result.success:
        optimal_temp = result.x
        optimal_ece = result.fun
        print(f"Optimal temperature: {optimal_temp:.3f}")
        print(f"ECE after calibration: {optimal_ece:.4f}")
    else:
        print("Optimization failed, using default temperature")
        optimal_temp = 1.0

    return optimal_temp


def calibrate_model(model: nn.Module,
                   validation_loader: DataLoader,
                   device: torch.device,
                   save_path: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate model using temperature scaling

    Args:
        model: Trained model to calibrate
        validation_loader: Validation data for calibration
        device: Device to run on
        save_path: Path to save calibration results

    Returns:
        Tuple of (optimal_temperature, calibration_metrics)
    """
    print("Starting model calibration...")

    # Find optimal temperature
    optimal_temp = find_optimal_temperature(validation_loader, model, device)

    # Compute metrics before and after calibration
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in validation_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute ECE before calibration
    ece_before = compute_ece_for_temperature(1.0, all_logits, all_targets)

    # Compute ECE after calibration
    ece_after = compute_ece_for_temperature(optimal_temp, all_logits, all_targets)

    # Calculate improvement
    ece_reduction = ((ece_before - ece_after) / ece_before) * 100 if ece_before > 0 else 0

    calibration_metrics = {
        'optimal_temperature': optimal_temp,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'ece_reduction_percent': ece_reduction
    }

    print(f"Calibration Results:")
    print(f"  Temperature: {optimal_temp:.3f}")
    print(f"  ECE before: {ece_before:.4f}")
    print(f"  ECE after: {ece_after:.4f}")
    print(f"  Reduction: {ece_reduction:.1f}%")

    # Save calibration info
    if save_path:
        calibration_info = {
            'temperature': optimal_temp,
            'metrics': calibration_metrics,
            'method': 'temperature_scaling'
        }

        with open(save_path, 'w') as f:
            json.dump(calibration_info, f, indent=2)

        print(f"Calibration results saved to: {save_path}")

    return optimal_temp, calibration_metrics


def apply_temperature_calibration(logits: torch.Tensor,
                                temperature: float) -> torch.Tensor:
    """Apply temperature calibration to logits"""
    return logits / temperature


def load_calibration_temperature(calibration_path: str) -> float:
    """Load saved calibration temperature"""
    try:
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
        return calibration_data.get('temperature', 1.0)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load calibration from {calibration_path}, using temperature=1.0")
        return 1.0


def update_thresholds_with_calibration(thresholds_path: str,
                                     calibration_temperature: float,
                                     validation_metrics: Dict[str, float]):
    """Update thresholds.json with calibration info"""

    # Load existing thresholds or create new
    try:
        with open(thresholds_path, 'r') as f:
            thresholds_data = json.load(f)
    except FileNotFoundError:
        thresholds_data = {}

    # Update with calibration info
    thresholds_data.update({
        'calibration': {
            'method': 'temperature_scaling',
            'temperature': calibration_temperature,
            'metrics': validation_metrics
        },
        'updated_at': str(torch.datetime.now() if hasattr(torch, 'datetime') else 'unknown')
    })

    # Save updated thresholds
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_data, f, indent=2)

    print(f"Updated thresholds saved to: {thresholds_path}")


# Example usage and testing
if __name__ == "__main__":
    print("Temperature calibration module loaded successfully")

    # Test basic functionality
    test_logits = torch.randn(10, 1, 64, 64)
    test_targets = torch.randint(0, 2, (10, 1, 64, 64)).float()

    ece_uncalibrated = compute_ece_for_temperature(1.0, test_logits, test_targets)
    ece_calibrated = compute_ece_for_temperature(1.5, test_logits, test_targets)

    print(f"Test ECE (T=1.0): {ece_uncalibrated:.4f}")
    print(f"Test ECE (T=1.5): {ece_calibrated:.4f}")