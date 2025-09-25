"""
Error analysis by different slices for flood segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns # Optional dependency
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum


@dataclass
class SliceResult:
    """Results for a single slice"""
    slice_name: str
    slice_value: str
    sample_count: int
    iou: float
    f1: float
    precision: float
    recall: float
    ece: float
    brier: float


class SliceType(Enum):
    """Types of slices to analyze"""
    BACKSCATTER_INTENSITY = "backscatter_intensity_quantiles"
    SLOPE_BINS = "slope_bins"
    LANDCOVER_CLASS = "landcover_class"


def compute_backscatter_intensity_slices(images: torch.Tensor,
 predictions: torch.Tensor,
 targets: torch.Tensor,
 n_quantiles: int = 5) -> List[SliceResult]:
 """
 Analyze errors by SAR backscatter intensity quantiles

 Args:
 images: Input SAR images (B, C, H, W)
 predictions: Model logits (B, 1, H, W)
 targets: Ground truth masks (B, 1, H, W)
 n_quantiles: Number of quantiles to create

 Returns:
 List of slice results
 """
 from .metrics import compute_expected_calibration_error, compute_brier_score

 # Calculate mean backscatter intensity per image
 mean_intensities = images.mean(dim=(1, 2, 3)).cpu().numpy()

 # Create quantiles
 quantile_boundaries = np.quantile(mean_intensities, np.linspace(0, 1, n_quantiles + 1))

 slice_results = []

 for i in range(n_quantiles):
 # Define quantile range
 lower_bound = quantile_boundaries[i]
 upper_bound = quantile_boundaries[i + 1]

 # Find samples in this quantile
 if i == n_quantiles - 1: # Include upper boundary in last quantile
 in_quantile = (mean_intensities >= lower_bound) & (mean_intensities <= upper_bound)
 else:
 in_quantile = (mean_intensities >= lower_bound) & (mean_intensities < upper_bound)

 if not in_quantile.any():
 continue

 # Extract samples for this quantile
 quantile_preds = predictions[in_quantile]
 quantile_targets = targets[in_quantile]

 # Compute metrics
 metrics = compute_slice_metrics(quantile_preds, quantile_targets)

 slice_name = f"backscatter_q{i+1}"
 slice_value = f"{lower_bound:.3f}-{upper_bound:.3f}"

 slice_results.append(SliceResult(
 slice_name=slice_name,
 slice_value=slice_value,
 sample_count=in_quantile.sum(),
 **metrics
 ))

 return slice_results


def compute_slope_slices(images: torch.Tensor,
 predictions: torch.Tensor,
 targets: torch.Tensor,
 slope_bins: List[Tuple[float, float]] = None) -> List[SliceResult]:
 """
 Analyze errors by terrain slope bins

 Note: This is a simplified version. In production, you would use actual DEM data.
 Here we estimate slope from SAR texture/gradient.
 """
 if slope_bins is None:
 slope_bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]

 slice_results = []

 # Estimate slope from image gradient (simplified)
 for batch_idx in range(images.shape[0]):
 img = images[batch_idx, 0] # Take first channel

 # Compute gradient magnitude as proxy for slope
 grad_x = torch.diff(img, dim=1, prepend=img[:, :1])
 grad_y = torch.diff(img, dim=0, prepend=img[:1, :])
 gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

 # Average gradient per image as slope proxy
 if batch_idx == 0:
 estimated_slopes = gradient_magnitude.mean().unsqueeze(0)
 else:
 estimated_slopes = torch.cat([estimated_slopes, gradient_magnitude.mean().unsqueeze(0)])

 estimated_slopes = estimated_slopes.cpu().numpy()

 for i, (lower, upper) in enumerate(slope_bins):
 # Find samples in this slope range
 in_bin = (estimated_slopes >= lower) & (estimated_slopes < upper)

 if not in_bin.any():
 continue

 # Extract samples
 bin_preds = predictions[in_bin]
 bin_targets = targets[in_bin]

 # Compute metrics
 metrics = compute_slice_metrics(bin_preds, bin_targets)

 slice_name = f"slope_bin_{i+1}"
 slice_value = f"{lower:.1f}-{upper:.1f}"

 slice_results.append(SliceResult(
 slice_name=slice_name,
 slice_value=slice_value,
 sample_count=in_bin.sum(),
 **metrics
 ))

 return slice_results


def compute_landcover_slices(images: torch.Tensor,
 predictions: torch.Tensor,
 targets: torch.Tensor,
 landcover_classes: List[str] = None) -> List[SliceResult]:
 """
 Analyze errors by landcover class

 Note: Simplified version using SAR intensity patterns as landcover proxy.
 In production, would use actual landcover data.
 """
 if landcover_classes is None:
 landcover_classes = ["water", "urban", "vegetation", "bare_soil"]

 slice_results = []

 # Simple landcover classification based on SAR statistics
 # This is a rough approximation for demonstration
 for batch_idx in range(images.shape[0]):
 img = images[batch_idx, 0].cpu().numpy()

 # Simple heuristics based on SAR characteristics
 mean_val = img.mean()
 std_val = img.std()

 # Classify based on intensity and texture
 if mean_val < 0.2: # Very low backscatter
 landcover_class = "water"
 elif mean_val > 0.8: # High backscatter
 landcover_class = "urban"
 elif std_val > 0.3: # High texture
 landcover_class = "vegetation"
 else:
 landcover_class = "bare_soil"

 if batch_idx == 0:
 landcover_labels = [landcover_class]
 else:
 landcover_labels.append(landcover_class)

 # Compute metrics for each landcover class
 for landcover_class in landcover_classes:
 # Find samples of this landcover class
 class_mask = np.array([label == landcover_class for label in landcover_labels])

 if not class_mask.any():
 continue

 # Extract samples
 class_preds = predictions[class_mask]
 class_targets = targets[class_mask]

 # Compute metrics
 metrics = compute_slice_metrics(class_preds, class_targets)

 slice_results.append(SliceResult(
 slice_name=f"landcover_{landcover_class}",
 slice_value=landcover_class,
 sample_count=class_mask.sum(),
 **metrics
 ))

 return slice_results


def compute_slice_metrics(predictions: torch.Tensor,
 targets: torch.Tensor,
 threshold: float = 0.5) -> Dict[str, float]:
 """Compute metrics for a slice"""
 from .metrics import compute_expected_calibration_error, compute_brier_score

 # Binary classification metrics
 probs = torch.sigmoid(predictions)
 pred_masks = (probs > threshold).float()

 # Flatten
 pred_flat = pred_masks.view(-1)
 target_flat = targets.view(-1)

 # Confusion matrix
 tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
 fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
 tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
 fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()

 # Compute metrics
 iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
 precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
 recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
 f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

 # Calibration metrics
 ece = compute_expected_calibration_error(predictions, targets)
 brier = compute_brier_score(predictions, targets)

 return {
 'iou': iou,
 'f1': f1,
 'precision': precision,
 'recall': recall,
 'ece': ece,
 'brier': brier
 }


def analyze_error_slices(model: torch.nn.Module,
 data_loader,
 device: torch.device,
 slice_types: List[SliceType] = None) -> Dict[str, List[SliceResult]]:
 """
 Comprehensive error slice analysis

 Args:
 model: Trained model
 data_loader: Data loader for evaluation
 device: Device to run on
 slice_types: Types of slices to compute

 Returns:
 Dictionary of slice results by slice type
 """
 if slice_types is None:
 slice_types = [SliceType.BACKSCATTER_INTENSITY, SliceType.SLOPE_BINS, SliceType.LANDCOVER_CLASS]

 print(f"Analyzing error slices...")

 model.eval()

 # Collect all data
 all_images = []
 all_predictions = []
 all_targets = []

 with torch.no_grad():
 for batch_idx, (images, targets) in enumerate(data_loader):
 images = images.to(device)
 targets = targets.to(device)

 # Forward pass
 logits = model(images)

 # Store data
 all_images.append(images.cpu())
 all_predictions.append(logits.cpu())
 all_targets.append(targets.cpu())

 # Limit data for faster analysis (optional)
 if batch_idx >= 15: # Use first 15 batches
 break

 # Concatenate all data
 all_images = torch.cat(all_images, dim=0)
 all_predictions = torch.cat(all_predictions, dim=0)
 all_targets = torch.cat(all_targets, dim=0)

 print(f" Analyzing {all_images.shape[0]} samples")

 # Compute slices
 slice_results = {}

 for slice_type in slice_types:
 print(f" Computing {slice_type.value}...")

 if slice_type == SliceType.BACKSCATTER_INTENSITY:
 results = compute_backscatter_intensity_slices(all_images, all_predictions, all_targets)
 elif slice_type == SliceType.SLOPE_BINS:
 results = compute_slope_slices(all_images, all_predictions, all_targets)
 elif slice_type == SliceType.LANDCOVER_CLASS:
 results = compute_landcover_slices(all_images, all_predictions, all_targets)
 else:
 continue

 slice_results[slice_type.value] = results

 return slice_results


def create_slice_summary_table(slice_results: Dict[str, List[SliceResult]]) -> pd.DataFrame:
 """Create summary table of slice results"""

 rows = []

 for slice_type, results in slice_results.items():
 for result in results:
 rows.append({
 'Slice Type': slice_type,
 'Slice Name': result.slice_name,
 'Slice Value': result.slice_value,
 'Sample Count': result.sample_count,
 'IoU': result.iou,
 'F1': result.f1,
 'Precision': result.precision,
 'Recall': result.recall,
 'ECE': result.ece,
 'Brier': result.brier
 })

 return pd.DataFrame(rows)


def plot_slice_performance(slice_results: Dict[str, List[SliceResult]],
 output_dir: str):
 """Create performance plots for different slices"""

 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)

 # Set style
 # plt.style.use('seaborn-v0_8') # Comment out seaborn style

 for slice_type, results in slice_results.items():
 if not results:
 continue

 # Create subplot
 fig, axes = plt.subplots(2, 2, figsize=(12, 10))
 fig.suptitle(f'Performance Analysis: {slice_type.replace("_", " ").title()}', fontsize=16)

 # Extract data
 slice_names = [r.slice_name for r in results]
 iou_scores = [r.iou for r in results]
 f1_scores = [r.f1 for r in results]
 ece_scores = [r.ece for r in results]
 sample_counts = [r.sample_count for r in results]

 # IoU by slice
 axes[0, 0].bar(range(len(slice_names)), iou_scores, color='skyblue', alpha=0.7)
 axes[0, 0].set_title('IoU Score by Slice')
 axes[0, 0].set_ylabel('IoU')
 axes[0, 0].set_xticks(range(len(slice_names)))
 axes[0, 0].set_xticklabels(slice_names, rotation=45)
 axes[0, 0].grid(True, alpha=0.3)

 # F1 by slice
 axes[0, 1].bar(range(len(slice_names)), f1_scores, color='lightcoral', alpha=0.7)
 axes[0, 1].set_title('F1 Score by Slice')
 axes[0, 1].set_ylabel('F1')
 axes[0, 1].set_xticks(range(len(slice_names)))
 axes[0, 1].set_xticklabels(slice_names, rotation=45)
 axes[0, 1].grid(True, alpha=0.3)

 # ECE by slice
 axes[1, 0].bar(range(len(slice_names)), ece_scores, color='lightgreen', alpha=0.7)
 axes[1, 0].set_title('Expected Calibration Error by Slice')
 axes[1, 0].set_ylabel('ECE')
 axes[1, 0].set_xticks(range(len(slice_names)))
 axes[1, 0].set_xticklabels(slice_names, rotation=45)
 axes[1, 0].grid(True, alpha=0.3)

 # Sample count by slice
 axes[1, 1].bar(range(len(slice_names)), sample_counts, color='gold', alpha=0.7)
 axes[1, 1].set_title('Sample Count by Slice')
 axes[1, 1].set_ylabel('Sample Count')
 axes[1, 1].set_xticks(range(len(slice_names)))
 axes[1, 1].set_xticklabels(slice_names, rotation=45)
 axes[1, 1].grid(True, alpha=0.3)

 plt.tight_layout()

 # Save plot
 plot_path = output_dir / f"{slice_type}_performance.png"
 plt.savefig(plot_path, dpi=300, bbox_inches='tight')
 plt.close()

 print(f" Saved plot: {plot_path}")


def save_slice_results(slice_results: Dict[str, List[SliceResult]],
 output_path: str):
 """Save slice results to JSON"""

 # Convert to serializable format
 serializable_results = {}

 for slice_type, results in slice_results.items():
 serializable_results[slice_type] = [
 {
 'slice_name': r.slice_name,
 'slice_value': r.slice_value,
 'sample_count': r.sample_count,
 'metrics': {
 'iou': r.iou,
 'f1': r.f1,
 'precision': r.precision,
 'recall': r.recall,
 'ece': r.ece,
 'brier': r.brier
 }
 }
 for r in results
 ]

 output_path = Path(output_path)
 output_path.parent.mkdir(parents=True, exist_ok=True)

 with open(output_path, 'w') as f:
 json.dump(serializable_results, f, indent=2)

 print(f"Slice results saved to: {output_path}")


def main_slice_analysis(model_path: str,
 data_loader,
 device: torch.device,
 output_dir: str = "artifacts/evaluation/"):
 """
 Main slice analysis pipeline

 Args:
 model_path: Path to trained model
 data_loader: Validation data loader
 device: Device to run on
 output_dir: Output directory for results
 """
 from models.unet import UNet

 print("🔬 DrShym Climate Error Slice Analysis")
 print("=" * 40)

 # Load model
 print(f" Loading model...")
 checkpoint = torch.load(model_path, map_location=device)
 model_config = checkpoint.get('config', {}).get('model', {})

 model = UNet(
 in_channels=1,
 num_classes=1,
 encoder=model_config.get('encoder', 'resnet18'),
 pretrained=False
 ).to(device)

 model.load_state_dict(checkpoint['model_state_dict'])
 print(f"Model loaded from {model_path}")

 # Analyze slices
 slice_results = analyze_error_slices(model, data_loader, device)

 # Create summary table
 summary_df = create_slice_summary_table(slice_results)

 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)

 # Save summary table
 summary_path = output_dir / "slice_analysis_summary.csv"
 summary_df.to_csv(summary_path, index=False)
 print(f"Summary table saved to: {summary_path}")

 # Save detailed results
 results_path = output_dir / "slice_results.json"
 save_slice_results(slice_results, str(results_path))

 # Create plots
 print(f"Creating performance plots...")
 plot_slice_performance(slice_results, str(output_dir / "plots"))

 # Print summary
 print("\\n📈 Slice Analysis Summary:")
 print("=" * 30)

 for slice_type, results in slice_results.items():
 print(f"\\n{slice_type.replace('_', ' ').title()}:")
 for result in results:
 print(f" {result.slice_value}: F1={result.f1:.3f}, IoU={result.iou:.3f}, ECE={result.ece:.3f} (n={result.sample_count})")

 print("\\n Slice analysis completed successfully!")

 return slice_results, summary_df