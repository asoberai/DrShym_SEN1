"""
Uncertainty Analysis for Active Learning in Flood Segmentation
Analyzes prediction confidence and selects uncertain tiles for human review
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_prediction_results(predictions_dir: str) -> List[Dict]:
 """
 Load all prediction JSON files and extract uncertainty metrics

 Args:
 predictions_dir: Directory containing prediction JSON files

 Returns:
 List of prediction dictionaries with uncertainty metrics
 """
 predictions_path = Path(predictions_dir)
 if not predictions_path.exists():
 print(f"ERROR: Predictions directory not found: {predictions_dir}")
 return []

 prediction_files = list(predictions_path.glob("*_prediction.json"))
 print(f"Found {len(prediction_files)} prediction files")

 results = []

 for json_file in prediction_files:
 try:
 with open(json_file, 'r') as f:
 data = json.load(f)

 # Extract base filename without _prediction.json
 base_name = json_file.stem.replace('_prediction', '')

 # Calculate uncertainty metrics from nested structure
 prediction = data.get('prediction', {})
 confidence = prediction.get('confidence', 0.0)
 flood_percentage = prediction.get('flood_percentage', 0.0)

 # Uncertainty = distance to 0.5 (decision boundary)
 distance_to_boundary = abs(confidence - 0.5)
 uncertainty_score = 1.0 - distance_to_boundary * 2 # Higher = more uncertain

 # Alternative: Entropy-based uncertainty (approximation)
 # For binary case: H = -p*log(p) - (1-p)*log(1-p)
 p = confidence
 if p > 0.999: p = 0.999
 if p < 0.001: p = 0.001
 entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))

 result = {
 'file_name': base_name + '.tif',
 'json_file': str(json_file),
 'confidence': confidence,
 'flood_percentage': flood_percentage,
 'uncertainty_score': uncertainty_score, # Distance to boundary based
 'entropy': entropy, # Information theory based
 'model_version': data.get('model_version', 'unknown'),
 'prediction_time': data.get('prediction_time', 'unknown')
 }

 results.append(result)

 except Exception as e:
 print(f"WARNING: Error processing {json_file}: {e}")
 continue

 print(f"Successfully processed {len(results)} prediction files")
 return results


def analyze_uncertainty_distribution(results: List[Dict]) -> Dict:
 """
 Analyze the distribution of uncertainty metrics

 Args:
 results: List of prediction results

 Returns:
 Summary statistics
 """
 if not results:
 return {}

 df = pd.DataFrame(results)

 stats = {
 'total_samples': len(df),
 'confidence_stats': {
 'mean': df['confidence'].mean(),
 'std': df['confidence'].std(),
 'min': df['confidence'].min(),
 'max': df['confidence'].max(),
 'median': df['confidence'].median()
 },
 'uncertainty_stats': {
 'mean': df['uncertainty_score'].mean(),
 'std': df['uncertainty_score'].std(),
 'min': df['uncertainty_score'].min(),
 'max': df['uncertainty_score'].max(),
 'median': df['uncertainty_score'].median()
 },
 'entropy_stats': {
 'mean': df['entropy'].mean(),
 'std': df['entropy'].std(),
 'min': df['entropy'].min(),
 'max': df['entropy'].max(),
 'median': df['entropy'].median()
 },
 'flood_coverage_stats': {
 'mean': df['flood_percentage'].mean(),
 'std': df['flood_percentage'].std(),
 'min': df['flood_percentage'].min(),
 'max': df['flood_percentage'].max(),
 'median': df['flood_percentage'].median()
 }
 }

 return stats


def select_uncertain_tiles(results: List[Dict],
 top_k: int = 20,
 method: str = 'uncertainty_score') -> List[Dict]:
 """
 Select the most uncertain tiles for human review

 Args:
 results: List of prediction results
 top_k: Number of tiles to select
 method: Uncertainty metric to use ('uncertainty_score' or 'entropy')

 Returns:
 List of most uncertain tiles
 """
 if not results:
 return []

 # Sort by uncertainty metric (descending - most uncertain first)
 sorted_results = sorted(results, key=lambda x: x[method], reverse=True)

 # Select top-k most uncertain
 uncertain_tiles = sorted_results[:top_k]

 print(f"\n Selected {len(uncertain_tiles)} most uncertain tiles using {method}:")
 print("=" * 70)

 for i, tile in enumerate(uncertain_tiles[:10], 1): # Show top 10
 print(f"{i:2d}. {tile['file_name']:<30} "
 f"Conf: {tile['confidence']:.3f} "
 f"Uncert: {tile['uncertainty_score']:.3f} "
 f"Flood: {tile['flood_percentage']:.1f}%")

 if len(uncertain_tiles) > 10:
 print(f" ... and {len(uncertain_tiles) - 10} more")

 return uncertain_tiles


def create_uncertainty_plots(results: List[Dict], output_dir: str):
 """
 Create visualization plots for uncertainty analysis

 Args:
 results: List of prediction results
 output_dir: Directory to save plots
 """
 if not results:
 return

 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)

 df = pd.DataFrame(results)

 # Set up matplotlib for non-GUI backend
 plt.switch_backend('Agg')

 # Create figure with subplots
 fig, axes = plt.subplots(2, 2, figsize=(15, 12))
 fig.suptitle('Uncertainty Analysis for Flood Segmentation Predictions', fontsize=16)

 # 1. Confidence distribution
 axes[0, 0].hist(df['confidence'], bins=30, alpha=0.7, color='blue', edgecolor='black')
 axes[0, 0].set_xlabel('Confidence Score')
 axes[0, 0].set_ylabel('Frequency')
 axes[0, 0].set_title('Confidence Score Distribution')
 axes[0, 0].axvline(df['confidence'].mean(), color='red', linestyle='--',
 label=f'Mean: {df["confidence"].mean():.3f}')
 axes[0, 0].legend()

 # 2. Uncertainty distribution
 axes[0, 1].hist(df['uncertainty_score'], bins=30, alpha=0.7, color='orange', edgecolor='black')
 axes[0, 1].set_xlabel('Uncertainty Score')
 axes[0, 1].set_ylabel('Frequency')
 axes[0, 1].set_title('Uncertainty Score Distribution')
 axes[0, 1].axvline(df['uncertainty_score'].mean(), color='red', linestyle='--',
 label=f'Mean: {df["uncertainty_score"].mean():.3f}')
 axes[0, 1].legend()

 # 3. Confidence vs Flood Percentage
 scatter = axes[1, 0].scatter(df['confidence'], df['flood_percentage'],
 alpha=0.6, c=df['uncertainty_score'],
 cmap='viridis', s=30)
 axes[1, 0].set_xlabel('Confidence Score')
 axes[1, 0].set_ylabel('Flood Percentage (%)')
 axes[1, 0].set_title('Confidence vs Flood Coverage')
 plt.colorbar(scatter, ax=axes[1, 0], label='Uncertainty Score')

 # 4. Entropy distribution
 axes[1, 1].hist(df['entropy'], bins=30, alpha=0.7, color='green', edgecolor='black')
 axes[1, 1].set_xlabel('Entropy')
 axes[1, 1].set_ylabel('Frequency')
 axes[1, 1].set_title('Entropy Distribution')
 axes[1, 1].axvline(df['entropy'].mean(), color='red', linestyle='--',
 label=f'Mean: {df["entropy"].mean():.3f}')
 axes[1, 1].legend()

 plt.tight_layout()

 # Save plot
 plot_path = output_path / 'uncertainty_analysis.png'
 plt.savefig(plot_path, dpi=300, bbox_inches='tight')
 plt.close()

 print(f"Uncertainty plots saved to: {plot_path}")


def save_uncertainty_analysis(results: List[Dict],
 uncertain_tiles: List[Dict],
 stats: Dict,
 output_dir: str):
 """
 Save complete uncertainty analysis results

 Args:
 results: All prediction results
 uncertain_tiles: Selected uncertain tiles
 stats: Summary statistics
 output_dir: Output directory
 """
 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)

 # Save full results DataFrame
 df = pd.DataFrame(results)
 csv_path = output_path / 'prediction_uncertainty_analysis.csv'
 df.to_csv(csv_path, index=False)
 print(f"Full analysis saved to: {csv_path}")

 # Save uncertain tiles for review
 uncertain_df = pd.DataFrame(uncertain_tiles)
 uncertain_csv_path = output_path / 'uncertain_tiles_for_review.csv'
 uncertain_df.to_csv(uncertain_csv_path, index=False)
 print(f"Uncertain tiles saved to: {uncertain_csv_path}")

 # Save summary statistics
 stats_path = output_path / 'uncertainty_statistics.json'
 with open(stats_path, 'w') as f:
 json.dump(stats, f, indent=2)
 print(f"📈 Statistics saved to: {stats_path}")

 # Create review instructions file
 instructions_path = output_path / 'review_instructions.md'
 with open(instructions_path, 'w') as f:
 f.write("""# Active Learning Review Instructions

## Overview
This directory contains the most uncertain predictions from the DrShym Climate flood segmentation model.
These tiles have been selected for human review to improve the model through active learning.

## Files
- `uncertain_tiles_for_review.csv`: Top uncertain tiles ranked by uncertainty score
- `prediction_uncertainty_analysis.csv`: Complete analysis of all predictions
- `uncertainty_analysis.png`: Visualization of uncertainty distributions
- `uncertainty_statistics.json`: Summary statistics

## Review Process
1. Review tiles in order of uncertainty score (highest first)
2. For each tile:
 - Check the original SAR image: `<tile_name>.tif`
 - Review model prediction: `<tile_name>_mask.png`
 - Check probability map: `<tile_name>_proba.png`
 - Annotate corrections if needed

## Target: < 30 seconds per tile annotation
- Focus on clear flood/no-flood decisions
- Mark ambiguous areas for expert review
- Note systematic errors for model improvement

## Priority Tiles
The top 10 most uncertain tiles should be reviewed first as they are likely
to provide the most valuable training signal for model improvement.
""")
 print(f"Review instructions saved to: {instructions_path}")


def main():
 parser = argparse.ArgumentParser(description='Analyze prediction uncertainty for active learning')
 parser.add_argument('--predictions', '-p', type=str, default='outputs/predictions',
 help='Directory containing prediction JSON files')
 parser.add_argument('--output', '-o', type=str, default='outputs/uncertainty_analysis',
 help='Output directory for analysis results')
 parser.add_argument('--top_k', '-k', type=int, default=20,
 help='Number of most uncertain tiles to select')
 parser.add_argument('--method', '-m', choices=['uncertainty_score', 'entropy'],
 default='uncertainty_score', help='Uncertainty metric to use')

 args = parser.parse_args()

 print(f"DrShym Climate Uncertainty Analysis")
 print("=" * 40)
 print(f"Predictions: {args.predictions}")
 print(f"Output: {args.output}")
 print(f"Top-k: {args.top_k}")
 print(f"Method: {args.method}")

 # Load prediction results
 results = load_prediction_results(args.predictions)

 if not results:
 print("ERROR: No valid prediction results found")
 return

 # Analyze uncertainty distribution
 stats = analyze_uncertainty_distribution(results)

 print("\n Uncertainty Statistics:")
 print("=" * 30)
 print(f"Total Samples: {stats['total_samples']}")
 print(f"Confidence - Mean: {stats['confidence_stats']['mean']:.3f}, "
 f"Std: {stats['confidence_stats']['std']:.3f}")
 print(f"Uncertainty - Mean: {stats['uncertainty_stats']['mean']:.3f}, "
 f"Std: {stats['uncertainty_stats']['std']:.3f}")
 print(f"Entropy - Mean: {stats['entropy_stats']['mean']:.3f}, "
 f"Std: {stats['entropy_stats']['std']:.3f}")

 # Select uncertain tiles for review
 uncertain_tiles = select_uncertain_tiles(results, args.top_k, args.method)

 # Create visualization plots
 create_uncertainty_plots(results, args.output)

 # Save all analysis results
 save_uncertainty_analysis(results, uncertain_tiles, stats, args.output)

 print(f"\n Uncertainty analysis completed!")
 print(f"{len(uncertain_tiles)} tiles selected for human review")
 print(f"Results saved to: {args.output}")
 print("\n🚀 Ready for active learning annotation workflow!")


if __name__ == "__main__":
 main()