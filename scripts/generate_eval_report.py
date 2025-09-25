#!/usr/bin/env python3
"""
Generate comprehensive HTML evaluation report for DrShym Climate flood segmentation
"""

import argparse
import sys
import yaml
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import pandas as pd
import numpy as np
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import will be done dynamically to avoid import errors
# from eval.slices import main_slice_analysis
# from eval.calibrate import main_calibration_pipeline
# from eval.metrics import AdvancedFloodMetrics
# from models.unet import UNet
# from torch.utils.data import DataLoader
# import torch


def create_reliability_diagram(reliability_data: Dict[str, np.ndarray],
                              title: str = "Reliability Diagram") -> str:
    """Create reliability diagram and return as base64 string"""

    plt.figure(figsize=(8, 6))

    bin_confidence = reliability_data['bin_confidence']
    bin_accuracy = reliability_data['bin_accuracy']
    bin_counts = reliability_data['bin_counts']

    # Plot reliability curve
    plt.plot(bin_confidence, bin_accuracy, 'o-', linewidth=2, markersize=8, label='Model')

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, label='Perfect Calibration')

    # Add sample counts as text
    for i, (conf, acc, count) in enumerate(zip(bin_confidence, bin_accuracy, bin_counts)):
        plt.annotate(f'{int(count)}', (conf, acc), textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=8)

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return image_base64


def create_slice_analysis_chart(slice_results: List[Dict[str, Any]],
                               title: str = "Error Analysis by Slice") -> str:
    """Create slice analysis bar chart and return as base64 string"""

    if not slice_results:
        return ""

    plt.figure(figsize=(12, 6))

    # Extract data for plotting
    slice_names = [r['slice_name'] for r in slice_results]
    iou_scores = [r['metrics']['iou'] for r in slice_results]
    sample_counts = [r['sample_count'] for r in slice_results]

    # Create bars
    bars = plt.bar(range(len(slice_names)), iou_scores, alpha=0.7)

    # Add sample counts on top of bars
    for i, (bar, count) in enumerate(zip(bars, sample_counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)

    plt.xticks(range(len(slice_names)), slice_names, rotation=45, ha='right')
    plt.ylabel('IoU Score')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return image_base64


def generate_html_report(results: Dict[str, Any], output_path: str = "eval_report.html"):
    """Generate comprehensive HTML evaluation report"""

    # Create charts
    reliability_chart = ""
    if 'calibration' in results and 'reliability_data' in results['calibration']:
        reliability_chart = create_reliability_diagram(
            results['calibration']['reliability_data'],
            "Model Calibration - Reliability Diagram"
        )

    slice_chart = ""
    if 'slices' in results and results['slices']:
        slice_chart = create_slice_analysis_chart(
            results['slices'],
            "Performance by Data Slice"
        )

    # HTML template
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DrShym Climate - Flood Segmentation Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #495057;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            color: #6c757d;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #dee2e6;
        }}
        h1, h2, h3 {{
            color: #495057;
        }}
        .status-good {{ color: #28a745; }}
        .status-fair {{ color: #ffc107; }}
        .status-poor {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DrShym Climate</h1>
        <h2>Flood Segmentation Model Evaluation</h2>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>Model Overview</h2>
        <p><strong>Architecture:</strong> {results.get('model_info', {}).get('architecture', 'UNet + ResNet')}</p>
        <p><strong>Parameters:</strong> {results.get('model_info', {}).get('parameters', 'N/A'):,}</p>
        <p><strong>Checkpoint:</strong> {results.get('checkpoint_path', 'N/A')}</p>
        <p><strong>Validation F1:</strong> {results.get('model_info', {}).get('validation_f1', 'N/A')}</p>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
"""

    # Add metric cards
    if 'metrics' in results:
        metrics = results['metrics']
        metric_cards = [
            ('IoU', metrics.get('iou', 0), 'iou'),
            ('F1 Score', metrics.get('f1', 0), 'f1'),
            ('Precision', metrics.get('precision', 0), 'precision'),
            ('Recall', metrics.get('recall', 0), 'recall'),
            ('ECE', metrics.get('ece', 0), 'ece'),
            ('Brier Score', metrics.get('brier', 0), 'brier')
        ]

        for name, value, key in metric_cards:
            # Color coding based on metric type and value
            if key in ['iou', 'f1', 'precision', 'recall']:
                status_class = 'status-good' if value > 0.7 else 'status-fair' if value > 0.5 else 'status-poor'
            else:  # ECE and Brier (lower is better)
                status_class = 'status-good' if value < 0.1 else 'status-fair' if value < 0.2 else 'status-poor'

            html_template += f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value {status_class}">{value:.3f}</div>
            </div>
"""

    html_template += """
        </div>
    </div>
"""

    # Add calibration section
    if reliability_chart:
        html_template += f"""
    <div class="section">
        <h2>Model Calibration</h2>
        <p>Calibration measures how well the model's confidence scores match its actual accuracy.</p>
        <div class="chart">
            <img src="data:image/png;base64,{reliability_chart}" alt="Reliability Diagram" />
        </div>
"""

        if 'calibration' in results:
            cal_metrics = results['calibration']
            html_template += f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Temperature</div>
                <div class="metric-value">{cal_metrics.get('optimal_temperature', 1.0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ECE Before</div>
                <div class="metric-value">{cal_metrics.get('ece_before', 0.0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ECE After</div>
                <div class="metric-value">{cal_metrics.get('ece_after', 0.0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ECE Reduction</div>
                <div class="metric-value">{cal_metrics.get('ece_reduction_percent', 0.0):.1f}%</div>
            </div>
        </div>
"""

        html_template += """
    </div>
"""

    # Add slice analysis section
    if slice_chart and 'slices' in results:
        html_template += f"""
    <div class="section">
        <h2>Error Analysis by Data Slice</h2>
        <p>Performance breakdown across different data characteristics.</p>
        <div class="chart">
            <img src="data:image/png;base64,{slice_chart}" alt="Slice Analysis" />
        </div>

        <h3>Detailed Slice Results</h3>
        <table>
            <thead>
                <tr>
                    <th>Slice</th>
                    <th>Sample Count</th>
                    <th>IoU</th>
                    <th>F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                </tr>
            </thead>
            <tbody>
"""

        for slice_result in results['slices']:
            metrics = slice_result['metrics']
            html_template += f"""
                <tr>
                    <td>{slice_result['slice_name']}</td>
                    <td>{slice_result['sample_count']}</td>
                    <td>{metrics.get('iou', 0):.3f}</td>
                    <td>{metrics.get('f1', 0):.3f}</td>
                    <td>{metrics.get('precision', 0):.3f}</td>
                    <td>{metrics.get('recall', 0):.3f}</td>
                </tr>
"""

        html_template += """
            </tbody>
        </table>
    </div>
"""

    # Footer
    html_template += f"""
    <div class="footer">
        <p>DrShym Climate Flood Segmentation System</p>
        <p>Report generated automatically by evaluation pipeline</p>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_template)

    print(f"HTML evaluation report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive evaluation report')
    parser.add_argument('--results-dir', type=str, default='outputs/eval',
                       help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='eval_report.html',
                       help='Output HTML file path')
    parser.add_argument('--config', type=str, default='configs/flood.yaml',
                       help='Configuration file')

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
        print(f"Warning: Config file {args.config} not found, using defaults")

    # Aggregate results from different evaluation components
    results = {
        'model_info': {
            'architecture': 'UNet + ResNet50',
            'parameters': 47400000,  # Approximate
            'validation_f1': 0.531
        },
        'checkpoint_path': 'artifacts/checkpoints/best_optimized.pt'
    }

    # Load metrics if available
    metrics_file = Path(args.results_dir) / 'metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                results['metrics'] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metrics from {metrics_file}: {e}")

    # Load calibration results if available
    calibration_file = Path(args.results_dir) / 'calibration.json'
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r') as f:
                results['calibration'] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load calibration from {calibration_file}: {e}")

    # Load slice analysis if available
    slices_file = Path(args.results_dir) / 'slices.json'
    if slices_file.exists():
        try:
            with open(slices_file, 'r') as f:
                results['slices'] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load slices from {slices_file}: {e}")

    # Generate HTML report
    output_path = generate_html_report(results, args.output)

    print(f"✅ Evaluation report generated: {output_path}")
    print(f"📊 Open {output_path} in your browser to view the results")


if __name__ == '__main__':
    main()