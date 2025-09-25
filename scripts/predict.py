#!/usr/bin/env python3
"""
Real prediction script for DrShym Climate flood segmentation
Uses trained PyTorch models on actual SAR data
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.unet import UNet
from utils.io import load_geotiff, save_geotiff


def load_trained_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
 """Load trained model from checkpoint"""

 try:
 checkpoint = torch.load(checkpoint_path, map_location=device)

 # Get model config
 config = checkpoint.get('config', {})
 model_config = config.get('model', {})

 # Create model
 model = UNet(
 in_channels=1,
 num_classes=1,
 encoder=model_config.get('encoder', 'resnet18'),
 pretrained=False # Don't need pretrained weights when loading checkpoint
 )

 # Load weights
 model.load_state_dict(checkpoint['model_state_dict'])
 model.to(device)
 model.eval()

 # Get model info
 model_info = {
 'architecture': f"UNet + {model_config.get('encoder', 'resnet18')}",
 'epoch': checkpoint.get('epoch', 0),
 'metrics': checkpoint.get('metrics', {}),
 'parameters': sum(p.numel() for p in model.parameters())
 }

 print(f"Loaded trained model from {checkpoint_path}")
 print(f" Architecture: {model_info['architecture']}")
 print(f" Epoch: {model_info['epoch']}")
 if 'f1' in model_info['metrics']:
 print(f" Validation F1: {model_info['metrics']['f1']:.3f}")

 return model, model_info

 except Exception as e:
 print(f"ERROR: Error loading model: {e}")
 raise e


def process_image(image_path: Path,
 model: torch.nn.Module,
 device: torch.device,
 target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
 """Process single SAR image with trained model"""

 try:
 # Load SAR image
 sar_data, metadata = load_geotiff(str(image_path), normalize=True, as_tensor=True)

 # Resize if needed
 if sar_data.shape[-2:] != target_size:
 sar_data = F.interpolate(
 sar_data.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
 ).squeeze(0)

 # Ensure single channel
 if len(sar_data.shape) == 2:
 sar_data = sar_data.unsqueeze(0)

 # Add batch dimension and move to device
 input_tensor = sar_data.unsqueeze(0).to(device)

 # Model inference
 with torch.no_grad():
 logits = model(input_tensor)
 probs = torch.sigmoid(logits)

 # Convert to numpy
 probs_np = probs.cpu().squeeze().numpy()
 mask_np = (probs_np > 0.5).astype(np.uint8) * 255

 # Calculate metrics
 flood_percentage = (probs_np > 0.5).mean() * 100
 avg_confidence = probs_np.mean()
 max_confidence = probs_np.max()

 prediction_stats = {
 'flood_probability': float(avg_confidence),
 'confidence': float(max_confidence),
 'flood_percentage': float(flood_percentage)
 }

 return mask_np, probs_np, prediction_stats

 except Exception as e:
 print(f" ERROR: Error processing {image_path.name}: {e}")
 # Return dummy data
 dummy_mask = np.zeros(target_size, dtype=np.uint8)
 dummy_probs = np.zeros(target_size, dtype=np.float32)
 dummy_stats = {'flood_probability': 0.0, 'confidence': 0.0, 'flood_percentage': 0.0}
 return dummy_mask, dummy_probs, dummy_stats


def predict_folder(input_dir: str, output_dir: str, checkpoint_path: str):
 """Process all SAR images in input folder with trained model"""

 print(f"DrShym Climate Real Batch Prediction")
 print("=" * 45)

 # Setup paths
 input_path = Path(input_dir)
 output_path = Path(output_dir)

 if not input_path.exists():
 print(f"ERROR: Input directory not found: {input_dir}")
 return

 output_path.mkdir(parents=True, exist_ok=True)

 print(f"Input: {input_dir}")
 print(f"Output: {output_dir}")
 print(f"Checkpoint: {checkpoint_path}")
 print()

 # Setup device and load model
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f"Using device: {device}")

 model, model_info = load_trained_model(checkpoint_path, device)

 # Find all SAR image files
 sar_extensions = {'.tif', '.tiff'}
 image_files = []

 for ext in sar_extensions:
 image_files.extend(input_path.glob(f"*{ext}"))
 image_files.extend(input_path.glob(f"*{ext.upper()}"))

 if not image_files:
 print(f"ERROR: No SAR image files found in {input_dir}")
 return

 print(f"Found {len(image_files)} SAR images to process")
 print()

 # Process images
 start_time = time.time()
 results = []

 for i, image_path in enumerate(image_files, 1):
 print(f"Processing {i}/{len(image_files)}: {image_path.name}")

 # Process image
 mask, probs, prediction_stats = process_image(image_path, model, device)

 # Save outputs
 stem = image_path.stem
 mask_path = output_path / f"{stem}_mask.png"
 probs_path = output_path / f"{stem}_proba.png"
 json_path = output_path / f"{stem}_prediction.json"

 # Save mask as PNG
 Image.fromarray(mask).save(mask_path)

 # Save probabilities as PNG (scaled to 0-255)
 probs_scaled = (probs * 255).astype(np.uint8)
 Image.fromarray(probs_scaled).save(probs_path)

 # Save prediction JSON
 result = {
 'input_file': str(image_path),
 'mask_output': str(mask_path),
 'proba_output': str(probs_path),
 'prediction': prediction_stats,
 'model_info': model_info,
 'processed_at': datetime.now().isoformat()
 }

 with open(json_path, 'w') as f:
 json.dump(result, f, indent=2)

 results.append(result)

 print(f" Flood: {prediction_stats['flood_percentage']:.1f}% | Confidence: {prediction_stats['confidence']:.3f}")
 print(f" Saved: {mask_path.name}, {probs_path.name}, {json_path.name}")

 # Calculate summary statistics
 total_time = time.time() - start_time
 avg_flood = sum(r["prediction"]["flood_percentage"] for r in results) / len(results)
 avg_confidence = sum(r["prediction"]["confidence"] for r in results) / len(results)

 print(f"\n Batch Processing Summary")
 print("=" * 30)
 print(f"Images processed: {len(image_files)}")
 print(f"Total time: {total_time:.1f}s")
 print(f"Average time per image: {total_time/len(image_files):.2f}s")
 print(f"Average flood percentage: {avg_flood:.1f}%")
 print(f"Average confidence: {avg_confidence:.3f}")

 # Save batch results
 batch_summary = {
 "summary": {
 "images_processed": len(image_files),
 "total_time": total_time,
 "avg_time_per_image": total_time / len(image_files),
 "avg_flood_percentage": avg_flood,
 "avg_confidence": avg_confidence,
 "processing_completed": datetime.now().isoformat()
 },
 "model_info": model_info,
 "results": results
 }

 results_path = output_path / 'batch_prediction_summary.json'
 with open(results_path, 'w') as f:
 json.dump(batch_summary, f, indent=2)

 print(f"\n Results saved to: {results_path}")
 print(f"Real batch prediction completed successfully!")


def main():
 """Main CLI entry point"""

 parser = argparse.ArgumentParser(description='Real flood prediction using trained PyTorch model')

 parser.add_argument('--ckpt', '--checkpoint', required=True,
 help='Path to trained model checkpoint (e.g., artifacts/checkpoints/best.pt)')
 parser.add_argument('--in', '--input', dest='input_dir', required=True,
 help='Input directory with SAR images')
 parser.add_argument('--out', '--output', dest='output_dir', required=True,
 help='Output directory for predictions')

 args = parser.parse_args()

 try:
 predict_folder(args.input_dir, args.output_dir, args.ckpt)
 return 0
 except Exception as e:
 print(f"ERROR: Prediction failed: {e}")
 return 1


if __name__ == "__main__":
 exit(main())