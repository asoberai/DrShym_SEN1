"""
Rapid Annotation Workflow for Active Learning
Streamlined interface for <30 second per tile human annotation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime


class RapidAnnotator:
 """
 Streamlined annotation interface for uncertain flood tiles
 Target: <30 seconds per tile annotation
 """

 def __init__(self,
 uncertain_tiles_csv: str,
 predictions_dir: str,
 sar_images_dir: str,
 annotations_dir: str = "outputs/annotations"):
 """
 Initialize rapid annotation workflow

 Args:
 uncertain_tiles_csv: CSV file with uncertain tiles ranked by uncertainty
 predictions_dir: Directory with prediction results
 sar_images_dir: Directory with original SAR images
 annotations_dir: Output directory for annotations
 """
 self.uncertain_tiles_csv = Path(uncertain_tiles_csv)
 self.predictions_dir = Path(predictions_dir)
 self.sar_images_dir = Path(sar_images_dir)
 self.annotations_dir = Path(annotations_dir)

 # Create annotations directory
 self.annotations_dir.mkdir(parents=True, exist_ok=True)

 # Load uncertain tiles
 self.uncertain_tiles = self.load_uncertain_tiles()

 # Annotation session metadata
 self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
 self.annotations = []

 print(f"Rapid Annotation Session: {self.session_id}")
 print(f"{len(self.uncertain_tiles)} uncertain tiles loaded")
 print("=" * 60)

 def load_uncertain_tiles(self) -> List[Dict]:
 """Load uncertain tiles from CSV"""
 import pandas as pd

 if not self.uncertain_tiles_csv.exists():
 print(f"ERROR: Uncertain tiles CSV not found: {self.uncertain_tiles_csv}")
 return []

 df = pd.read_csv(self.uncertain_tiles_csv)
 return df.to_dict('records')

 def display_tile_info(self, tile_info: Dict, tile_idx: int) -> None:
 """Display key information about the tile"""
 print(f"\n📍 TILE {tile_idx + 1}/{len(self.uncertain_tiles)}")
 print("=" * 50)
 print(f"🏞️ File: {tile_info['file_name']}")
 print(f"Confidence: {tile_info['confidence']:.3f}")
 print(f"Uncertainty: {tile_info['uncertainty_score']:.3f}")
 print(f"🌊 Model Prediction: {tile_info['flood_percentage']:.1f}% flood")

 # Quick visual assessment
 if tile_info['confidence'] < 0.6:
 print("WARNING: LOW CONFIDENCE - High priority for review")
 elif tile_info['flood_percentage'] > 20:
 print("🌊 HIGH FLOOD COVERAGE - Verify extent")
 else:
 print(f"AMBIGUOUS CASE - Check for subtle flooding")

 def get_annotation_paths(self, tile_info: Dict) -> Dict[str, Path]:
 """Get paths to all relevant files for a tile"""
 base_name = tile_info['file_name'].replace('.tif', '')

 return {
 'sar_image': self.sar_images_dir / tile_info['file_name'],
 'mask_pred': self.predictions_dir / f"{base_name}_mask.png",
 'proba_map': self.predictions_dir / f"{base_name}_proba.png",
 'json_file': Path(tile_info['json_file'])
 }

 def display_files_info(self, paths: Dict[str, Path]) -> bool:
 """Check and display file availability"""
 available = True

 for file_type, path in paths.items():
 if path.exists():
 size_mb = path.stat().st_size / (1024 * 1024)
 print(f"{file_type}: {path.name} ({size_mb:.1f}MB)")
 else:
 print(f"ERROR: {file_type}: NOT FOUND - {path}")
 available = False

 return available

 def get_rapid_annotation(self, tile_info: Dict) -> Optional[Dict]:
 """
 Get rapid annotation from user (target: <30 seconds)

 Returns:
 Annotation dictionary or None if skipped
 """
 print("\n🚀 RAPID ANNOTATION (Target: <30 seconds)")
 print("=" * 40)
 print("Options:")
 print(f" 1 = CORRECT - Model prediction is good")
 print(f" 2 = WRONG - Model prediction is clearly wrong")
 print(f" 3 = PARTIAL - Model got some but missed areas")
 print(f" 4 = AMBIGUOUS - Hard to determine ground truth")
 print(f" 5 = SKIP - Skip this tile for now")
 print(f" q = QUIT - End annotation session")

 start_time = datetime.now()

 while True:
 choice = input("\n👉 Your assessment (1-5, q): ").strip().lower()

 if choice == 'q':
 return None

 if choice in ['1', '2', '3', '4', '5']:
 break

 print("ERROR: Invalid input. Please enter 1-5 or 'q'")

 # Calculate annotation time
 annotation_time = (datetime.now() - start_time).total_seconds()

 # Map choices to labels
 choice_mapping = {
 '1': 'correct',
 '2': 'wrong',
 '3': 'partial',
 '4': 'ambiguous',
 '5': 'skip'
 }

 if choice == '5': # Skip
 print("⏭️ Skipped")
 return {'action': 'skip'}

 # Get optional comment for non-correct predictions
 comment = ""
 if choice in ['2', '3', '4']:
 comment = input("💬 Optional comment (press Enter to skip): ").strip()

 annotation = {
 'tile_name': tile_info['file_name'],
 'confidence': tile_info['confidence'],
 'uncertainty_score': tile_info['uncertainty_score'],
 'model_flood_percentage': tile_info['flood_percentage'],
 'assessment': choice_mapping[choice],
 'comment': comment,
 'annotation_time_seconds': annotation_time,
 'annotated_at': datetime.now().isoformat(),
 'session_id': self.session_id
 }

 # Performance feedback
 if annotation_time <= 30:
 print(f"⚡ Great! Annotated in {annotation_time:.1f}s (target: <30s)")
 else:
 print(f"⏰ Annotation took {annotation_time:.1f}s (target: <30s)")

 return annotation

 def save_annotation(self, annotation: Dict) -> None:
 """Save individual annotation"""
 if annotation.get('action') == 'skip':
 return

 self.annotations.append(annotation)

 # Save individual annotation file
 tile_name = annotation['tile_name'].replace('.tif', '')
 annotation_file = self.annotations_dir / f"{tile_name}_annotation.json"

 with open(annotation_file, 'w') as f:
 json.dump(annotation, f, indent=2)

 def save_session_summary(self) -> None:
 """Save complete annotation session summary"""
 if not self.annotations:
 print("📝 No annotations to save")
 return

 # Calculate session statistics
 total_time = sum(a['annotation_time_seconds'] for a in self.annotations)
 avg_time = total_time / len(self.annotations)
 fast_annotations = sum(1 for a in self.annotations if a['annotation_time_seconds'] <= 30)

 session_summary = {
 'session_id': self.session_id,
 'total_annotations': len(self.annotations),
 'total_time_seconds': total_time,
 'average_time_seconds': avg_time,
 'target_achieved_count': fast_annotations,
 'target_achievement_rate': fast_annotations / len(self.annotations) * 100,
 'annotations': self.annotations,
 'created_at': datetime.now().isoformat()
 }

 # Assessment breakdown
 assessments = {}
 for annotation in self.annotations:
 assessment = annotation['assessment']
 assessments[assessment] = assessments.get(assessment, 0) + 1
 session_summary['assessment_breakdown'] = assessments

 # Save session summary
 summary_file = self.annotations_dir / f"session_{self.session_id}_summary.json"
 with open(summary_file, 'w') as f:
 json.dump(session_summary, f, indent=2)

 print(f"\n SESSION SUMMARY")
 print("=" * 30)
 print(f"Total annotations: {len(self.annotations)}")
 print(f"Average time: {avg_time:.1f}s per tile")
 print(f"Target achieved: {fast_annotations}/{len(self.annotations)} ({fast_annotations/len(self.annotations)*100:.1f}%)")
 print(f"Assessment breakdown: {assessments}")
 print(f"Session saved: {summary_file}")

 def run_annotation_session(self, max_tiles: int = 10, start_idx: int = 0) -> None:
 """
 Run rapid annotation session

 Args:
 max_tiles: Maximum number of tiles to annotate
 start_idx: Starting index in uncertain tiles list
 """
 if not self.uncertain_tiles:
 print("ERROR: No uncertain tiles available")
 return

 print(f"Starting rapid annotation session")
 print(f"Annotating up to {max_tiles} tiles (starting from #{start_idx + 1})")
 print(f"⏱️ Target: <30 seconds per tile")

 end_idx = min(start_idx + max_tiles, len(self.uncertain_tiles))

 for i, tile_info in enumerate(self.uncertain_tiles[start_idx:end_idx], start_idx):
 # Display tile information
 self.display_tile_info(tile_info, i)

 # Check file availability
 paths = self.get_annotation_paths(tile_info)
 files_available = self.display_files_info(paths)

 if not files_available:
 print("WARNING: Skipping due to missing files")
 continue

 # Instruction for human reviewer
 print(f"\n REVIEW INSTRUCTIONS:")
 print(f"1. Check original SAR: {paths['sar_image'].name}")
 print(f"2. Review model mask: {paths['mask_pred'].name}")
 print(f"3. Check probability: {paths['proba_map'].name}")
 print(f"4. Make quick assessment based on flood patterns")

 # Get rapid annotation
 annotation = self.get_rapid_annotation(tile_info)

 if annotation is None: # Quit requested
 break

 # Save annotation
 self.save_annotation(annotation)

 # Save session summary
 self.save_session_summary()


def main():
 parser = argparse.ArgumentParser(description='Rapid annotation workflow for active learning')
 parser.add_argument('--uncertain_tiles', '-u', type=str,
 default='outputs/uncertainty_analysis/uncertain_tiles_for_review.csv',
 help='CSV file with uncertain tiles')
 parser.add_argument('--predictions', '-p', type=str, default='outputs/predictions',
 help='Directory with prediction results')
 parser.add_argument('--sar_images', '-s', type=str,
 default='/Users/aoberai/Documents/SARFlood/SEN_DATA/v1.1/data/flood_events/HandLabeled/S1Hand',
 help='Directory with original SAR images')
 parser.add_argument('--annotations', '-a', type=str, default='outputs/annotations',
 help='Output directory for annotations')
 parser.add_argument('--max_tiles', '-m', type=int, default=10,
 help='Maximum number of tiles to annotate')
 parser.add_argument('--start_idx', '-i', type=int, default=0,
 help='Starting index in uncertain tiles list')

 args = parser.parse_args()

 # Create rapid annotator
 annotator = RapidAnnotator(
 uncertain_tiles_csv=args.uncertain_tiles,
 predictions_dir=args.predictions,
 sar_images_dir=args.sar_images,
 annotations_dir=args.annotations
 )

 # Run annotation session
 annotator.run_annotation_session(
 max_tiles=args.max_tiles,
 start_idx=args.start_idx
 )

 print(f"\n Rapid annotation session completed!")
 print(f"Ready for model retraining with human feedback")


if __name__ == "__main__":
 main()