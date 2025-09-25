#!/usr/bin/env python3
"""
Export stitched scene predictions CLI utility for DrShym Climate MVP.
Usage: python scripts/export_stitched.py --proba_dir outputs/tiles --scene_meta data/scenes/meta.json --out outputs/scenes
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import time
from typing import Dict, List, Tuple


def load_scene_metadata(meta_path: str) -> Dict:
    """Load scene metadata from JSON file."""

    if not Path(meta_path).exists():
        print(f"WARNING: Metadata file not found: {meta_path}")
        print("Using default metadata...")

        # Create default metadata
        return {
            "scenes": {
                "default_scene": {
                    "bounds": [0, 0, 1024, 1024],
                    "tile_size": 512,
                    "overlap": 64,
                    "crs": "EPSG:4326",
                    "tiles": []
                }
            }
        }

    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {meta_path}")
        return metadata
    except Exception as e:
        print(f"ERROR: Error loading metadata: {e}")
        return {"scenes": {}}


def find_tile_files(proba_dir: str) -> Dict[str, List[Path]]:
    """Find probability map files organized by scene."""

    proba_path = Path(proba_dir)
    if not proba_path.exists():
        print(f"ERROR: Probability directory not found: {proba_dir}")
        return {}

    # Find all probability files
    proba_files = list(proba_path.glob("*_proba.png"))

    if not proba_files:
        print(f"WARNING: No probability files found in {proba_dir}")
        return {}

    # Group by scene (extract scene name from filename)
    scenes = {}
    for file_path in proba_files:
        # Extract scene name from filename (before first underscore or use full name)
        parts = file_path.stem.split('_')
        if len(parts) >= 2:
            scene_name = parts[0]
        else:
            scene_name = file_path.stem.replace('_proba', '')

        if scene_name not in scenes:
            scenes[scene_name] = []
        scenes[scene_name].append(file_path)

    print(f"Found {len(proba_files)} probability files across {len(scenes)} scenes")
    return scenes


def stitch_tiles(tile_files: List[Path], scene_bounds: Tuple[int, int, int, int],
                tile_size: int = 512, overlap: int = 64) -> np.ndarray:
    """Stitch tiles back into full scene."""

    x_min, y_min, x_max, y_max = scene_bounds
    scene_width = x_max - x_min
    scene_height = y_max - y_min

    print(f"Stitching scene: {scene_width}x{scene_height}")

    # Initialize output array
    stitched = np.zeros((scene_height, scene_width), dtype=np.float32)
    weight_map = np.zeros((scene_height, scene_width), dtype=np.float32)

    for tile_file in tile_files:
        try:
            # Load probability tile
            tile_img = Image.open(tile_file)
            tile_array = np.array(tile_img, dtype=np.float32) / 255.0

            # Extract tile coordinates from filename (basic approach)
            # For now, assume tiles are in order - proper implementation would parse coordinates
            # This is a simplified version for testing

            # For basic testing, place tiles in a grid
            file_idx = int(tile_file.stem.split('_')[-2]) if '_' in tile_file.stem else 0

            # Calculate grid position
            tiles_per_row = max(1, scene_width // (tile_size - overlap))
            row = file_idx // tiles_per_row
            col = file_idx % tiles_per_row

            y_start = row * (tile_size - overlap)
            x_start = col * (tile_size - overlap)

            # Clip to scene bounds
            y_end = min(y_start + tile_size, scene_height)
            x_end = min(x_start + tile_size, scene_width)

            if y_start < scene_height and x_start < scene_width:
                # Resize tile if needed
                tile_h = y_end - y_start
                tile_w = x_end - x_start

                if tile_array.shape != (tile_h, tile_w):
                    tile_img_resized = tile_img.resize((tile_w, tile_h), Image.LANCZOS)
                    tile_array = np.array(tile_img_resized, dtype=np.float32) / 255.0

                # Add to stitched image with weights
                stitched[y_start:y_end, x_start:x_end] += tile_array
                weight_map[y_start:y_end, x_start:x_end] += 1.0

        except Exception as e:
            print(f"Warning: Failed to process tile {tile_file}: {e}")

    # Normalize by weights
    valid_mask = weight_map > 0
    stitched[valid_mask] /= weight_map[valid_mask]

    return stitched


def export_scene(scene_name: str, stitched_proba: np.ndarray, output_dir: Path,
                threshold: float = 0.5) -> Dict:
    """Export stitched scene as various formats."""

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "scene_name": scene_name,
        "shape": stitched_proba.shape,
        "outputs": {}
    }

    try:
        # Export probability map as PNG
        proba_path = output_dir / f"{scene_name}_proba.png"
        proba_img = Image.fromarray((stitched_proba * 255).astype(np.uint8))
        proba_img.save(proba_path)
        results["outputs"]["proba_png"] = str(proba_path)

        # Export binary mask
        binary_mask = (stitched_proba > threshold).astype(np.uint8)
        mask_path = output_dir / f"{scene_name}_mask.png"
        mask_img = Image.fromarray(binary_mask * 255)
        mask_img.save(mask_path)
        results["outputs"]["mask_png"] = str(mask_path)

        # Calculate flood statistics
        total_pixels = stitched_proba.size
        flood_pixels = np.sum(binary_mask)
        flood_percentage = (flood_pixels / total_pixels) * 100

        results["statistics"] = {
            "total_pixels": int(total_pixels),
            "flood_pixels": int(flood_pixels),
            "flood_percentage": float(flood_percentage),
            "mean_confidence": float(np.mean(stitched_proba[binary_mask == 1])) if flood_pixels > 0 else 0.0
        }

        print(f"Exported scene {scene_name}: {flood_percentage:.1f}% flood coverage")

    except Exception as e:
        print(f"ERROR: Failed to export scene {scene_name}: {e}")
        results["error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Export stitched flood predictions from tiles'
    )
    parser.add_argument('--proba-dir', type=str, required=True,
                       help='Directory containing probability tile outputs')
    parser.add_argument('--scene-meta', type=str,
                       help='Scene metadata JSON file (optional)')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for stitched scenes')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask (default: 0.5)')

    args = parser.parse_args()

    print("DrShym Climate - Export Stitched Scenes")
    print("=" * 40)
    print(f"Input: {args.proba_dir}")
    print(f"Output: {args.out}")
    print(f"Threshold: {args.threshold}")

    # Load scene metadata if provided
    metadata = {}
    if args.scene_meta and Path(args.scene_meta).exists():
        metadata = load_scene_metadata(args.scene_meta)

    # Find tile files
    tile_groups = find_tile_files(args.proba_dir)

    if not tile_groups:
        print("ERROR: No tile files found")
        return 1

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each scene
    results = []

    for scene_name, tile_files in tile_groups.items():
        print(f"\nProcessing scene: {scene_name} ({len(tile_files)} tiles)")

        # Get scene bounds from metadata or use defaults
        scene_info = metadata.get("scenes", {}).get(scene_name, {})
        bounds = scene_info.get("bounds", [0, 0, 1024, 1024])
        tile_size = scene_info.get("tile_size", 512)
        overlap = scene_info.get("overlap", 64)

        # Stitch tiles
        start_time = time.time()
        stitched = stitch_tiles(tile_files, bounds, tile_size, overlap)
        stitch_time = time.time() - start_time

        print(f"Stitching completed in {stitch_time:.1f}s")

        # Export scene
        scene_results = export_scene(scene_name, stitched, output_dir, args.threshold)
        scene_results["processing_time"] = stitch_time
        results.append(scene_results)

    # Save summary results
    summary_file = output_dir / "export_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_scenes": len(results),
            "scenes": results,
            "parameters": {
                "threshold": args.threshold,
                "input_dir": args.proba_dir,
                "output_dir": args.out
            }
        }, f, indent=2)

    print(f"\nExport completed!")
    print(f"Processed {len(results)} scenes")
    print(f"Summary saved to: {summary_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())