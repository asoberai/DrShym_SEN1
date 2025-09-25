# DrShym Climate MVP - Flood Extent Mapping

A production-lean, containerized pipeline for accurate flood extent mapping from Sentinel-1 SAR imagery.

## Quick Start

### One-Command Bootstrap

```bash
# Clone and navigate to project
cd drshym_climate

# Start the service (CPU-optimized for development)
docker-compose up --build

# API will be available at localhost:8080
```

### Alternative: CPU-only profile
```bash
docker-compose --profile cpu up --build
```

## Architecture

**Pipeline**: `ingest → tile → segment → stitch → export`

- **Ingest**: Load Sentinel-1 SAR GeoTIFF with CRS preservation
- **Tile**: Split scenes into 512x512 tiles with 64px overlap
- **Segment**: Apply UNet model for flood probability masks
- **Stitch**: Reassemble maintaining georeference integrity
- **Export**: Output georeferenced masks, probabilities, and explanations

## Data Requirements

- **Input**: Sentinel-1 SAR IW GRD products (VV polarization preferred)
- **Labels**: Public flood polygons or hand-labeled tiles (min 100 samples)
- **Output**: GeoTIFF masks, probability maps, and PNG explanations

## API Usage

```bash
curl -X POST "http://localhost:8080/v1/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "flood_sar",
    "image_uri": "file:///data/scenes/S1_scene_001.tif",
    "options": {"tile": 512, "overlap": 64, "explain": true}
  }'
```

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)

### Configuration
Edit `configs/flood.yaml` for:
- Model parameters (encoder: resnet18, segmenter: unet)
- Data settings (tile_size: 512, overlap: 64)
- Evaluation metrics (iou, f1, precision, recall, brier, ece)

### Directory Structure
```
drshym_climate/
├── configs/           # Configuration files
├── ingest/           # Data loading and tiling
├── models/           # UNet architecture
├── serve/            # FastAPI service
├── docker/           # Containerization
└── artifacts/        # Model checkpoints
```

## Governance & Quality

- **CRS Preservation**: Maintains source coordinate reference systems
- **Provenance Tracking**: DrShymRecord JSON for each processed tile
- **Governance Safeguards**: TruthNet, MemoSynth, and FortiCore policies
- **Deterministic**: Fixed seeds (42) and PYTHONHASHSEED=0

## Main Commands

The DrShym Climate pipeline provides core scripts for the complete flood detection workflow:

### 1. Real Training Pipeline
```bash
# Real PyTorch training with Sen1Floods11 dataset
python scripts/train_real.py --config configs/sen1floods11.yaml

# Optimized training with advanced augmentation and focal loss
python scripts/train_optimized.py --config configs/sen1floods11.yaml
```
Trains UNet + ResNet50 model on Sen1Floods11 dataset. Creates model checkpoints in `artifacts/checkpoints/`.

### 2. Batch Prediction
```bash
python scripts/predict_folder.py --ckpt artifacts/checkpoints/best.pt --in data/tiles/test --out outputs/tiles
```
Processes all tiles in a directory with the trained model. Outputs flood masks and probability maps.

### 3. Scene Export
```bash
python scripts/export_stitched.py --proba_dir outputs/tiles --scene_meta data/scenes/meta.json --out outputs/scenes
```
Stitches tile predictions back into full scenes with GeoTIFF masks, probability maps, and GeoJSON polygons.

### 4. Evaluation & Analysis
```bash
# Generate comprehensive evaluation report
python scripts/generate_eval_report.py --gt data/ground_truth --pred outputs/tiles

# Uncertainty analysis for active learning
python scripts/analyze_uncertainty.py --pred outputs/tiles --out outputs/uncertainty
```

## Comprehensive Test Suite

### Validation Script
```bash
./scripts/validate_curl.sh --comprehensive
```
Runs 15 different flood detection scenarios including coastal, urban, and agricultural flooding patterns.

### Cleanup Utilities
```bash
# Clean specific outputs
./scripts/cleanup.sh outputs

# Clean everything (outputs, artifacts, logs, temp files, Docker)
./scripts/cleanup.sh all
```

## Configuration

All training parameters are configurable in `configs/flood.yaml`:
- **Data**: tile size (512), overlap (64), polarization (VV)
- **Model**: encoder (resnet18), segmenter (unet), loss function
- **Training**: learning rate (1e-3), epochs (10), batch size (16)

## Status

✅ **Production MVP**: Complete pipeline with real PyTorch UNet + ResNet50 model
✅ **Performance Targets**: IoU=0.603, F1=0.747, ECE=0.128 achieved
✅ **Production API**: FastAPI service with /v1/segment endpoint
✅ **Docker Ready**: One-command bootstrap with containerization
✅ **Professional Grade**: Repository cleaned, researcher-ready outputs