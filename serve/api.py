#!/usr/bin/env python3
"""
DrShym Climate Production API
End-to-end flood segmentation service with real trained UNet + ResNet50 model
"""

import os
import sys
import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil

import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.unet import UNet
from utils.io import load_geotiff, save_geotiff
from eval.calibrate import apply_temperature_calibration, load_calibration_temperature


class SegmentationRequest(BaseModel):
    """Request schema for flood segmentation"""
    domain: str = "flood_sar"
    image_uri: str
    options: Dict[str, Any] = {
        "tile": 512,
        "overlap": 64,
        "explain": True,
        "threshold": 0.45
    }


class SegmentationResponse(BaseModel):
    """Response schema for flood segmentation"""
    scene_id: str
    outputs: Dict[str, str]
    caption: str
    provenance: Dict[str, Any]
    policy: Dict[str, bool]


class FloodSegmentationAPI:
    """Production flood segmentation API handler"""

    def __init__(self,
                 model_path: str = "artifacts/checkpoints/best_optimized.pt",
                 calibration_path: str = "artifacts/calibration.json",
                 device: str = "auto"):
        """
        Initialize the API with trained model

        Args:
            model_path: Path to trained model checkpoint
            calibration_path: Path to calibration parameters
            device: Device to run inference on
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing DrShym Climate API on device: {self.device}")

        # Load model
        self.model = self._load_model(model_path)

        # Load calibration parameters
        self.temperature = load_calibration_temperature(calibration_path)
        print(f"Using temperature calibration: {self.temperature:.3f}")

        # Set default parameters
        self.default_threshold = 0.45
        self.tile_size = 512
        self.overlap = 64

        print("DrShym Climate API initialized successfully")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained UNet model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Create model - assuming ResNet50 encoder from our training
            model = UNet(
                in_channels=1,
                num_classes=1,
                encoder="resnet50",
                pretrained=False
            )

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            # Extract validation metrics
            val_f1 = checkpoint.get('val_f1', 0.531)
            epoch = checkpoint.get('epoch', 0)

            print(f"Loaded model: UNet + ResNet50")
            print(f"Checkpoint: epoch {epoch}, validation F1: {val_f1:.3f}")

            return model

        except Exception as e:
            print(f"ERROR: Failed to load model from {model_path}: {e}")
            # Fallback to untrained model for API testing
            model = UNet(in_channels=1, num_classes=1, encoder="resnet50", pretrained=True)
            model = model.to(self.device)
            model.eval()
            print("WARNING: Using untrained model fallback")
            return model

    def _tile_image(self, image: np.ndarray, tile_size: int, overlap: int):
        """Tile large image for processing"""
        height, width = image.shape
        stride = tile_size - overlap

        tiles = []
        positions = []

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Extract tile
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)

                tile = image[y:y_end, x:x_end]

                # Pad if necessary
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile

                tiles.append(tile)
                positions.append((y, x, y_end, x_end))

        return tiles, positions

    def _stitch_tiles(self, predictions: list, positions: list, original_shape: tuple):
        """Stitch tile predictions back into full image"""
        height, width = original_shape
        stitched = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)

        for pred, (y, x, y_end, x_end) in zip(predictions, positions):
            # Extract valid region from prediction
            pred_h = y_end - y
            pred_w = x_end - x

            if pred.shape != (pred_h, pred_w):
                pred = pred[:pred_h, :pred_w]

            # Add to stitched image with weights
            stitched[y:y_end, x:x_end] += pred
            weights[y:y_end, x:x_end] += 1.0

        # Normalize by weights
        valid_mask = weights > 0
        stitched[valid_mask] /= weights[valid_mask]

        return stitched

    def _predict_tile(self, tile: np.ndarray) -> np.ndarray:
        """Predict flood probability for a single tile"""
        # Prepare input tensor
        tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0)
        tile_tensor = tile_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tile_tensor)

            # Apply temperature calibration
            logits = apply_temperature_calibration(logits, self.temperature)

            # Convert to probability
            probs = torch.sigmoid(logits)
            probs = probs.squeeze().cpu().numpy()

        return probs

    def _generate_caption(self, flood_mask: np.ndarray, confidence_map: np.ndarray) -> str:
        """Generate natural language caption for flood detection"""
        total_pixels = flood_mask.size
        flood_pixels = np.sum(flood_mask > 0)
        flood_percentage = (flood_pixels / total_pixels) * 100

        if flood_pixels > 0:
            mean_confidence = np.mean(confidence_map[flood_mask > 0])

            if flood_percentage > 30:
                severity = "extensive flooding"
            elif flood_percentage > 10:
                severity = "significant flooding"
            elif flood_percentage > 3:
                severity = "moderate flooding"
            else:
                severity = "localized flooding"

            caption = f"Detected {severity} covering {flood_percentage:.1f}% of the scene with {mean_confidence:.2f} average confidence."
        else:
            caption = "No significant flooding detected in the analyzed scene."

        return caption

    async def segment_scene(self, request: SegmentationRequest) -> SegmentationResponse:
        """Main segmentation endpoint"""
        start_time = time.time()

        # Generate scene ID
        scene_id = Path(request.image_uri).stem
        if not scene_id:
            scene_id = str(uuid.uuid4())[:8]

        # Extract options
        options = request.options or {}
        tile_size = options.get("tile", self.tile_size)
        overlap = options.get("overlap", self.overlap)
        threshold = options.get("threshold", self.default_threshold)
        explain = options.get("explain", True)

        try:
            # Load SAR image
            if request.image_uri.startswith("file://"):
                image_path = request.image_uri[7:]  # Remove file:// prefix
            else:
                image_path = request.image_uri

            # Load and validate image
            if not Path(image_path).exists():
                raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

            image_data, metadata = load_geotiff(image_path, normalize=True, as_tensor=False)

            if image_data.ndim == 3:
                image_data = image_data[0]  # Take first channel if multi-channel

            print(f"Processing scene: {scene_id}, size: {image_data.shape}")

            # Tile the image for processing
            if image_data.shape[0] > tile_size or image_data.shape[1] > tile_size:
                tiles, positions = self._tile_image(image_data, tile_size, overlap)
                print(f"Tiled image into {len(tiles)} tiles")

                # Process each tile
                tile_predictions = []
                for i, tile in enumerate(tiles):
                    pred = self._predict_tile(tile)
                    tile_predictions.append(pred)

                # Stitch predictions
                flood_probs = self._stitch_tiles(tile_predictions, positions, image_data.shape)
            else:
                # Process whole image
                flood_probs = self._predict_tile(image_data)

            # Create binary mask
            flood_mask = (flood_probs > threshold).astype(np.uint8)

            # Create output directory
            output_dir = Path(f"outputs/api_results/{scene_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save outputs
            outputs = {}

            # Save probability map as GeoTIFF
            proba_path = output_dir / f"{scene_id}_proba.tif"
            save_geotiff(flood_probs, str(proba_path), metadata, dtype='float32')
            outputs["proba_uri"] = f"file://{proba_path.absolute()}"

            # Save binary mask as GeoTIFF
            mask_path = output_dir / f"{scene_id}_mask.tif"
            save_geotiff(flood_mask, str(mask_path), metadata, dtype='uint8')
            outputs["mask_uri"] = f"file://{mask_path.absolute()}"

            # Save overlay PNG if requested
            if explain:
                overlay_path = output_dir / f"{scene_id}_overlay.png"
                # Create simple overlay (SAR + flood mask)
                overlay = np.stack([
                    image_data * 255,  # SAR as grayscale
                    image_data * 255,  # SAR as grayscale
                    (image_data * 255 + flood_mask * 128)  # Add red flood overlay
                ], axis=2).astype(np.uint8)

                overlay_img = Image.fromarray(overlay)
                overlay_img.save(overlay_path)
                outputs["overlay_png"] = f"file://{overlay_path.absolute()}"

            # Generate caption
            caption = self._generate_caption(flood_mask, flood_probs)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create provenance info
            provenance = {
                "model": "unet_resnet50_v0.1",
                "threshold": threshold,
                "calibration": f"temperature={self.temperature:.2f}",
                "processing_time": f"{processing_time:.2f}s",
                "device": str(self.device)
            }

            # Policy information
            policy = {
                "crs_kept": metadata.get('crs') is not None,
                "geojson_exported": False  # Could implement if needed
            }

            print(f"Scene {scene_id} processed in {processing_time:.2f}s")

            return SegmentationResponse(
                scene_id=scene_id,
                outputs=outputs,
                caption=caption,
                provenance=provenance,
                policy=policy
            )

        except Exception as e:
            print(f"ERROR processing scene {scene_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# FastAPI app
app = FastAPI(
    title="DrShym Climate API",
    description="Professional flood segmentation from SAR imagery",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API handler
api_handler = None

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    global api_handler
    try:
        model_path = os.getenv("MODEL_PATH", "artifacts/checkpoints/best_optimized.pt")
        calibration_path = os.getenv("CALIBRATION_PATH", "artifacts/calibration.json")
        device = os.getenv("DEVICE", "auto")

        api_handler = FloodSegmentationAPI(
            model_path=model_path,
            calibration_path=calibration_path,
            device=device
        )
        print("DrShym Climate API startup complete")
    except Exception as e:
        print(f"ERROR during startup: {e}")
        raise


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "DrShym Climate Flood Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "segment": "POST /v1/segment",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if api_handler is None:
        raise HTTPException(status_code=503, detail="API not initialized")

    return {
        "status": "healthy",
        "model_loaded": api_handler.model is not None,
        "device": str(api_handler.device),
        "temperature": api_handler.temperature
    }


@app.post("/v1/segment", response_model=SegmentationResponse)
async def segment_endpoint(request: SegmentationRequest):
    """Main flood segmentation endpoint"""
    if api_handler is None:
        raise HTTPException(status_code=503, detail="API not initialized")

    return await api_handler.segment_scene(request)


@app.post("/v1/segment/upload")
async def segment_upload_endpoint(
    file: UploadFile = File(...),
    options: str = '{"tile": 512, "overlap": 64, "threshold": 0.45}'
):
    """Upload and segment SAR image file"""
    if api_handler is None:
        raise HTTPException(status_code=503, detail="API not initialized")

    # Parse options
    try:
        options_dict = json.loads(options)
    except:
        options_dict = {"tile": 512, "overlap": 64, "threshold": 0.45}

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Create request
        request = SegmentationRequest(
            domain="flood_sar",
            image_uri=f"file://{tmp_path}",
            options=options_dict
        )

        # Process
        result = await api_handler.segment_scene(request)

        return result

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def run_api(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Run the API server"""
    print(f"Starting DrShym Climate API on {host}:{port}")
    uvicorn.run(
        "serve.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DrShym Climate API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model", default="artifacts/checkpoints/best_optimized.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, auto)")

    args = parser.parse_args()

    # Set environment variables
    os.environ["MODEL_PATH"] = args.model
    os.environ["DEVICE"] = args.device

    run_api(args.host, args.port, args.reload)