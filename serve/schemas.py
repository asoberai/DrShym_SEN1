#!/usr/bin/env python3
"""
Request and response schemas for the DrShym Climate MVP API.
Defines the exact API contract per CLAUDE.md specification.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
from datetime import datetime
import re
import json


# DrShymRecord v0.1 Schema - Store one JSON per tile for metadata tracking
class Provenance(BaseModel):
    """Processing provenance for DrShym tiles"""
    source_uri: str = Field(..., description="Original source file URI")
    processing: List[str] = Field(..., description="Processing steps applied")

    class Config:
        schema_extra = {
            "example": {
                "source_uri": "s3://sen1floods11/S1_20210111_T1234.tif",
                "processing": ["speckle_filter:lee", "normalize:0-1"]
            }
        }


class DrShymRecord(BaseModel):
 """
 DrShym Climate Tile Metadata Record v0.1

 Store one JSON per tile beside the image to maintain provenance
 and enable reproducible flood detection workflows.
 """
 image_id: str = Field(..., description="Unique tile identifier")
 modality: str = Field(..., description="Data modality (e.g. sentinel1_sar_vv)")
 crs: str = Field(..., description="Coordinate Reference System")
 pixel_spacing: Tuple[float, float] = Field(..., description="Pixel spacing in CRS units [x, y]")
 tile_size: Tuple[int, int] = Field(..., description="Tile dimensions [width, height]")
 bounds: Tuple[float, float, float, float] = Field(..., description="Bounding box [minx, miny, maxx, maxy]")
 provenance: Provenance = Field(..., description="Processing provenance")
 label_set: List[str] = Field(..., description="Available labels for this tile")

 # Optional metadata
 created_at: Optional[datetime] = Field(None, description="Record creation timestamp")
 schema_version: str = Field("0.1", description="DrShymRecord schema version")

 class Config:
 schema_extra = {
 "example": {
 "image_id": "S1_20210111_T1234_tile_00042",
 "modality": "sentinel1_sar_vv",
 "crs": "EPSG:32644",
 "pixel_spacing": [10.0, 10.0],
 "tile_size": [512, 512],
 "bounds": [123456.0, 1234567.0, 128456.0, 1239567.0],
 "provenance": {
 "source_uri": "s3://sen1floods11/S1_20210111_T1234.tif",
 "processing": ["speckle_filter:lee", "normalize:0-1"]
 },
 "label_set": ["flooded", "non_flooded"],
 "created_at": "2024-01-15T10:30:00Z",
 "schema_version": "0.1"
 }
 }

 def save_to_file(self, file_path: Path) -> None:
 """Save DrShymRecord to JSON file"""
 with open(file_path, 'w') as f:
 json.dump(self.dict(), f, indent=2, default=str)

 @classmethod
 def load_from_file(cls, file_path: Path) -> 'DrShymRecord':
 """Load DrShymRecord from JSON file"""
 with open(file_path, 'r') as f:
 data = json.load(f)
 return cls(**data)


class SegmentationOptions(BaseModel):
 """Options for segmentation processing."""
 tile: int = Field(default=512, description="Tile size for processing")
 overlap: int = Field(default=64, description="Overlap between tiles in pixels")
 explain: bool = Field(default=True, description="Generate explainability outputs")
 threshold: Optional[float] = Field(default=None, description="Custom flood threshold (0-1)")

 @validator('tile')
 def validate_tile_size(cls, v):
 if v not in [256, 512, 1024]:
 raise ValueError('tile must be one of: 256, 512, 1024')
 return v

 @validator('overlap')
 def validate_overlap(cls, v):
 if v < 0 or v > 256:
 raise ValueError('overlap must be between 0 and 256')
 return v

 @validator('threshold')
 def validate_threshold(cls, v):
 if v is not None and (v < 0 or v > 1):
 raise ValueError('threshold must be between 0 and 1')
 return v


class SegmentationRequest(BaseModel):
 """Request schema for POST /v1/segment endpoint."""
 domain: str = Field(description="Processing domain identifier")
 image_uri: str = Field(description="URI to input SAR image file")
 options: Optional[SegmentationOptions] = Field(default_factory=SegmentationOptions)

 @validator('domain')
 def validate_domain(cls, v):
 if v != "flood_sar":
 raise ValueError('domain must be "flood_sar"')
 return v

 @validator('image_uri')
 def validate_image_uri(cls, v):
 # Support file:// URIs and direct file paths
 if v.startswith('file://'):
 file_path = v[7:] # Remove 'file://' prefix
 else:
 file_path = v

 path = Path(file_path)
 if not path.suffix.lower() in ['.tif', '.tiff']:
 raise ValueError('image_uri must point to a .tif or .tiff file')

 return v


class SegmentationOutputs(BaseModel):
 """Output file URIs from segmentation."""
 mask_uri: str = Field(description="URI to binary flood mask GeoTIFF")
 proba_uri: str = Field(description="URI to flood probability GeoTIFF")
 overlay_png: Optional[str] = Field(description="URI to explainability overlay PNG")


class ProvenanceInfo(BaseModel):
 """Model provenance and processing metadata."""
 model: str = Field(description="Model identifier and version")
 threshold: float = Field(description="Flood detection threshold used")
 calibration: Optional[str] = Field(description="Calibration method and parameters")
 processing_time: Optional[float] = Field(description="Processing time in seconds")
 tile_count: Optional[int] = Field(description="Number of tiles processed")


class PolicyCompliance(BaseModel):
 """Policy compliance indicators."""
 crs_kept: bool = Field(description="Whether original CRS was preserved")
 geojson_exported: bool = Field(description="Whether GeoJSON polygons were exported")
 governance_passed: bool = Field(default=True, description="Whether governance checks passed")


class SegmentationResponse(BaseModel):
 """Response schema for POST /v1/segment endpoint."""
 scene_id: str = Field(description="Unique identifier for the processed scene")
 outputs: SegmentationOutputs = Field(description="Output file URIs")
 caption: str = Field(description="Natural language description of flood detection results")
 provenance: ProvenanceInfo = Field(description="Processing metadata and model info")
 policy: PolicyCompliance = Field(description="Policy compliance status")
 error: Optional[str] = Field(default=None, description="Error message if processing failed")

 class Config:
 schema_extra = {
 "example": {
 "scene_id": "S1_scene_001",
 "outputs": {
 "mask_uri": "file:///outputs/S1_scene_001_mask.tif",
 "proba_uri": "file:///outputs/S1_scene_001_proba.tif",
 "overlay_png": "file:///outputs/S1_scene_001_overlay.png"
 },
 "caption": "Flooding detected along the river plain on the eastern half, contiguous 1.8 km band near low-slope areas.",
 "provenance": {
 "model": "unet_resnet18_v0.1",
 "threshold": 0.45,
 "calibration": "temperature=1.7",
 "processing_time": 12.3,
 "tile_count": 16
 },
 "policy": {
 "crs_kept": True,
 "geojson_exported": True,
 "governance_passed": True
 }
 }
 }


class HealthResponse(BaseModel):
 """Health check response schema."""
 status: str = Field(description="Service health status")
 model_loaded: bool = Field(description="Whether model is successfully loaded")
 version: str = Field(description="API version")
 uptime_seconds: float = Field(description="Service uptime in seconds")


class ErrorResponse(BaseModel):
 """Error response schema."""
 error: str = Field(description="Error message")
 detail: Optional[str] = Field(description="Detailed error information")
 scene_id: Optional[str] = Field(description="Scene ID if applicable")


def extract_scene_id(image_uri: str) -> str:
 """
 Extract scene ID from image URI.

 Args:
 image_uri: URI to the input image

 Returns:
 Scene ID extracted from filename
 """
 # Remove file:// prefix if present
 if image_uri.startswith('file://'):
 file_path = image_uri[7:]
 else:
 file_path = image_uri

 # Get filename without extension
 filename = Path(file_path).stem

 # Clean scene ID (remove special characters, keep alphanumeric and underscores)
 scene_id = re.sub(r'[^a-zA-Z0-9_]', '_', filename)

 return scene_id


def create_output_uris(scene_id: str, output_dir: str = "/outputs") -> SegmentationOutputs:
 """
 Create standardized output URIs for a scene.

 Args:
 scene_id: Scene identifier
 output_dir: Base output directory

 Returns:
 SegmentationOutputs with file URIs
 """
 base_path = f"file://{output_dir}/{scene_id}"

 return SegmentationOutputs(
 mask_uri=f"{base_path}_mask.tif",
 proba_uri=f"{base_path}_proba.tif",
 overlay_png=f"{base_path}_overlay.png"
 )


if __name__ == "__main__":
 # Test schema validation
 from pydantic import ValidationError

 # Test valid request
 valid_request = {
 "domain": "flood_sar",
 "image_uri": "file:///data/scenes/S1_scene_001.tif",
 "options": {
 "tile": 512,
 "overlap": 64,
 "explain": True
 }
 }

 try:
 request = SegmentationRequest(**valid_request)
 print(f"Valid request: {request.json()}")
 except ValidationError as e:
 print(f"ERROR: Validation error: {e}")

 # Test invalid domain
 invalid_request = valid_request.copy()
 invalid_request["domain"] = "invalid_domain"

 try:
 request = SegmentationRequest(**invalid_request)
 print(f"ERROR: Should have failed: {request.json()}")
 except ValidationError as e:
 print(f"Correctly caught invalid domain: {e}")

 # Test scene ID extraction
 test_uris = [
 "file:///data/S1_scene_001.tif",
 "/data/scenes/S1A_IW_GRDH_1SDV_20230101T120000.tif",
 "complex-scene.name_with-dashes.tif"
 ]

 for uri in test_uris:
 scene_id = extract_scene_id(uri)
 print(f"URI: {uri} -> Scene ID: {scene_id}")

 print(f"Schema tests completed")


# Helper functions for DrShymRecord creation and management
def create_drshym_record(
 image_path: Path,
 tile_id: str,
 crs: str,
 bounds: Tuple[float, float, float, float],
 source_uri: str,
 processing_steps: List[str],
 modality: str = "sentinel1_sar_vv",
 pixel_spacing: Tuple[float, float] = (10.0, 10.0),
 tile_size: Tuple[int, int] = (512, 512),
 label_set: List[str] = None
) -> DrShymRecord:
 """
 Create a DrShymRecord for a processed tile

 Args:
 image_path: Path to the tile image
 tile_id: Unique tile identifier
 crs: Coordinate reference system
 bounds: Bounding box coordinates [minx, miny, maxx, maxy]
 source_uri: Original source file URI
 processing_steps: List of processing steps applied
 modality: Data modality
 pixel_spacing: Pixel spacing in CRS units [x, y]
 tile_size: Tile dimensions [width, height]
 label_set: Available labels

 Returns:
 DrShymRecord instance
 """
 if label_set is None:
 label_set = ["flooded", "non_flooded"]

 provenance = Provenance(
 source_uri=source_uri,
 processing=processing_steps
 )

 return DrShymRecord(
 image_id=tile_id,
 modality=modality,
 crs=crs,
 pixel_spacing=pixel_spacing,
 tile_size=tile_size,
 bounds=bounds,
 provenance=provenance,
 label_set=label_set,
 created_at=datetime.utcnow(),
 schema_version="0.1"
 )


def save_tile_with_metadata(
 image_data,
 image_path: Path,
 record: DrShymRecord,
 save_image_func=None
) -> Path:
 """
 Save tile image and its DrShymRecord metadata

 Args:
 image_data: Image data to save
 image_path: Path for the image file
 record: DrShymRecord metadata
 save_image_func: Function to save image data

 Returns:
 Path to the metadata JSON file
 """
 # Save image if function provided
 if save_image_func:
 save_image_func(image_data, image_path)

 # Save metadata JSON beside the image
 metadata_path = image_path.with_suffix('.json')
 record.save_to_file(metadata_path)

 return metadata_path


def create_training_record(
 tile_path: Path,
 original_sar_path: Path,
 original_label_path: Path
) -> DrShymRecord:
 """
 Create DrShymRecord for training tiles from Sen1Floods11 dataset

 Args:
 tile_path: Path to the processed tile
 original_sar_path: Path to original SAR file
 original_label_path: Path to original label file

 Returns:
 DrShymRecord for the training tile
 """
 # Generate tile ID from filename
 tile_id = tile_path.stem

 # Processing steps for Sen1Floods11 data
 processing_steps = [
 "load_geotiff:sen1floods11",
 "normalize:robust_percentile_1_99",
 "resize:512x512",
 "convert:float32"
 ]

 # Create record with Sen1Floods11 specific values
 return create_drshym_record(
 image_path=tile_path,
 tile_id=tile_id,
 crs="EPSG:4326", # Sen1Floods11 uses WGS84
 bounds=(0.0, 0.0, 1.0, 1.0), # Placeholder bounds for training tiles
 source_uri=str(original_sar_path),
 processing_steps=processing_steps,
 modality="sentinel1_sar_vv",
 pixel_spacing=(10.0, 10.0),
 tile_size=(512, 512),
 label_set=["flooded", "non_flooded"]
 )