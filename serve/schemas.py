"""
DrShym Climate API Schemas - Clean version
Professional flood segmentation API schemas for researchers
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime


class Provenance(BaseModel):
    """Processing provenance information"""
    source_uri: str = Field(..., description="Original data source URI")
    processing: List[str] = Field(default_factory=list, description="Processing steps applied")
    model_version: Optional[str] = Field(None, description="Model version used")
    timestamp: Optional[datetime] = Field(None, description="Processing timestamp")

    class Config:
        schema_extra = {
            "example": {
                "source_uri": "s3://sen1floods11/S1_20210111_T1234.tif",
                "processing": ["speckle_filter:lee", "normalize:0-1"],
                "model_version": "resnet50-unet-v1.0",
                "timestamp": "2023-05-15T10:30:00Z"
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
                    "processing": ["speckle_filter:lee", "normalize:0-1"],
                    "model_version": "resnet50-unet-v1.0",
                    "timestamp": "2023-05-15T10:30:00Z"
                },
                "label_set": ["flood", "water", "land"],
                "created_at": "2023-05-15T10:30:00Z",
                "schema_version": "0.1"
            }
        }


class SegmentationResponse(BaseModel):
    """Response schema for segmentation endpoint"""
    success: bool = Field(..., description="Whether the operation succeeded")
    flood_percentage: float = Field(..., description="Percentage of pixels classified as flood")
    confidence: float = Field(..., description="Model confidence score")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    metadata: Optional[DrShymRecord] = Field(None, description="Tile metadata if available")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "flood_percentage": 12.5,
                "confidence": 0.89,
                "processing_time_ms": 850,
                "metadata": {
                    "image_id": "user_upload_001",
                    "modality": "sentinel1_sar_vv",
                    "processing_time": "2023-05-15T10:30:00Z"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format. Expected GeoTIFF with single band.",
                "error_code": "INVALID_FORMAT"
            }
        }