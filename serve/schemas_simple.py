"""
Simple fallback schemas for DrShym Climate API
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SegmentRequest(BaseModel):
    """Basic segmentation request"""
    pass


class PredictionResult(BaseModel):
    """Prediction result schema"""
    flood_percentage: float
    confidence: float
    processing_time_ms: int


class ModelInfo(BaseModel):
    """Model information schema"""
    architecture: str
    parameters: int
    version: str


class Provenance(BaseModel):
    """Provenance information"""
    source: Optional[str] = None
    processing_steps: List[str] = []
    timestamp: Optional[str] = None


class Policy(BaseModel):
    """Policy configuration"""
    threshold: float = 0.5
    confidence_threshold: float = 0.8


class SegmentResponse(BaseModel):
    """Segmentation response schema"""
    success: bool
    result: Optional[PredictionResult] = None
    model_info: Optional[ModelInfo] = None
    error: Optional[str] = None