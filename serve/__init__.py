"""DrShym Climate Serving Module"""

try:
    from .schemas import (
        SegmentRequest,
        SegmentResponse,
        PredictionResult,
        ModelInfo,
        Provenance,
        Policy
    )
except ImportError:
    # Fallback to simple schemas without pydantic
    from .schemas_simple import (
        SegmentRequest,
        SegmentResponse,
        PredictionResult,
        ModelInfo,
        Provenance,
        Policy
    )

__all__ = [
    'SegmentRequest',
    'SegmentResponse',
    'PredictionResult',
    'ModelInfo',
    'Provenance',
    'Policy'
]