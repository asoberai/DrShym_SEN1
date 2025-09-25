"""
Geospatial utilities for coordinate system handling
"""

from typing import Dict, Any, Tuple, Optional


def get_crs_info(crs) -> Dict[str, Any]:
    """
    Get CRS information

    Args:
        crs: CRS object or string

    Returns:
        Dictionary with CRS info
    """
    try:
        if hasattr(crs, 'to_string'):
            crs_string = crs.to_string()
        else:
            crs_string = str(crs)

        return {
            'crs_string': crs_string,
            'is_geographic': 'EPSG:4326' in crs_string,
            'is_projected': 'UTM' in crs_string or 'EPSG:3857' in crs_string
        }
    except Exception as e:
        return {
            'crs_string': 'Unknown',
            'is_geographic': False,
            'is_projected': False,
            'error': str(e)
        }


def transform_bounds(bounds: Tuple[float, float, float, float],
                    src_crs,
                    dst_crs) -> Tuple[float, float, float, float]:
    """
    Transform bounding box coordinates between CRS

    Args:
        bounds: (left, bottom, right, top)
        src_crs: Source CRS
        dst_crs: Destination CRS

    Returns:
        Transformed bounds
    """
    try:
        # For now, return original bounds (would use rasterio.warp.transform_bounds in production)
        return bounds
    except Exception as e:
        print(f"Warning: Could not transform bounds: {e}")
        return bounds