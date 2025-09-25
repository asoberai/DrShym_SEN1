"""
I/O utilities for loading and saving GeoTIFF files
"""

import rasterio
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def load_geotiff(file_path: str,
                 normalize: bool = True,
                 as_tensor: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load GeoTIFF file and return array with metadata

    Args:
        file_path: Path to GeoTIFF file
        normalize: Whether to normalize values to [0, 1]
        as_tensor: Whether to return as PyTorch tensor

    Returns:
        Tuple of (data, metadata)
    """
    try:
        with rasterio.open(file_path) as src:
            # Read data
            data = src.read()

            # Get metadata
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'shape': data.shape,
                'dtype': data.dtype,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
                'count': src.count
            }

            # Handle multiple channels - take first channel for SAR
            if len(data.shape) == 3 and data.shape[0] > 1:
                data = data[0] # Take first channel
            elif data.shape[0] == 1:
                data = data[0]

            # Always ensure data is float32 first
            data = data.astype(np.float32)

            # Clean invalid values FIRST before normalization
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Cleaning invalid values in {file_path}")
                data[np.isnan(data)] = 0.0
                data[np.isposinf(data)] = np.finfo(np.float32).max / 1000
                data[np.isneginf(data)] = np.finfo(np.float32).min / 1000

            # Normalize if requested
            if normalize:
                if data.dtype in [np.uint8, np.uint16]:
                    if data.dtype == np.uint8:
                        data = data / 255.0
                    elif data.dtype == np.uint16:
                        data = data / 65535.0
                else:
                    # For SAR float data, use robust normalization
                    try:
                        # Use percentile-based normalization
                        p1, p99 = np.percentile(data, [1, 99])
                        if p99 > p1:
                            data = np.clip(data, p1, p99)
                            data = (data - p1) / (p99 - p1)
                        else:
                            data_min, data_max = data.min(), data.max()
                            if data_max > data_min:
                                data = (data - data_min) / (data_max - data_min)
                    except Exception:
                        data = np.clip(data, 0, 1)

            # Final safety check
            if np.isnan(data).any() or np.isinf(data).any():
                data = np.nan_to_num(data, nan=0.5, posinf=1.0, neginf=0.0)

            # Convert to tensor if requested
            if as_tensor:
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                data = torch.from_numpy(data.copy())
                if len(data.shape) == 2:
                    data = data.unsqueeze(0)

            return data, metadata

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return dummy data for robustness
        dummy_data = np.zeros((512, 512), dtype=np.float32)
        if as_tensor:
            dummy_data = torch.from_numpy(dummy_data.copy()).unsqueeze(0)

        metadata = {
            'crs': None,
            'transform': None,
            'shape': dummy_data.shape,
            'dtype': dummy_data.dtype,
            'error': str(e)
        }

        return dummy_data, metadata


def save_geotiff(data: np.ndarray,
                 file_path: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 dtype: str = 'uint8') -> bool:
    """
    Save array as GeoTIFF file

    Args:
        data: Data array to save
        file_path: Output file path
        metadata: GeoTIFF metadata (CRS, transform, etc.)
        dtype: Output data type

    Returns:
        Success status
    """
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Handle tensor input
        if torch.is_tensor(data):
            data = data.cpu().numpy()

        # Ensure correct shape
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]

        # Convert data type
        if dtype == 'uint8':
            data = (data * 255).astype(np.uint8)
        elif dtype == 'uint16':
            data = (data * 65535).astype(np.uint16)

        # Set up rasterio profile
        profile = {
            'driver': 'GTiff',
            'height': data.shape[1],
            'width': data.shape[2],
            'count': data.shape[0],
            'dtype': data.dtype,
            'compress': 'lzw'
        }

        # Add metadata if provided
        if metadata:
            if metadata.get('crs'):
                profile['crs'] = metadata['crs']
            if metadata.get('transform'):
                profile['transform'] = metadata['transform']

        # Write file
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(data)

        return True

    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def load_sar_label_pair(sar_path: str,
                        label_path: str,
                        target_size: Tuple[int, int] = (512, 512)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load SAR image and corresponding label, resize to target size

    Args:
        sar_path: Path to SAR image
        label_path: Path to label image
        target_size: Target size (H, W)

    Returns:
        Tuple of (sar_tensor, label_tensor)
    """
    # Load SAR image
    sar_data, sar_meta = load_geotiff(sar_path, normalize=True, as_tensor=True)

    # Load label
    label_data, label_meta = load_geotiff(label_path, normalize=False, as_tensor=True)

    # Ensure SAR data is single channel
    if len(sar_data.shape) == 3 and sar_data.shape[0] > 1:
        sar_data = sar_data[0:1]
    elif len(sar_data.shape) == 2:
        sar_data = sar_data.unsqueeze(0)

    # Ensure label is single channel
    if len(label_data.shape) == 3 and label_data.shape[0] > 1:
        label_data = label_data[0:1]
    elif len(label_data.shape) == 2:
        label_data = label_data.unsqueeze(0)

    # Resize if needed
    if sar_data.shape[-2:] != target_size:
        sar_data = torch.nn.functional.interpolate(
            sar_data.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0)

    if label_data.shape[-2:] != target_size:
        label_data = torch.nn.functional.interpolate(
            label_data.unsqueeze(0).float(), size=target_size, mode='nearest'
        ).squeeze(0)

    # Ensure label is binary
    label_data = (label_data > 0).float()

    return sar_data, label_data