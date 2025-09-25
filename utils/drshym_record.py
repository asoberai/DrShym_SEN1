"""
Simple DrShymRecord implementation without pydantic dependency
Implements DrShymRecord v0.1 specification for metadata tracking
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple


class DrShymRecord:
    """
    DrShym Climate Tile Metadata Record v0.1

    Store one JSON per tile beside the image to maintain provenance
    and enable reproducible flood detection workflows.
    """

    def __init__(self,
                 image_id: str,
                 modality: str,
                 crs: str,
                 pixel_spacing: Tuple[float, float],
                 tile_size: Tuple[int, int],
                 bounds: Tuple[float, float, float, float],
                 provenance: Dict[str, Any],
                 label_set: List[str],
                 created_at: datetime = None,
                 schema_version: str = "0.1"):

        self.image_id = image_id
        self.modality = modality
        self.crs = crs
        self.pixel_spacing = pixel_spacing
        self.tile_size = tile_size
        self.bounds = bounds
        self.provenance = provenance
        self.label_set = label_set
        self.created_at = created_at or datetime.utcnow()
        self.schema_version = schema_version

    @classmethod
    def simple(cls,
              tile_id: str,
              source_scene: str,
              bbox: List[float],
              flood_probability: float = None,
              model_version: str = None):
        """
        Simple constructor for basic use cases

        Args:
            tile_id: Unique tile identifier
            source_scene: Source scene filename
            bbox: Bounding box coordinates [xmin, ymin, xmax, ymax]
            flood_probability: Optional flood probability
            model_version: Optional model version
        """
        return cls(
            image_id=tile_id,
            modality="SAR_C_band",
            crs="EPSG:4326",
            pixel_spacing=(10.0, 10.0),
            tile_size=(512, 512),
            bounds=tuple(bbox),
            provenance={
                "source_scene": source_scene,
                "model_version": model_version or "unknown",
                "flood_probability": flood_probability
            },
            label_set=["no_flood", "flood"]
        )

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "image_id": self.image_id,
            "modality": self.modality,
            "crs": self.crs,
            "pixel_spacing": self.pixel_spacing,
            "tile_size": self.tile_size,
            "bounds": self.bounds,
            "provenance": self.provenance,
            "label_set": self.label_set,
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version
        }

    def to_dict(self) -> Dict[str, Any]:
        """Alias for dict() method"""
        return self.dict()

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DrShymRecord':
        """Create DrShymRecord from dictionary"""
        data = data.copy()
        # Convert created_at back to datetime if it's a string
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'DrShymRecord':
        """Create DrShymRecord from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_file(self, file_path: Path) -> None:
        """Save DrShymRecord to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'DrShymRecord':
        """Load DrShymRecord from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Convert created_at back to datetime
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])

        return cls(**data)


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

    # Provenance information
    provenance = {
        "source_uri": str(original_sar_path),
        "processing": processing_steps
    }

    # Create record with Sen1Floods11 specific values
    return DrShymRecord(
        image_id=tile_id,
        modality="sentinel1_sar_vv",
        crs="EPSG:4326",  # Sen1Floods11 uses WGS84
        pixel_spacing=(10.0, 10.0),
        tile_size=(512, 512),
        bounds=(0.0, 0.0, 1.0, 1.0),  # Placeholder bounds for training tiles
        provenance=provenance,
        label_set=["flooded", "non_flooded"],
        created_at=datetime.utcnow(),
        schema_version="0.1"
    )


def create_model_record(
    model_id: str,
    config: Dict[str, Any],
    learning_rate: float
) -> DrShymRecord:
    """
    Create DrShymRecord for trained model

    Args:
        model_id: Unique model identifier
        config: Training configuration
        learning_rate: Learning rate used

    Returns:
        DrShymRecord for the trained model
    """
    # Processing steps for model training
    processing_steps = [
        "data_augmentation:flip_rotate_brightness",
        "loss_function:stable_bce_loss",
        "optimizer:adam",
        f"epochs:{config['model']['epochs']}",
        f"learning_rate:{learning_rate}",
        f"batch_size:{config['model']['batch_size']}"
    ]

    # Provenance information
    provenance = {
        "source_uri": "sen1floods11_handlabeled_dataset",
        "processing": processing_steps
    }

    return DrShymRecord(
        image_id=model_id,
        modality="model_checkpoint",
        crs="N/A",
        pixel_spacing=(1.0, 1.0),
        tile_size=(1, 1),
        bounds=(0.0, 0.0, 1.0, 1.0),
        provenance=provenance,
        label_set=["flooded", "non_flooded"],
        created_at=datetime.utcnow(),
        schema_version="0.1"
    )


# Example usage
if __name__ == "__main__":
    # Test DrShymRecord creation
    example_record = DrShymRecord(
        image_id="S1_20210111_T1234_tile_00042",
        modality="sentinel1_sar_vv",
        crs="EPSG:32644",
        pixel_spacing=(10.0, 10.0),
        tile_size=(512, 512),
        bounds=(123456.0, 1234567.0, 128456.0, 1239567.0),
        provenance={
            "source_uri": "s3://sen1floods11/S1_20210111_T1234.tif",
            "processing": ["speckle_filter:lee", "normalize:0-1"]
        },
        label_set=["flooded", "non_flooded"]
    )

    # Test serialization
    print("DrShymRecord created successfully")
    print(json.dumps(example_record.dict(), indent=2))

    # Test file save/load
    test_path = Path("/tmp/test_record.json")
    example_record.save_to_file(test_path)
    loaded_record = DrShymRecord.load_from_file(test_path)
    print(f"File save/load successful: {loaded_record.image_id}")
    test_path.unlink()  # Clean up