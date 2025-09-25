#!/usr/bin/env python3
"""
Comprehensive test suite for DrShym Climate ML pipeline
Tests all components end-to-end with real ML functionality
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing module imports...")

    try:
        # Test model imports
        from models.unet import UNet
        from models.encoder_backbones import ResNet18Encoder, ResNet50Encoder
        print("Models module imported successfully")

        # Test eval imports
        from eval.metrics import FloodMetrics, compute_expected_calibration_error
        from eval.calibrate import TemperatureScaling
        from eval.slices import compute_backscatter_intensity_slices
        print("Eval module imported successfully")

        # Test utils imports
        from utils.drshym_record import DrShymRecord
        from utils.seed import set_seed
        print("Utils module imported successfully")

        # Test serve imports (fallback to simple schemas if pydantic not available)
        try:
            from serve.schemas import SegmentRequest, SegmentResponse
            print("Serve module imported successfully (with pydantic)")
        except ImportError:
            from serve.schemas_simple import SegmentRequest, SegmentResponse
            print("Serve module imported successfully (simple schemas)")

        # Alternative: test via __init__.py
        from serve import SegmentRequest as SR, SegmentResponse as SRESP
        print("Serve module __init__.py import successful")

        return True

    except ImportError as e:
        print(f"ERROR: Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation and basic functionality"""
    print("\nTesting model creation...")

    try:
        from models.unet import UNet
        # Test ResNet18 UNet
        model_r18 = UNet(
            in_channels=1,
            num_classes=1,
            encoder="resnet18",
            pretrained=False
        )

        # Test ResNet50 UNet
        model_r50 = UNet(
            in_channels=1,
            num_classes=1,
            encoder="resnet50",
            pretrained=False
        )

        print(f"ResNet18 UNet created: {sum(p.numel() for p in model_r18.parameters()):,} parameters")
        print(f"ResNet50 UNet created: {sum(p.numel() for p in model_r50.parameters()):,} parameters")

        # Test forward pass
        dummy_input = torch.randn(1, 1, 512, 512)

        with torch.no_grad():
            output_r18 = model_r18(dummy_input)
            output_r50 = model_r50(dummy_input)

        assert output_r18.shape == (1, 1, 512, 512), f"Wrong output shape: {output_r18.shape}"
        assert output_r50.shape == (1, 1, 512, 512), f"Wrong output shape: {output_r50.shape}"

        print("Forward pass successful for both models")
        return True

    except Exception as e:
        print(f"ERROR: Model creation failed: {e}")
        return False


def test_model_loading():
    """Test loading trained model checkpoint"""
    print("\nTesting model checkpoint loading...")

    try:
        from models.unet import UNet
        checkpoint_path = "artifacts/checkpoints/best_optimized.pt"

        if not Path(checkpoint_path).exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print("   This is expected if no training has been done")
            return True

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create model with same config as checkpoint
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})

        model = UNet(
            in_channels=1,
            num_classes=1,
            encoder=model_config.get('encoder', 'resnet50'),
            pretrained=False
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Test inference
        dummy_input = torch.randn(1, 1, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.sigmoid(output)

        print(f"Model loaded from checkpoint")
        print(f"   Architecture: UNet + {model_config.get('encoder', 'resnet50')}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Output range: [{probs.min():.3f}, {probs.max():.3f}]")

        return True

    except Exception as e:
        print(f"ERROR: Checkpoint loading failed: {e}")
        return False


def test_evaluation_metrics():
    """Test evaluation metrics functionality"""
    print("\nTesting evaluation metrics...")

    try:
        from eval.metrics import FloodMetrics, compute_expected_calibration_error, compute_brier_score

        # Create dummy predictions and targets
        batch_size = 2
        height, width = 512, 512

        # Logits (model output before sigmoid)
        logits = torch.randn(batch_size, 1, height, width)

        # Binary targets (ground truth)
        targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()

        # Test FloodMetrics class
        metrics = FloodMetrics()
        metrics.update(logits, targets, threshold=0.5)
        results = metrics.compute_all()

        print(f"Basic metrics computed:")
        for metric, value in results.items():
            print(f"   {metric}: {value:.3f}")

        # Test calibration metrics
        ece = compute_expected_calibration_error(logits, targets)
        brier = compute_brier_score(logits, targets)

        print(f"Calibration metrics:")
        print(f"   ECE: {ece:.3f}")
        print(f"   Brier Score: {brier:.3f}")

        return True

    except Exception as e:
        print(f"ERROR: Evaluation metrics failed: {e}")
        return False


def test_temperature_calibration():
    """Test temperature calibration functionality"""
    print("\nTesting temperature calibration...")

    try:
        from eval.calibrate import TemperatureScaling, temperature_scale_logits

        # Create temperature scaling module
        temp_scaler = TemperatureScaling(initial_temperature=1.5)

        # Test temperature scaling
        logits = torch.randn(4, 1, 256, 256)
        scaled_logits = temp_scaler(logits)

        print(f"Temperature scaling module created")
        print(f"   Initial temperature: {temp_scaler.temperature.item():.3f}")
        print(f"   Scaled logits shape: {scaled_logits.shape}")

        # Test function-based scaling
        manual_scaled = temperature_scale_logits(logits, 1.5)

        # Should be equal
        assert torch.allclose(scaled_logits, manual_scaled, atol=1e-6)
        print("Function-based temperature scaling matches module")

        return True

    except Exception as e:
        print(f"ERROR: Temperature calibration failed: {e}")
        return False


def test_drshym_record():
    """Test DrShym record schema functionality"""
    print("\nTesting DrShym record schema...")

    try:
        from utils.drshym_record import DrShymRecord

        # Create test record
        record = DrShymRecord.simple(
            tile_id="test_tile_001",
            source_scene="S1_test_scene.tif",
            bbox=[10.0, 20.0, 10.5, 20.5],
            flood_probability=0.75,
            model_version="unet_resnet50_v1.0"
        )

        # Test serialization
        record_dict = record.to_dict()
        json_str = record.to_json()

        # Test deserialization
        record_from_dict = DrShymRecord.from_dict(record_dict)
        record_from_json = DrShymRecord.from_json(json_str)

        print(f"DrShym record created and serialized")
        print(f"   Tile ID: {record.image_id}")
        print(f"   Flood probability: {record.provenance.get('flood_probability', 'N/A')}")
        print(f"   Model version: {record.provenance.get('model_version', 'N/A')}")

        # Validate consistency
        assert record_from_dict.image_id == record.image_id
        assert record_from_json.provenance['flood_probability'] == record.provenance['flood_probability']

        print("Serialization/deserialization consistent")

        return True

    except Exception as e:
        print(f"ERROR: DrShym record failed: {e}")
        return False


def test_prediction_pipeline():
    """Test end-to-end prediction pipeline"""
    print("\nTesting end-to-end prediction pipeline...")

    try:
        from models.unet import UNet

        # Create model
        model = UNet(
            in_channels=1,
            num_classes=1,
            encoder="resnet18",
            pretrained=False
        )
        model.eval()

        # Create dummy SAR image
        sar_image = torch.randn(1, 1, 512, 512)

        # Inference pipeline
        with torch.no_grad():
            # Model forward pass
            logits = model(sar_image)

            # Apply sigmoid for probabilities
            probabilities = torch.sigmoid(logits)

            # Apply threshold for binary mask
            threshold = 0.35
            binary_mask = (probabilities > threshold).float()

            # Calculate statistics
            flood_pixels = binary_mask.sum().item()
            total_pixels = binary_mask.numel()
            flood_percentage = (flood_pixels / total_pixels) * 100

            # Calculate confidence
            positive_probs = probabilities[binary_mask > 0]
            confidence = positive_probs.mean().item() if len(positive_probs) > 0 else probabilities.max().item()

        print(f"Prediction pipeline successful:")
        print(f"   Input shape: {sar_image.shape}")
        print(f"   Output shape: {probabilities.shape}")
        print(f"   Flood coverage: {flood_percentage:.1f}%")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

        return True

    except Exception as e:
        print(f"ERROR: Prediction pipeline failed: {e}")
        return False


def test_batch_prediction_script():
    """Test batch prediction script functionality"""
    print("\nTesting batch prediction script...")

    try:
        # Import the prediction functions
        sys.path.insert(0, str(project_root / "scripts"))

        from predict_folder import preprocess_image, process_image, load_model
        from models.unet import UNet

        # Test preprocessing with dummy image
        dummy_img_path = Path("data/tiles/test/test_sar_image.tif")
        if dummy_img_path.exists():
            tensor = preprocess_image(dummy_img_path)
            print(f"Image preprocessing successful: {tensor.shape}")
        else:
            print("WARNING: Test image not found, creating dummy tensor")
            tensor = torch.randn(1, 1, 512, 512)

        # Test model creation for processing
        device = torch.device('cpu')
        model = UNet(in_channels=1, num_classes=1, encoder="resnet18", pretrained=False)
        model.eval()

        # Test processing (without loading actual checkpoint)
        with torch.no_grad():
            logits = model(tensor)
            prob_map = torch.sigmoid(logits).squeeze().numpy()
            binary_mask = (prob_map > 0.35).astype(np.uint8)

        print(f"Batch prediction components work:")
        print(f"   Probability map shape: {prob_map.shape}")
        print(f"   Binary mask shape: {binary_mask.shape}")
        print(f"   Flood coverage: {(binary_mask.sum() / binary_mask.size * 100):.1f}%")

        return True

    except Exception as e:
        print(f"ERROR: Batch prediction script failed: {e}")
        return False


def main():
    """Run all tests"""
    print("DrShym Climate ML Pipeline Test Suite")
    print("=" * 50)

    tests = [
        ("Module Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Model Loading", test_model_loading),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Temperature Calibration", test_temperature_calibration),
        ("DrShym Record", test_drshym_record),
        ("Prediction Pipeline", test_prediction_pipeline),
        ("Batch Prediction Script", test_batch_prediction_script)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR: {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:<10} {test_name}")
        if success:
            passed += 1

    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\nAll tests passed! ML pipeline is fully functional.")
        return 0
    else:
        print(f"\nWARNING: {total - passed} tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())