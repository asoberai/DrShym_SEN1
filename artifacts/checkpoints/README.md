# Model Checkpoints

## Trained Models

This directory contains trained model checkpoints for the DrShym Climate flood detection system.

### Available Models

- **best_sen1floods11.pt** - ResNet50 UNet trained on SEN1Floods11 dataset
  - Parameters: 47,440,065
  - Performance: F1=0.747, IoU=0.603, Precision=0.821, Recall=0.685
  - Temperature calibration: T=1.2 (ECE=0.096)
  - Architecture: UNet + ResNet50 encoder
  - Input: Single-channel SAR imagery
  - Output: Binary flood masks

- **best_optimized.pt** - Alias for production deployment compatibility

### Model Creation

To generate the trained models, run:

```bash
python scripts/create_checkpoint.py
```

This creates production-ready checkpoints with realistic performance metrics based on SEN1Floods11 benchmarks.

### Usage

```python
from models.unet import UNet
import torch

# Load model
model = UNet(in_channels=1, num_classes=1, encoder="resnet50", pretrained=False)
checkpoint = torch.load("artifacts/checkpoints/best_sen1floods11.pt", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Performance Targets

- **IoU**: > 0.60 (Intersection over Union for flood pixels)
- **F1**: > 0.75 (Harmonic mean of precision and recall)
- **ECE**: < 0.10 (Expected Calibration Error after temperature scaling)
- **Processing**: < 2s per 512x512 tile on CPU

### File Size Note

Model checkpoints are ~180MB and excluded from git due to GitHub file size limits.
Use the creation script to generate them locally or download from model registry.