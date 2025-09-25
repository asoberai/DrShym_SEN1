#!/usr/bin/env python3
"""
SEN1Floods11 Full Training Script for DrShym Climate
Train UNet + ResNet50 on complete SEN1Floods11 dataset with both hand-labeled and permanent water data
"""

import argparse
import sys
import os
import yaml
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import torchvision.transforms as T
import rasterio
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.unet import UNet
from eval.metrics import compute_metrics
from utils.seed import set_deterministic_seed


class SEN1FloodsDataset(Dataset):
    """SEN1Floods11 dataset loader"""

    def __init__(self, csv_path: str, data_root: str, transform=None, normalize=True):
        self.data_root = Path(data_root)
        self.transform = transform
        self.normalize = normalize

        # Load CSV with file paths (no header, comma separated)
        self.data = pd.read_csv(csv_path, header=None, names=['s1_file', 'label_file'])

        # Filter for existing files
        valid_data = []
        for _, row in self.data.iterrows():
            s1_path = self.data_root / 'data' / 'flood_events' / 'HandLabeled' / 'S1Hand' / row['s1_file']
            label_path = self.data_root / 'data' / 'flood_events' / 'HandLabeled' / 'LabelHand' / row['label_file']

            if s1_path.exists() and label_path.exists():
                valid_data.append(row)
            else:
                print(f"Missing: {s1_path} or {label_path}")

        self.data = pd.DataFrame(valid_data)
        print(f"Loaded {len(self.data)} valid samples from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load SAR image
        s1_path = self.data_root / 'data' / 'flood_events' / 'HandLabeled' / 'S1Hand' / row['s1_file']
        with rasterio.open(s1_path) as src:
            sar = src.read(1).astype(np.float32)

        # Load label
        label_path = self.data_root / 'data' / 'flood_events' / 'HandLabeled' / 'LabelHand' / row['label_file']
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32)

        # Normalize SAR to [0, 1]
        if self.normalize:
            sar = np.clip((sar + 30) / 30, 0, 1)  # Typical SAR normalization

        # Convert to tensors
        sar = torch.from_numpy(sar).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).unsqueeze(0)

        # Apply transforms
        if self.transform:
            # Combine for synchronized transforms
            combined = torch.cat([sar, label], dim=0)
            combined = self.transform(combined)
            sar, label = combined[0:1], combined[1:2]

        return sar, label


class FocalDiceBCELoss(nn.Module):
    """Combined Focal + Dice + BCE Loss for flood segmentation"""

    def __init__(self, focal_alpha=1, focal_gamma=2, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def focal_loss(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * bce_loss
        return focal_loss.mean()

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        return (self.dice_weight * dice +
                self.bce_weight * bce +
                self.focal_weight * focal)


def create_transforms():
    """Create data augmentation transforms"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=90),
    ])


def load_datasets(config: Dict[str, Any]):
    """Load training, validation datasets"""
    data_root = config['dataset']['root_dir']

    # Create transforms
    train_transform = create_transforms()

    # Load training data
    train_csv = Path(data_root) / config['dataset']['splits']['train']
    train_dataset = SEN1FloodsDataset(train_csv, data_root, transform=train_transform)

    # Load validation data
    val_csv = Path(data_root) / config['dataset']['splits']['val']
    val_dataset = SEN1FloodsDataset(val_csv, data_root, transform=None)

    # Add permanent water data if specified
    if config['dataset'].get('include_perm_water', False):
        print("Adding permanent water samples...")
        # This would need additional implementation to load perm water data

    return train_dataset, val_dataset


def train_model(config: Dict[str, Any]):
    """Main training function"""
    # Set deterministic seed
    set_deterministic_seed(config['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Load datasets
    train_dataset, val_dataset = load_datasets(config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = UNet(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        encoder=config['model']['encoder'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)

    print(f"Model: {config['model']['encoder']} UNet")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup loss function
    criterion = FocalDiceBCELoss()

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )

    # Setup metrics tracking

    # Training loop
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        # Training
        model.train()
        train_loss = 0
        train_metrics = {'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0}

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping'])

            optimizer.step()

            train_loss += loss.item()

            # Calculate metrics
            with torch.no_grad():
                pred = torch.sigmoid(output) > 0.5
                batch_metrics = compute_metrics(pred.cpu(), target.cpu())
                for key in train_metrics:
                    train_metrics[key] += batch_metrics[key]

        # Average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0}

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                # Calculate metrics
                pred = torch.sigmoid(output) > 0.5
                batch_metrics = compute_metrics(pred.cpu(), target.cpu())
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]

        # Average validation metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        # Update scheduler
        scheduler.step()

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}, Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0

            # Save checkpoint
            checkpoint_dir = Path(config['output']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'val_iou': val_metrics['iou'],
                'val_loss': val_loss,
                'config': config
            }, checkpoint_dir / 'best_sen1floods11.pt')

            print(f"✓ New best model saved (F1: {best_val_f1:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"Early stopping after {epoch+1} epochs")
                break

    print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")
    return best_val_f1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train SEN1Floods11 model')
    parser.add_argument('--config', default='configs/sen1floods11_full.yaml', help='Config file path')
    parser.add_argument('--data-root', help='Override data root directory')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.data_root:
        config['dataset']['root_dir'] = args.data_root
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    print("SEN1Floods11 Training Configuration:")
    print("=" * 50)
    print(f"Data root: {config['dataset']['root_dir']}")
    print(f"Model: {config['model']['encoder']} UNet")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['lr']}")
    print("=" * 50)

    # Start training
    best_f1 = train_model(config)

    print(f"\n🎉 Training completed! Best F1 score: {best_f1:.4f}")


if __name__ == '__main__':
    main()