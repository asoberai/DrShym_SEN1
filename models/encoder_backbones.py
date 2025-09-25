"""
Encoder backbones for UNet segmentation models
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class ResNet18Encoder(nn.Module):
    """ResNet18-based encoder for UNet"""

    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super(ResNet18Encoder, self).__init__()

        # Load ResNet18 with proper weights parameter
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)

        # Modify first conv layer for single channel input (SAR)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and in_channels == 1:
            # Initialize with weights from pretrained RGB model (average across channels)
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Channel dimensions for skip connections
        self.channels = [64, 64, 128, 256, 512]

    def forward(self, x):
        """Forward pass returning features at different scales"""
        features = []

        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 64 channels, 1/2 scale

        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)
        features.append(x)  # 64 channels, 1/4 scale

        x = self.layer2(x)
        features.append(x)  # 128 channels, 1/8 scale

        x = self.layer3(x)
        features.append(x)  # 256 channels, 1/16 scale

        x = self.layer4(x)
        features.append(x)  # 512 channels, 1/32 scale

        return features


class ResNet50Encoder(nn.Module):
    """ResNet50-based encoder for UNet with better feature extraction"""

    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super(ResNet50Encoder, self).__init__()

        # Load ResNet50 with proper weights parameter
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)

        # Modify first conv layer for single channel input (SAR)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and in_channels == 1:
            # Initialize with weights from pretrained RGB model (average across channels)
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder layers (ResNet50 has more channels due to bottleneck blocks)
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # Channel dimensions for skip connections [initial, layer1, layer2, layer3, layer4]
        self.channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        """Forward pass returning features at different scales"""
        features = []

        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 64 channels, 1/2 scale

        x = self.maxpool(x)

        # Encoder layers with skip connections
        x = self.layer1(x)
        features.append(x)  # 256 channels, 1/4 scale

        x = self.layer2(x)
        features.append(x)  # 512 channels, 1/8 scale

        x = self.layer3(x)
        features.append(x)  # 1024 channels, 1/16 scale

        x = self.layer4(x)
        features.append(x)  # 2048 channels, 1/32 scale

        return features


class ConvBlock(nn.Module):
    """Basic conv block with BN and ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""

    def __init__(self, in_channels: int, upsample_out_channels: int, out_channels: int, concat_channels: int):
        super(UpBlock, self).__init__()

        # Upsampling - using 'upsample' name to match checkpoint
        self.upsample = nn.ConvTranspose2d(in_channels, upsample_out_channels, kernel_size=2, stride=2)

        # Conv block after concatenation with correct input channels
        self.conv = ConvBlock(concat_channels, out_channels)

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Apply conv block
        return self.conv(x)