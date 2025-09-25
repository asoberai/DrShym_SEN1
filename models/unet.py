"""
UNet segmentation model with ResNet encoder
"""

import torch
import torch.nn as nn
from .encoder_backbones import ResNet18Encoder, ResNet50Encoder, ConvBlock, UpBlock


class UNet(nn.Module):
    """UNet with ResNet18 encoder for flood segmentation"""

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 encoder: str = "resnet18",
                 pretrained: bool = True):
        super(UNet, self).__init__()

        if encoder == "resnet18":
            self.encoder = ResNet18Encoder(pretrained=pretrained, in_channels=in_channels)
        elif encoder == "resnet50":
            self.encoder = ResNet50Encoder(pretrained=pretrained, in_channels=in_channels)
        else:
            raise ValueError(f"Encoder {encoder} not supported. Choose from: resnet18, resnet50")

        # Get encoder channels
        encoder_channels = self.encoder.channels  # ResNet18: [64, 64, 128, 256, 512] or ResNet50: [64, 256, 512, 1024, 2048]

        # Decoder path matching exact checkpoint architecture
        if encoder == "resnet50":
            # From checkpoint analysis:
            # up1: 2048->1024 upsample, concat(1024+1024)=2048, conv->512
            # up2: 512->256 upsample, concat(256+512)=768, conv->256
            # up3: 256->128 upsample, concat(128+256)=384, conv->128
            # up4: 128->64 upsample, concat(64+64)=128, conv->64
            self.up1 = UpBlock(in_channels=2048, upsample_out_channels=1024, out_channels=512, concat_channels=2048)
            self.up2 = UpBlock(in_channels=512, upsample_out_channels=256, out_channels=256, concat_channels=768)
            self.up3 = UpBlock(in_channels=256, upsample_out_channels=128, out_channels=128, concat_channels=384)
            self.up4 = UpBlock(in_channels=128, upsample_out_channels=64, out_channels=64, concat_channels=128)
        else:
            # ResNet18 channels: [64, 64, 128, 256, 512]
            self.up1 = UpBlock(in_channels=encoder_channels[4], upsample_out_channels=encoder_channels[3], out_channels=256, concat_channels=encoder_channels[4] + encoder_channels[3])
            self.up2 = UpBlock(in_channels=256, upsample_out_channels=encoder_channels[2], out_channels=128, concat_channels=256 + encoder_channels[2])
            self.up3 = UpBlock(in_channels=128, upsample_out_channels=encoder_channels[1], out_channels=64, concat_channels=128 + encoder_channels[1])
            self.up4 = UpBlock(in_channels=64, upsample_out_channels=encoder_channels[0], out_channels=64, concat_channels=64 + encoder_channels[0])

        # Final classifier
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        # Get input size for final upsampling
        input_size = x.shape[2:]

        # Encoder
        features = self.encoder(x)

        # Decoder with skip connections
        x = self.up1(features[4], features[3])  # 1/16 scale
        x = self.up2(x, features[2])            # 1/8 scale
        x = self.up3(x, features[1])            # 1/4 scale
        x = self.up4(x, features[0])            # 1/2 scale

        # Final upsampling to original size
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # Final classifier
        x = self.final_conv(x)

        return x

    def get_model_info(self):
        """Get model architecture information"""
        return {
            "architecture": "UNet + ResNet18",
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }