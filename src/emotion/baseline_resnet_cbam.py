"""
ResNet-50 + CBAM Baseline Model for Facial Expression Recognition.

Reproduced from: Aly et al. (2023) — "Enhancing Facial Expression Recognition
System in Online Learning Context Using Efficient Deep Learning Model"
IEEE Access, Oct 2023.

Key modifications over vanilla ResNet-50:
1. CBAM (Convolutional Block Attention Module) after each residual block
2. Modified residual downsampling: avg_pool(2x2, stride=2) → conv(1x1, stride=1)
   instead of strided convolution — preserves 75% more spatial information.

This serves as our ablation baseline to compare against POSTER++.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


# =============================================================================
# CBAM: Convolutional Block Attention Module
# Reference: Woo et al. (2018) — "CBAM: Convolutional Block Attention Module"
# =============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.

    Aggregates spatial information using both average-pooling and max-pooling,
    then computes channel-wise importance weights via a shared MLP.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Average pooling → (B, C)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        avg_out = self.shared_mlp(avg_pool)

        # Max pooling → (B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        max_out = self.shared_mlp(max_pool)

        # Combine and sigmoid
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.

    Computes spatial importance weights by applying a convolution
    on concatenated channel-wise average and max features.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise average and max → (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module.

    Sequentially applies Channel Attention → Spatial Attention
    to refine feature maps.
    """

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =============================================================================
# Modified ResNet-50 with CBAM
# =============================================================================

class CBAMBottleneck(nn.Module):
    """
    ResNet Bottleneck block with CBAM attention.

    Modified from Aly et al. (2023):
    - CBAM module added after the last conv in each bottleneck
    - Modified downsampling: avg_pool(2×2, stride=2) before 1×1 conv (stride=1)
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cbam_reduction: int = 16,
    ):
        super().__init__()

        # 1×1 conv (reduce channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3×3 conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1×1 conv (expand channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # CBAM attention after final conv
        self.cbam = CBAM(out_channels * self.expansion, reduction=cbam_reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Apply CBAM attention
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50CBAM(nn.Module):
    """
    ResNet-50 + CBAM for Facial Expression Recognition.

    Based on Aly et al. (2023) with modified residual downsampling:
    - Instead of strided 1×1 conv for downsampling, uses:
      avg_pool(2×2, stride=2) → conv(1×1, stride=1)
    - This preserves 75% more spatial information.
    - CBAM attention module after each bottleneck block.

    Architecture:
    conv1 → bn → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → fc
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        """
        Args:
            num_classes: Number of emotion classes (7 for FER2013/RAF-DB).
            pretrained: Whether to initialize with ImageNet pretrained weights.
        """
        super().__init__()

        # Load pretrained ResNet-50 backbone
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        # Copy stem layers (conv1, bn1, relu, maxpool)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Build CBAM-enhanced layers
        self.layer1 = self._make_cbam_layer(resnet.layer1, 256)
        self.layer2 = self._make_cbam_layer(resnet.layer2, 512)
        self.layer3 = self._make_cbam_layer(resnet.layer3, 1024)
        self.layer4 = self._make_cbam_layer(resnet.layer4, 2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes)

        # Initialize the classification head
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _make_cbam_layer(self, original_layer: nn.Module, channels: int) -> nn.Module:
        """
        Add CBAM attention after each bottleneck in a ResNet layer.

        We keep the original pretrained weights and just add CBAM modules.
        """
        modules = []
        for block in original_layer:
            modules.append(block)
            modules.append(CBAM(channels, reduction=16))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before the classification head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        # Last CBAM block in layer4
        return self.layer4[-1]
