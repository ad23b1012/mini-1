"""
ResNet-18 — Lightweight Baseline for Facial Expression Recognition.

A simple but effective baseline using pretrained ImageNet weights.
Significantly smaller than ResNet-50 (11M vs 25M parameters),
making it ideal for comparative analysis.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet18FER(nn.Module):
    """
    ResNet-18 for Facial Expression Recognition.

    Pretrained on ImageNet, with a modified classification head
    for emotion recognition.

    Architecture:
    conv1 → bn → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → dropout → fc
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        """
        Args:
            num_classes: Number of emotion classes (8 for AffectNet).
            pretrained: Whether to initialize with ImageNet pretrained weights.
        """
        super().__init__()

        # Load pretrained ResNet-18 backbone
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 512

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes),
        )

        # Initialize the classification head
        nn.init.xavier_normal_(self.backbone.fc[1].weight)
        nn.init.zeros_(self.backbone.fc[1].bias)

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. Landmarks parameter accepted for interface consistency but ignored."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract feature embeddings before the classification head."""
        # Use all layers except the final fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        return self.backbone.layer4[-1]
