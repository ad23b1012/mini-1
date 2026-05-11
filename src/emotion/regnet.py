"""
RegNetY-800MF — Fast & Accurate CNN Backbone for FER.

Adapted from: Radosavovic et al. (2020) — "Designing Network Design Spaces"
— CVPR 2020.

RegNetY-800MF includes Squeeze-and-Excitation (SE) channel attention,
giving it better feature selectivity than ResNet-18 at a similar parameter
count (~6.3M vs 11.7M). Trains in ~3-4h on AffectNet, targeting 72-74% val.

Parameters: ~6.3M | FLOPs: ~0.83G | Input: 224x224
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class RegNetY800MF_FER(nn.Module):
    """
    RegNetY-800MF for Facial Expression Recognition.

    Pretrained on ImageNet-1K. RegNetY variants include Squeeze-and-Excitation
    blocks for automatic channel recalibration, which helps focus on the most
    discriminative facial features for each emotion.

    Architecture:
    Stem → Stage1 → Stage2 (SE) → Stage3 (SE) → Stage4 (SE) → AvgPool → Head
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        """
        Args:
            num_classes: Number of emotion classes (8 for AffectNet).
            pretrained: Whether to initialize with ImageNet pretrained weights.
        """
        super().__init__()

        # Load pretrained RegNetY-800MF backbone
        if pretrained:
            self.backbone = models.regnet_y_800mf(
                weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
            )
        else:
            self.backbone = models.regnet_y_800mf(weights=None)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 784 for RegNetY-800MF

        # Replace classification head with dropout regularization
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
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
        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)
        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)
        return x

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization.
        
        RegNetY-800MF trunk_output block 3 is the last convolutional stage,
        ideal for spatial attention maps.
        """
        return self.backbone.trunk_output.block4  # Last stage of trunk
