"""
EfficientNet-B4 — Parameter-Efficient Strong Baseline for FER.

Adapted from: Tan & Le (2019) — "EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks" — ICML 2019.

EfficientNet-B4 offers a strong accuracy-efficiency trade-off with 1792
embedding dimensions and compound scaling.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class EfficientNetB4FER(nn.Module):
    """
    EfficientNet-B4 for Facial Expression Recognition.

    Pretrained on ImageNet, with a modified classifier head
    for emotion recognition. Uses compound scaling for
    balanced depth/width/resolution.

    Parameters: ~19M (vs ResNet-50's ~25M)
    Input: 224x224 (can handle up to 380x380 natively)
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        """
        Args:
            num_classes: Number of emotion classes (8 for AffectNet).
            pretrained: Whether to initialize with ImageNet pretrained weights.
        """
        super().__init__()

        # Load pretrained EfficientNet-B4
        if pretrained:
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b4(weights=None)

        # Get the number of features from the classifier
        num_features = self.backbone.classifier[1].in_features  # 1792

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, num_classes),
        )

        # Initialize the classification head
        nn.init.xavier_normal_(self.backbone.classifier[1].weight)
        nn.init.zeros_(self.backbone.classifier[1].bias)

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. Landmarks parameter accepted for interface consistency but ignored."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract feature embeddings before the classification head."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        # Last convolutional block in the features
        return self.backbone.features[-1]
