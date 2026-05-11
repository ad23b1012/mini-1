"""
ConvNeXt-Tiny — High-Accuracy Modern CNN Backbone for FER.

Adapted from: Liu et al. (2022) — "A ConvNet for the 2020s" — CVPR 2022.

ConvNeXt-Tiny achieves 82.5% top-1 on ImageNet while remaining a pure CNN,
making it fully compatible with Grad-CAM/Grad-ECLIP attention visualization.
Significantly outperforms ResNet-18 (69.8%) and EfficientNet-B4 (74.6%).

Parameters: ~28.6M | FLOPs: ~4.5G | Input: 224x224
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ConvNeXtTinyFER(nn.Module):
    """
    ConvNeXt-Tiny for Facial Expression Recognition.

    Pretrained on ImageNet-1K with the official ConvNeXt recipe.
    Replaces the classifier head for 8-class AffectNet emotion recognition.

    Architecture (feature extraction):
    Stem (4×4 conv) → Stage1 (3 blocks) → Stage2 (3 blocks)
                     → Stage3 (9 blocks) → Stage4 (3 blocks)
                     → LayerNorm → AdaptiveAvgPool → Flatten → Head
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        """
        Args:
            num_classes: Number of emotion classes (8 for AffectNet).
            pretrained: Whether to initialize with ImageNet pretrained weights.
        """
        super().__init__()

        # Load pretrained ConvNeXt-Tiny backbone
        if pretrained:
            self.backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.convnext_tiny(weights=None)

        # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        # in_features = 768 for ConvNeXt-Tiny
        num_features = self.backbone.classifier[2].in_features  # 768

        # Replace classifier head with a stronger regularized head
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),               # [B, 768, 1, 1] → [B, 768]
            nn.LayerNorm(num_features, eps=1e-6),
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes),
        )

        # Initialize the new classification head
        nn.init.xavier_normal_(self.backbone.classifier[3].weight)
        nn.init.zeros_(self.backbone.classifier[3].bias)

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. Landmarks parameter accepted for interface consistency but ignored."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract feature embeddings before the classification head."""
        x = self.backbone.features(x)          # All 4 stages → [B, 768, 7, 7]
        x = self.backbone.avgpool(x)            # AdaptiveAvgPool2d → [B, 768, 1, 1]
        x = self.backbone.classifier[0](x)     # Flatten → [B, 768]
        x = self.backbone.classifier[1](x)     # LayerNorm
        return x

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization.
        
        ConvNeXt-Tiny Stage 4 is features[7] — the last convolutional stage
        before the global average pool, ideal for spatial attention maps.
        """
        return self.backbone.features[7]  # Last ConvNeXt block stage
