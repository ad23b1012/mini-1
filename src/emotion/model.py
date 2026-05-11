"""
POSTER V2 (POSTER++) — Primary Emotion Classifier.

Adapted from: Zheng et al. (2023) — "POSTER V2: A simpler and stronger facial
expression recognition network" — ICCV 2023.

Architecture:
- Two-stream design:
  1. Image backbone (IR-50) → visual feature maps
  2. Landmark-guided cross-attention → focuses on facial regions
- Option to fuse real geometric coordinates (from MediaPipe) via an internal
  dense projection layer, avoiding the simplistic 'fake token' limitation.

Achieves 92.21% on RAF-DB and 67.49% on AffectNet-8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# =============================================================================
# IR-50 Backbone
# =============================================================================

class IRBlock(nn.Module):
    """Improved Residual Block for face recognition."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class IR50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        self.layer1 = self._make_layer(64, 64, 3, stride=2)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 14, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [IRBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(IRBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_s3 = self.layer3(x)    
        feat_s4 = self.layer4(feat_s3)  
        return feat_s3, feat_s4, feat_s4


# =============================================================================
# Window-based Cross-Attention (from POSTER V2)
# =============================================================================

class WindowCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# =============================================================================
# POSTER V2 Full Model
# =============================================================================

class POSTERV2(nn.Module):
    def __init__(self, num_classes: int = 8, num_landmark_tokens: int = 49,
                 embed_dim: int = 512, num_heads: int = 8, depth: int = 2, dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_landmark_tokens = num_landmark_tokens

        self.backbone = IR50Backbone()

        self.proj_s3 = nn.Sequential(nn.Conv2d(256, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim))
        self.proj_s4 = nn.Sequential(nn.Conv2d(512, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim))

        # Fallback landmark queries if structural MediaPipe data is missing
        self.fallback_landmark_tokens = nn.Parameter(
            torch.randn(1, num_landmark_tokens, embed_dim) * 0.02
        )
        
        # Projection layer extracting 49 structural query vectors dynamically
        # directly from MediaPipe's geometric (468, 2) facial map output
        self.landmark_proj = nn.Sequential(
            nn.Linear(468 * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_landmark_tokens * embed_dim)
        )

        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": WindowCrossAttention(dim=embed_dim, num_heads=num_heads,
                                                   attn_drop=dropout, proj_drop=dropout),
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
                ),
            })
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(embed_dim, num_classes))

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_landmark_queries(self, B: int, landmarks: Optional[torch.Tensor]) -> torch.Tensor:
        """Helper to generate active query tokens based on real landmarks or fallback to parameters."""
        if landmarks is not None and landmarks.sum() != 0:
            landmarks_flat = landmarks.flatten(start_dim=1)
            landmark_queries_flat = self.landmark_proj(landmarks_flat)
            landmark_queries = landmark_queries_flat.view(B, self.num_landmark_tokens, self.embed_dim)
            return landmark_queries
        return self.fallback_landmark_tokens.expand(B, -1, -1)

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]

        feat_s3, feat_s4, _ = self.backbone(x)
        feat_s3 = self.proj_s3(feat_s3).flatten(2).transpose(1, 2)
        feat_s4 = self.proj_s4(feat_s4).flatten(2).transpose(1, 2)
        image_tokens = torch.cat([feat_s3, feat_s4], dim=1)

        landmark_queries = self._get_landmark_queries(B, landmarks)
        
        for layer in self.cross_attention_layers:
            attended = layer["cross_attn"](layer["norm1"](landmark_queries), layer["norm2"](image_tokens))
            landmark_queries = landmark_queries + attended
            landmark_queries = landmark_queries + layer["ffn"](landmark_queries)

        features = self.norm(landmark_queries).mean(dim=1)
        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        feat_s3, feat_s4, _ = self.backbone(x)
        feat_s3 = self.proj_s3(feat_s3).flatten(2).transpose(1, 2)
        feat_s4 = self.proj_s4(feat_s4).flatten(2).transpose(1, 2)
        image_tokens = torch.cat([feat_s3, feat_s4], dim=1)

        landmark_queries = self._get_landmark_queries(B, landmarks)

        for layer in self.cross_attention_layers:
            attended = layer["cross_attn"](layer["norm1"](landmark_queries), layer["norm2"](image_tokens))
            landmark_queries = landmark_queries + attended
            landmark_queries = landmark_queries + layer["ffn"](landmark_queries)

        return self.norm(landmark_queries).mean(dim=1)

    def get_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


def build_model(model_name: str = "poster_v2", num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    if model_name == "poster_v2":
        model = POSTERV2(num_classes=num_classes)
    elif model_name == "resnet50_cbam":
        from src.emotion.baseline_resnet_cbam import ResNet50CBAM
        model = ResNet50CBAM(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet18":
        from src.emotion.resnet18 import ResNet18FER
        model = ResNet18FER(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "efficientnet_b4":
        from src.emotion.efficientnet_b4 import EfficientNetB4FER
        model = EfficientNetB4FER(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "convnext_tiny":
        from src.emotion.convnext import ConvNeXtTinyFER
        model = ConvNeXtTinyFER(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "regnet_y_800mf":
        from src.emotion.regnet import RegNetY800MF_FER
        model = RegNetY800MF_FER(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Use 'poster_v2', 'resnet50_cbam', 'resnet18', 'efficientnet_b4', "
            f"'convnext_tiny', or 'regnet_y_800mf'."
        )
    return model
