"""
Grad-ECLIP: Gradient-based Visual Explanation for Emotion Classification.

Adapted from: Zhao et al. (2024) — "Gradient-based Visual Explanation for CLIP"
ICML 2024. Official repo: https://github.com/Cyang-Zhao/Grad-Eclip

Key insight: Standard Grad-CAM produces sparse/noisy heatmaps on Vision Transformers.
Grad-ECLIP decomposes the encoder architecture to discover relationships between the
output score and intermediate spatial features, producing higher-quality attention maps
via channel + spatial importance weights.

We adapt this for our emotion classifier (POSTER V2 / ResNet-50+CBAM).
Falls back to pytorch-grad-cam's EigenGradCAM for CNN backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union
from PIL import Image
import cv2


class GradECLIP:
    """
    Grad-ECLIP attention map generator.

    Generates high-quality attention heatmaps by computing gradients
    of the classification score with respect to intermediate features,
    then applying channel and spatial importance weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: The emotion classifier model.
            target_layer: The layer to extract attention from.
                          If None, uses model.get_target_layer().
            device: Device for computation.
        """
        self.model = model
        self.device = device

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer"):
            self.target_layer = model.get_target_layer()
        else:
            raise ValueError(
                "Must provide target_layer or model must have get_target_layer() method."
            )

        # Storage for hooks
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-ECLIP attention map.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W).
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Attention heatmap as numpy array (H_feat, W_feat) normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. "
                               "Check target_layer compatibility.")

        if hasattr(self.model, "reshape_transform"):
            activations = self.model.reshape_transform(activations)
            gradients = self.model.reshape_transform(gradients)

        # ===== Grad-ECLIP: Channel + Spatial Importance Weighting =====

        # Channel importance: global average of gradients per channel
        # This tells us WHICH feature channels are important for this emotion
        channel_weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Spatial importance: per-location gradient magnitude
        # This tells us WHERE the model is "looking"
        spatial_weights = gradients.abs().mean(dim=1, keepdim=True)  # (1, 1, H, W)

        # Combined attention: channel-weighted activations * spatial importance
        cam = (channel_weights * activations * spatial_weights).sum(dim=1, keepdim=True)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze()  # (H, W)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def generate_for_image(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> tuple:
        """
        Generate attention heatmap overlaid on the original image.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W).
            original_image: Original image as numpy array (H, W, 3) in RGB.
            target_class: Target class. If None, uses predicted class.
            colormap: OpenCV colormap for heatmap visualization.
            alpha: Heatmap overlay transparency.

        Returns:
            Tuple of (heatmap_overlay, raw_cam):
            - heatmap_overlay: Image with heatmap overlay (H, W, 3) in RGB
            - raw_cam: Raw attention map (H_feat, W_feat) normalized to [0, 1]
        """
        raw_cam = self.generate(input_tensor, target_class)

        # Resize heatmap to original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(raw_cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert to heatmap
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)

        return overlay, raw_cam


class StandardGradCAM:
    """
    Standard Grad-CAM fallback for CNN-based models.

    Uses pytorch-grad-cam library for more robust CAM generation.
    Used as comparison baseline in ablation studies.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer"):
            self.target_layer = model.get_target_layer()
        else:
            raise ValueError("Must provide target_layer or model must have get_target_layer()")

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate standard Grad-CAM attention map."""
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            reshape_transform = getattr(self.model, "reshape_transform", None)
            cam = GradCAM(
                model=self.model, 
                target_layers=[self.target_layer], 
                reshape_transform=reshape_transform
            )

            targets = None
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]

            grayscale_cam = cam(input_tensor=input_tensor.to(self.device), targets=targets)
            return grayscale_cam[0]

        except ImportError:
            print("[Warning] pytorch-grad-cam not installed, falling back to manual Grad-CAM")
            return self._manual_grad_cam(input_tensor, target_class)

    def _manual_grad_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Manual Grad-CAM implementation as fallback."""
        activations = None
        gradients = None

        def fwd_hook(m, i, o):
            nonlocal activations
            activations = o.detach()

        def bwd_hook(m, gi, go):
            nonlocal gradients
            gradients = go[0].detach()

        fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        fwd_handle.remove()
        bwd_handle.remove()

        if hasattr(self.model, "reshape_transform"):
            activations = self.model.reshape_transform(activations)
            gradients = self.model.reshape_transform(gradients)

        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


def build_attention_generator(
    model: nn.Module,
    method: str = "grad_cam",
    target_layer: Optional[nn.Module] = None,
    device: str = "cuda",
):
    """
    Factory function to build an attention map generator.

    Args:
        model: The emotion classifier model.
        method: "grad_eclip" or "grad_cam".
        target_layer: Target layer for attention extraction.
        device: Device for computation.

    Returns:
        An attention generator instance.
    """
    if method == "grad_eclip":
        return GradECLIP(model, target_layer, device)
    elif method in ("grad_cam", "gradcam"):
        return StandardGradCAM(model, target_layer, device)
    else:
        raise ValueError(f"Unknown attention method: {method}")
