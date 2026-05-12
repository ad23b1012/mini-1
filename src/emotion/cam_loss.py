"""
CAM-based Attention Loss for XAI-Guided Training (Phase 2).

Computes Class Activation Maps (CAM) directly from the final FC layer weights
during the forward pass — no second backward pass required, adding only ~5%
training overhead compared to Grad-CAM's ~100% overhead.

The CAM is compared against FACS-derived spatial masks using MSE loss,
penalising the model when it attends to incorrect facial regions even if
the classification prediction is correct.

Reference:
  Zhou et al. (2016) — "Learning Deep Features for Discriminative Localization"
  Ross et al. (2017) — "Right for the Right Reasons" (inspiration for loss design)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAMAttentionLoss(nn.Module):
    """
    FACS-Constrained CAM Attention Loss.

    Computes CAM from the FC layer weights and feature maps,
    then penalises deviations from FACS-derived region masks.

    This loss is added to the classification loss during training:
        L_total = L_focal + λ × L_cam_attention
    """

    def __init__(self):
        super().__init__()

    def compute_cam(
        self,
        feature_maps: torch.Tensor,
        fc_weights: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Class Activation Maps from FC layer weights.

        This is a forward-pass-only operation — no gradients needed.

        Args:
            feature_maps: [B, C, H, W] — output of the last convolutional stage.
            fc_weights:   [num_classes, C] — weights of the final Linear layer.
            labels:       [B] — ground-truth class indices.

        Returns:
            cam: [B, H, W] — normalised CAM, values in [0, 1].
        """
        # Select FC weights corresponding to each sample's label
        batch_weights = fc_weights[labels]  # [B, C]

        # Weighted sum over channels: cam[b, h, w] = Σ_c w[b,c] × f[b,c,h,w]
        cam = torch.einsum("bc,bchw->bhw", batch_weights, feature_maps)

        # ReLU: only keep positive activations (regions that help the class)
        cam = F.relu(cam)

        # Normalise each sample's CAM to [0, 1]
        cam_max = cam.amax(dim=(1, 2), keepdim=True) + 1e-8
        cam = cam / cam_max

        return cam

    def forward(
        self,
        feature_maps: torch.Tensor,
        fc_weights: torch.Tensor,
        labels: torch.Tensor,
        facs_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the FACS-constrained CAM attention loss.

        Args:
            feature_maps: [B, C, H, W] from the last conv stage.
            fc_weights:   [num_classes, C] from the classification Linear layer.
            labels:       [B] ground-truth emotion labels.
            facs_masks:   [B, H, W] FACS-derived spatial prior masks.

        Returns:
            loss: Scalar MSE loss between the CAM and the FACS masks.
        """
        cam = self.compute_cam(feature_maps, fc_weights, labels)
        return F.mse_loss(cam, facs_masks)
