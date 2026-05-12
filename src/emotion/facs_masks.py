"""
FACS Region Mask Builder — Converts emotion labels to spatial attention priors.

Uses Facial Action Coding System (FACS) knowledge to define which facial regions
should be most active for each emotion class. These masks serve as supervision
signals during XAI-guided training (Phase 2).

Reference: Ekman & Friesen (1978) — Facial Action Coding System.

The masks are built on a 7×7 spatial grid corresponding to the output
of ConvNeXt-Tiny's Stage 4 (features[7]) before global average pooling.
"""

import torch


# =============================================================================
# Facial region definitions on a 7×7 spatial grid
# =============================================================================
# Each region is (y_start, y_end, x_start, x_end) — row-major on 7×7
# These approximate the spatial layout of a centred, cropped face image.

REGION_COORDS_7x7 = {
    "brow":       (0, 2, 1, 6),   # Top rows, central columns — eyebrow region
    "left_eye":   (2, 4, 4, 6),   # Mid rows, right side (mirrored in image)
    "right_eye":  (2, 4, 1, 3),   # Mid rows, left side
    "nose":       (2, 5, 2, 5),   # Centre block
    "cheeks":     (3, 5, 0, 7),   # Wide mid-lower band
    "mouth":      (4, 6, 2, 5),   # Lower centre
    "chin":       (6, 7, 2, 5),   # Bottom centre
}


# =============================================================================
# Emotion → FACS Action Units → Facial Regions
# =============================================================================
# AffectNet-8 class indices:
#   0=anger, 1=contempt, 2=disgust, 3=fear,
#   4=happy, 5=neutral, 6=sad, 7=surprise
#
# Mapping based on Ekman's FACS literature:
#   Anger:    AU4 (brow furrow) + AU23/24 (lip tighten/press)
#   Contempt: AU12R/L (unilateral lip pull) + AU14 (dimpler)
#   Disgust:  AU9 (nose wrinkle) + AU10 (upper lip raise)
#   Fear:     AU1+AU2 (brow raise) + AU4 (brow furrow) + AU5 (eye widen)
#   Happy:    AU6 (cheek raise) + AU12 (lip corner pull)
#   Neutral:  No strong AUs → no region constraint
#   Sad:      AU1 (inner brow raise) + AU15 (lip corner depress) + AU17 (chin raise)
#   Surprise: AU1+AU2 (brow raise) + AU5 (eye widen) + AU26 (jaw drop)

EMOTION_REGIONS = {
    0: ["brow", "mouth"],                               # anger
    1: ["mouth"],                                        # contempt
    2: ["nose", "mouth"],                                # disgust
    3: ["brow", "left_eye", "right_eye"],                # fear
    4: ["cheeks", "mouth"],                              # happy
    5: [],                                               # neutral (no constraint)
    6: ["brow", "mouth", "chin"],                        # sad
    7: ["brow", "left_eye", "right_eye", "mouth"],       # surprise
}


def build_facs_masks(
    labels: torch.Tensor,
    grid_size: int = 7,
) -> torch.Tensor:
    """
    Build FACS-derived spatial attention masks for a batch of labels.

    For each emotion label, activates the facial regions that FACS theory
    says should be most important. Neutral faces get a uniform mask
    (no region is penalised).

    Args:
        labels:    [B] tensor of emotion class indices (0–7).
        grid_size: Spatial size of the feature map. Default 7 for ConvNeXt-Tiny.

    Returns:
        masks: [B, H, W] tensor, each mask normalised to sum to 1.
    """
    B = labels.size(0)
    masks = torch.zeros(B, grid_size, grid_size, device=labels.device)

    for i in range(B):
        label_idx = labels[i].item()
        regions = EMOTION_REGIONS.get(label_idx, [])

        if not regions:
            # Neutral — uniform mask means no penalty anywhere
            masks[i] = 1.0 / (grid_size * grid_size)
            continue

        for region_name in regions:
            y1, y2, x1, x2 = REGION_COORDS_7x7[region_name]
            masks[i, y1:y2, x1:x2] = 1.0

        # Normalise so each mask sums to 1 (for fair MSE comparison)
        total = masks[i].sum()
        if total > 0:
            masks[i] = masks[i] / total

    return masks
