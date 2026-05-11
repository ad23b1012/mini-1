"""
Attention Region Parser — Maps heatmap to semantic facial regions.

Converts a raw attention heatmap into human-readable descriptions of
which facial regions the model is focusing on. This drives the
semantic grounding of our textual explanations.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttentionRegion:
    """A semantic facial region with attention intensity."""
    name: str
    intensity: float  # 0-1, average attention in this region
    centroid: Tuple[float, float]  # (x, y) normalized
    description: str  # Human-readable description


# =============================================================================
# Facial region definitions (based on MediaPipe Face Mesh landmark clusters)
# =============================================================================

# Each region is defined by a bounding box in normalized coordinates (0-1)
# and the associated landmark indices for more precise boundary detection
FACIAL_REGIONS = {
    "forehead": {
        "y_range": (0.0, 0.25),
        "x_range": (0.2, 0.8),
        "description": "forehead area",
    },
    "left_eyebrow": {
        "y_range": (0.2, 0.35),
        "x_range": (0.55, 0.85),
        "description": "left eyebrow region",
    },
    "right_eyebrow": {
        "y_range": (0.2, 0.35),
        "x_range": (0.15, 0.45),
        "description": "right eyebrow region",
    },
    "left_eye": {
        "y_range": (0.30, 0.45),
        "x_range": (0.55, 0.80),
        "description": "left eye region",
    },
    "right_eye": {
        "y_range": (0.30, 0.45),
        "x_range": (0.20, 0.45),
        "description": "right eye region",
    },
    "nose": {
        "y_range": (0.35, 0.60),
        "x_range": (0.35, 0.65),
        "description": "nose area",
    },
    "left_cheek": {
        "y_range": (0.45, 0.70),
        "x_range": (0.65, 0.90),
        "description": "left cheek area",
    },
    "right_cheek": {
        "y_range": (0.45, 0.70),
        "x_range": (0.10, 0.35),
        "description": "right cheek area",
    },
    "mouth": {
        "y_range": (0.60, 0.80),
        "x_range": (0.30, 0.70),
        "description": "mouth and lip region",
    },
    "chin": {
        "y_range": (0.80, 1.0),
        "x_range": (0.30, 0.70),
        "description": "chin area",
    },
}


class RegionParser:
    """
    Parses attention heatmaps into semantic facial region descriptions.

    Algorithm:
    1. Normalize heatmap to [0, 1]
    2. For each predefined facial region, compute average attention intensity
    3. Rank regions by intensity
    4. Generate semantic descriptions for high-attention regions
    """

    def __init__(
        self,
        threshold: float = 0.3,
        top_k: int = 3,
    ):
        """
        Args:
            threshold: Minimum attention intensity to consider a region "active".
            top_k: Maximum number of regions to report.
        """
        self.threshold = threshold
        self.top_k = top_k

    def parse(self, heatmap: np.ndarray) -> List[AttentionRegion]:
        """
        Parse a heatmap into semantic facial regions.

        Args:
            heatmap: Attention heatmap as numpy array, any shape.
                     Will be resized to a standard grid for region mapping.

        Returns:
            List of AttentionRegion sorted by intensity (highest first).
        """
        # Normalize heatmap to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize to standard grid for consistent region mapping
        h, w = heatmap.shape[:2]
        if h < 10 or w < 10:
            heatmap = cv2.resize(heatmap, (56, 56), interpolation=cv2.INTER_LINEAR)
            h, w = 56, 56

        regions = []
        for name, region_def in FACIAL_REGIONS.items():
            y_start = int(region_def["y_range"][0] * h)
            y_end = int(region_def["y_range"][1] * h)
            x_start = int(region_def["x_range"][0] * w)
            x_end = int(region_def["x_range"][1] * w)

            # Extract region from heatmap
            region_mask = heatmap[y_start:y_end, x_start:x_end]

            if region_mask.size == 0:
                continue

            intensity = float(region_mask.mean())
            centroid_y = (region_def["y_range"][0] + region_def["y_range"][1]) / 2
            centroid_x = (region_def["x_range"][0] + region_def["x_range"][1]) / 2

            regions.append(AttentionRegion(
                name=name,
                intensity=intensity,
                centroid=(centroid_x, centroid_y),
                description=region_def["description"],
            ))

        # Sort by intensity (highest first)
        regions.sort(key=lambda r: r.intensity, reverse=True)

        # Filter by threshold and top_k
        active_regions = [r for r in regions if r.intensity >= self.threshold][:self.top_k]

        return active_regions if active_regions else regions[:1]  # At least return top region

    def format_for_prompt(self, regions: List[AttentionRegion]) -> str:
        """
        Format parsed regions as text for the VLM prompt.

        Args:
            regions: List of AttentionRegion from self.parse().

        Returns:
            Formatted string describing where the model is focusing.
        """
        if not regions:
            return "Model attention is diffuse across the entire face."

        lines = []
        for region in regions:
            intensity_label = self._intensity_label(region.intensity)
            lines.append(
                f"- {intensity_label} attention on {region.description} "
                f"(intensity: {region.intensity:.2f})"
            )

        return "\n".join(lines)

    @staticmethod
    def _intensity_label(intensity: float) -> str:
        """Convert intensity value to a descriptive label."""
        if intensity >= 0.7:
            return "Strong"
        elif intensity >= 0.4:
            return "Moderate"
        elif intensity >= 0.2:
            return "Mild"
        else:
            return "Weak"

    def get_summary(self, regions: List[AttentionRegion]) -> str:
        """
        Generate a one-sentence summary of where the model is looking.

        Args:
            regions: List of AttentionRegion from self.parse().

        Returns:
            A concise summary string.
        """
        if not regions:
            return "The model attention is evenly distributed across the face."

        region_names = [r.description for r in regions[:3]]
        if len(region_names) == 1:
            return f"The model primarily focused on the {region_names[0]}."
        elif len(region_names) == 2:
            return f"The model focused on the {region_names[0]} and {region_names[1]}."
        else:
            return (
                f"The model focused on the {region_names[0]}, {region_names[1]}, "
                f"and {region_names[2]}."
            )
