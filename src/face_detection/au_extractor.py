"""
Action Unit (AU) Feature Extraction from MediaPipe Landmarks.

Converts raw facial landmarks into FACS-based Action Unit features
with binary activation states and semantic text labels for XAI.

All distances are normalized by Inter-Ocular Distance (IOD) — the distance
between inner eye corners — which is more stable across age groups, face
shapes, and image resolutions than face width.

Reference: Facial Action Coding System (FACS) by Ekman & Friesen.
MediaPipe landmark indices: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AUFeature:
    """A single Action Unit feature measurement."""
    au_code: str  # e.g., "AU1"
    name: str  # e.g., "Inner Brow Raise"
    raw_distance: float  # Raw euclidean distance
    normalized_distance: float  # Normalized by inter-ocular distance (IOD)
    is_active: bool  # Whether this AU is activated (above threshold)
    semantic_label: str  # e.g., "inner eyebrows raised"


@dataclass
class AUExtractionResult:
    """Complete AU extraction result for a face."""
    features: dict  # AU code → AUFeature
    active_aus: list  # List of active AU codes
    description_text: list  # List of semantic descriptions for active AUs
    feature_vector: np.ndarray  # Numerical feature vector for model input


# ============================================================================
# MediaPipe Face Mesh Landmark Indices
# ============================================================================
# These are carefully selected landmark pairs that correspond to FACS AUs.
# See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Landmark groups for AU computation
LANDMARKS = {
    # Eyebrow landmarks
    "left_inner_brow": 63,
    "right_inner_brow": 293,
    "left_outer_brow": 46,
    "right_outer_brow": 276,
    "left_brow_mid": 66,
    "right_brow_mid": 296,

    # Eye landmarks
    "left_eye_upper": 159,
    "left_eye_lower": 145,
    "right_eye_upper": 386,
    "right_eye_lower": 374,

    # Nose landmarks
    "nose_tip": 1,
    "nose_bridge": 6,

    # Upper face reference points (for brow distance)
    "left_eye_inner_corner": 133,
    "right_eye_inner_corner": 362,
    "left_eye_outer_corner": 33,
    "right_eye_outer_corner": 263,
    "forehead_top": 10,

    # Mouth landmarks
    "upper_lip_top": 13,
    "lower_lip_bottom": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip_center": 0,
    "lower_lip_center": 17,

    # Cheek landmarks
    "left_cheek": 48,
    "right_cheek": 278,
    "left_cheek_low": 115,
    "right_cheek_low": 344,

    # Face width reference
    "face_left": 234,
    "face_right": 454,

    # Chin
    "chin": 152,

    # Jaw
    "jaw_left": 172,
    "jaw_right": 397,
}


class AUExtractor:
    """
    Extracts Facial Action Unit (AU) features from MediaPipe landmarks.

    Maps geometric distances between landmark pairs to FACS Action Units,
    producing both numerical features and semantic text descriptions
    for downstream XAI explanation.

    All distances are normalized by Inter-Ocular Distance (IOD) for
    stability across different face sizes, ages, and image resolutions.
    
    Thresholds were calibrated on a mixed AffectNet/FFHQ dataset containing
    diverse demographics (infants, adults, elderly), resolutions (367-2006px),
    and expression intensities.
    """

    # IOD-normalized thresholds calibrated from dataset observations
    # Format: (threshold_value, comparison_direction)
    # "greater" = active when ratio > threshold (e.g., brow raised = larger distance)
    # "less" = active when ratio < threshold (e.g., brow furrowed = smaller distance)
    DEFAULT_THRESHOLDS = {
        "au1_inner_brow_raise": 0.42,       # IOD ratio, active when > threshold
        "au2_outer_brow_raise": 0.40,       # IOD ratio, active when > threshold
        "au4_brow_furrow": 0.55,            # IOD ratio, active when < threshold (closer = furrowed)
        "au5_eye_wide": 0.16,              # IOD ratio, active when > threshold
        "au6_cheek_raise": 0.16,           # IOD ratio, active when < threshold (shorter = raised)
        "au12_lip_corner_pull": 1.10,      # IOD ratio, active when > threshold (wider = smile)
        "au14_dimpler_asymmetry": 0.04,    # IOD ratio, active when > threshold (asymmetry)
        "au15_lip_corner_depress": 0.05,   # IOD ratio (signed), active when > threshold
        "au17_chin_raise": 0.35,           # IOD ratio, active when < threshold
        "au24_lip_press": 0.04,            # IOD ratio, active when < threshold (lips pressed)
        "au25_lips_part": 0.08,            # IOD ratio, active when > threshold
        "au26_jaw_drop": 0.65,             # IOD ratio, active when > threshold
    }

    def __init__(self, thresholds: Optional[dict] = None):
        """
        Initialize AU extractor with activation thresholds.

        Args:
            thresholds: Dictionary of AU thresholds. If None, uses defaults
                        calibrated on AffectNet dataset.
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    @staticmethod
    def _distance(landmarks: np.ndarray, idx1: int, idx2: int) -> float:
        """Compute Euclidean distance between two landmarks (2D, x-y only)."""
        p1 = landmarks[idx1, :2]
        p2 = landmarks[idx2, :2]
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def _inter_ocular_distance(landmarks: np.ndarray) -> float:
        """
        Compute Inter-Ocular Distance (IOD) — distance between inner eye corners.
        
        IOD is more stable than face width across:
        - Different age groups (babies vs adults)
        - Different face shapes
        - Different image resolutions
        """
        left_inner = landmarks[LANDMARKS["left_eye_inner_corner"], :2]
        right_inner = landmarks[LANDMARKS["right_eye_inner_corner"], :2]
        iod = float(np.linalg.norm(left_inner - right_inner))
        return max(iod, 1e-6)  # Avoid division by zero

    @staticmethod
    def _face_width(landmarks: np.ndarray) -> float:
        """Compute face width for distance normalization (legacy, kept for reference)."""
        left = landmarks[LANDMARKS["face_left"], :2]
        right = landmarks[LANDMARKS["face_right"], :2]
        width = float(np.linalg.norm(left - right))
        return max(width, 1e-6)

    def extract(self, landmarks: np.ndarray) -> AUExtractionResult:
        """
        Extract all AU features from facial landmarks.

        Args:
            landmarks: Normalized landmarks array of shape (468, 3) or (478, 3)
                       from MediaPipe Face Mesh.

        Returns:
            AUExtractionResult with features, active AUs, descriptions, and feature vector.
        """
        iod = self._inter_ocular_distance(landmarks)
        features = {}

        # ---- AU1: Inner Brow Raise ----
        # Distance from inner eyebrow to inner eye corner
        # When brows are raised, this distance increases
        left_dist = self._distance(
            landmarks, LANDMARKS["left_inner_brow"], LANDMARKS["left_eye_inner_corner"]
        )
        right_dist = self._distance(
            landmarks, LANDMARKS["right_inner_brow"], LANDMARKS["right_eye_inner_corner"]
        )
        raw = (left_dist + right_dist) / 2
        norm = raw / iod
        features["AU1"] = AUFeature(
            au_code="AU1",
            name="Inner Brow Raise",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm > self.thresholds["au1_inner_brow_raise"],
            semantic_label="inner eyebrows raised",
        )

        # ---- AU2: Outer Brow Raise ----
        # Distance from outer eyebrow to outer eye corner
        left_dist = self._distance(
            landmarks, LANDMARKS["left_outer_brow"], LANDMARKS["left_eye_outer_corner"]
        )
        right_dist = self._distance(
            landmarks, LANDMARKS["right_outer_brow"], LANDMARKS["right_eye_outer_corner"]
        )
        raw = (left_dist + right_dist) / 2
        norm = raw / iod
        features["AU2"] = AUFeature(
            au_code="AU2",
            name="Outer Brow Raise",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm > self.thresholds["au2_outer_brow_raise"],
            semantic_label="outer eyebrows raised",
        )

        # ---- AU4: Brow Lowerer (Furrow) ----
        # Distance between inner brow points — smaller = furrowed
        raw = self._distance(
            landmarks, LANDMARKS["left_inner_brow"], LANDMARKS["right_inner_brow"]
        )
        norm = raw / iod
        features["AU4"] = AUFeature(
            au_code="AU4",
            name="Brow Lowerer",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm < self.thresholds["au4_brow_furrow"],
            semantic_label="eyebrows furrowed",
        )

        # ---- AU5: Upper Lid Raise ----
        # Eye aperture — distance between upper and lower eyelids
        left_dist = self._distance(
            landmarks, LANDMARKS["left_eye_upper"], LANDMARKS["left_eye_lower"]
        )
        right_dist = self._distance(
            landmarks, LANDMARKS["right_eye_upper"], LANDMARKS["right_eye_lower"]
        )
        raw = (left_dist + right_dist) / 2
        norm = raw / iod
        features["AU5"] = AUFeature(
            au_code="AU5",
            name="Upper Lid Raise",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm > self.thresholds["au5_eye_wide"],
            semantic_label="eyes wide open",
        )

        # ---- AU6: Cheek Raise ----
        # Distance from lower eyelid to cheek bone landmark
        # When cheeks push up (genuine smile), this distance DECREASES
        left_dist = self._distance(
            landmarks, LANDMARKS["left_eye_lower"], LANDMARKS["left_cheek"]
        )
        right_dist = self._distance(
            landmarks, LANDMARKS["right_eye_lower"], LANDMARKS["right_cheek"]
        )
        raw = (left_dist + right_dist) / 2
        norm = raw / iod
        features["AU6"] = AUFeature(
            au_code="AU6",
            name="Cheek Raise",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm < self.thresholds["au6_cheek_raise"],
            semantic_label="cheeks raised (squinting)",
        )

        # ---- AU12: Lip Corner Puller (Smile) ----
        # Mouth width relative to IOD — wider = smile
        raw = self._distance(
            landmarks, LANDMARKS["mouth_left"], LANDMARKS["mouth_right"]
        )
        norm = raw / iod
        features["AU12"] = AUFeature(
            au_code="AU12",
            name="Lip Corner Puller",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm > self.thresholds["au12_lip_corner_pull"],
            semantic_label="lip corners pulled up (smile)",
        )

        # ---- AU14: Dimpler (NEW — for contempt detection) ----
        # Measures ASYMMETRY between left and right lip corner heights
        # Contempt is characterized by a unilateral lip corner pull
        left_corner_y = landmarks[LANDMARKS["mouth_left"], 1]
        right_corner_y = landmarks[LANDMARKS["mouth_right"], 1]
        raw = abs(left_corner_y - right_corner_y)
        norm = raw / iod
        features["AU14"] = AUFeature(
            au_code="AU14",
            name="Dimpler",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm > self.thresholds["au14_dimpler_asymmetry"],
            semantic_label="asymmetric lip movement (one-sided)",
        )

        # ---- AU15: Lip Corner Depressor ----
        # Vertical position of mouth corners relative to upper lip center
        # In MediaPipe coords, Y increases downward
        # Positive value = corners are BELOW lip center = depressed/frown
        left_corner_y = landmarks[LANDMARKS["mouth_left"], 1]
        right_corner_y = landmarks[LANDMARKS["mouth_right"], 1]
        center_lip_y = landmarks[LANDMARKS["upper_lip_center"], 1]
        raw_signed = ((left_corner_y + right_corner_y) / 2) - center_lip_y
        norm_signed = raw_signed / iod
        features["AU15"] = AUFeature(
            au_code="AU15",
            name="Lip Corner Depressor",
            raw_distance=abs(raw_signed),
            normalized_distance=norm_signed,  # Signed: positive = depressed
            is_active=norm_signed > self.thresholds["au15_lip_corner_depress"],
            semantic_label="lip corners pulled down (frown)",
        )

        # ---- AU17: Chin Raiser ----
        # Distance from lower lip to chin — smaller = chin pushed up
        raw = self._distance(
            landmarks, LANDMARKS["lower_lip_center"], LANDMARKS["chin"]
        )
        norm = raw / iod
        features["AU17"] = AUFeature(
            au_code="AU17",
            name="Chin Raiser",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm < self.thresholds["au17_chin_raise"],
            semantic_label="chin raised / pushed up",
        )

        # ---- AU24: Lip Pressor (NEW — for contempt/anger) ----
        # Distance between upper and lower lips at center
        # When lips are pressed together, this distance is very small
        raw = self._distance(
            landmarks, LANDMARKS["upper_lip_top"], LANDMARKS["lower_lip_bottom"]
        )
        norm = raw / iod
        features["AU24"] = AUFeature(
            au_code="AU24",
            name="Lip Pressor",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm < self.thresholds["au24_lip_press"],
            semantic_label="lips pressed together",
        )

        # ---- AU25: Lips Part ----
        # Distance between upper and lower lip — larger = parted
        raw = self._distance(
            landmarks, LANDMARKS["upper_lip_top"], LANDMARKS["lower_lip_bottom"]
        )
        norm = raw / iod
        features["AU25"] = AUFeature(
            au_code="AU25",
            name="Lips Part",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm > self.thresholds["au25_lips_part"],
            semantic_label="lips parted",
        )

        # ---- AU26: Jaw Drop ----
        # Distance from upper lip to chin — larger = jaw dropped
        raw = self._distance(
            landmarks, LANDMARKS["upper_lip_top"], LANDMARKS["chin"]
        )
        norm = raw / iod
        features["AU26"] = AUFeature(
            au_code="AU26",
            name="Jaw Drop",
            raw_distance=raw,
            normalized_distance=norm,
            is_active=norm > self.thresholds["au26_jaw_drop"],
            semantic_label="mouth open (jaw dropped)",
        )

        # Compile results
        active_aus = [code for code, feat in features.items() if feat.is_active]
        descriptions = [feat.semantic_label for feat in features.values() if feat.is_active]

        # Create numerical feature vector (normalized distances)
        feature_vector = np.array(
            [feat.normalized_distance for feat in features.values()],
            dtype=np.float32,
        )

        return AUExtractionResult(
            features=features,
            active_aus=active_aus,
            description_text=descriptions,
            feature_vector=feature_vector,
        )

    def format_for_prompt(self, result: AUExtractionResult) -> str:
        """
        Format AU extraction results as a bulleted list for the VLM prompt.

        Args:
            result: AUExtractionResult from self.extract().

        Returns:
            Formatted string for inclusion in the explanation prompt.
        """
        if not result.active_aus:
            return "- No strong facial action units detected (neutral expression)"

        lines = []
        for au_code in result.active_aus:
            feat = result.features[au_code]
            lines.append(
                f"- {feat.au_code} ({feat.name}): {feat.semantic_label} "
                f"(strength: {feat.normalized_distance:.4f})"
            )
        return "\n".join(lines)
