"""
Face Detection & Landmark Extraction.

Uses MediaPipe Tasks API (0.10.35):
  Stage 1: FaceDetector (BlazeFace SSD) → tight face bounding box
  Stage 2: FaceLandmarker → 468 landmark extraction on the cropped face

The key difference from using FaceLandmarker-only:
- FaceDetector → trained SSD model → TIGHT box around face (forehead to chin)
- FaceLandmarker → landmarks span jaw contour + hairline → LOOSE box
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from PIL import Image

import mediapipe as mp


@dataclass
class FaceDetectionResult:
    """Result of face detection on a single image."""
    face_found: bool
    landmarks: Optional[np.ndarray] = None       # Shape: (468, 3) — normalized (x, y, z)
    landmarks_pixel: Optional[np.ndarray] = None  # Shape: (468, 2) — pixel coordinates
    face_crop: Optional[np.ndarray] = None        # Cropped face region (BGR)
    face_crop_pil: Optional[Image.Image] = None   # Cropped face as PIL Image
    bbox: Optional[tuple] = None                  # (x1, y1, x2, y2) bounding box
    image_shape: Optional[tuple] = None           # (height, width) of original image


class FaceDetector:
    """
    Two-stage face processor using MediaPipe Tasks API:
      Stage 1: FaceDetector (BlazeFace SSD) → tight face bounding box
      Stage 2: FaceLandmarker → 468 landmarks on the cropped face
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        face_crop_padding: float = 0.10,
        detector_model_path: str = None,
        landmarker_model_path: str = None,
        **kwargs,
    ):
        """
        Args:
            min_detection_confidence: Minimum confidence for detection.
            face_crop_padding: Padding around face bbox (10% for face-only).
            detector_model_path: Path to BlazeFace .tflite model.
            landmarker_model_path: Path to FaceLandmarker .task model.
        """
        self.face_crop_padding = face_crop_padding
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Default model paths
        if detector_model_path is None:
            detector_model_path = os.path.join(base_dir, "models", "blaze_face_short_range.tflite")
        if landmarker_model_path is None:
            landmarker_model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        # Stage 1: Face Detector (BlazeFace SSD — gives tight bbox)
        self.face_detector = None
        if os.path.exists(detector_model_path):
            base_options = mp.tasks.BaseOptions(model_asset_path=detector_model_path)
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=min_detection_confidence,
            )
            self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            print(f"[FaceDetector] BlazeFace SSD loaded from {detector_model_path}")
        else:
            print(f"[FaceDetector] WARNING: Face detector model not found at {detector_model_path}")

        # Stage 2: Face Landmarker (468 landmarks)
        self.face_landmarker = None
        if os.path.exists(landmarker_model_path):
            base_options = mp.tasks.BaseOptions(model_asset_path=landmarker_model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            print(f"[FaceDetector] FaceLandmarker loaded from {landmarker_model_path}")
        else:
            print(f"[FaceDetector] WARNING: Landmarker model not found at {landmarker_model_path}")

    def _detect_face_bbox(self, image_rgb: np.ndarray) -> Optional[tuple]:
        """
        Use BlazeFace SSD to get a TIGHT face bounding box.
        
        Returns:
            (x1, y1, x2, y2) pixel coordinates or None.
        """
        if self.face_detector is None:
            return None

        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.face_detector.detect(mp_image)

        if not results.detections:
            return None

        # Take the detection with highest confidence
        best_det = max(results.detections, key=lambda d: d.categories[0].score)
        bbox = best_det.bounding_box

        # bbox has origin_x, origin_y, width, height in pixels
        x_min = bbox.origin_x
        y_min = bbox.origin_y
        box_w = bbox.width
        box_h = bbox.height

        # Apply padding
        pad_x = int(box_w * self.face_crop_padding)
        pad_y = int(box_h * self.face_crop_padding)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_min + box_w + pad_x)
        y2 = min(h, y_min + box_h + pad_y)

        # Enforce square crop
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w > crop_h:
            diff = crop_w - crop_h
            y1 = max(0, y1 - diff // 2)
            y2 = min(h, y1 + crop_w)
            if y2 - y1 < crop_w:
                y1 = max(0, y2 - crop_w)
        elif crop_h > crop_w:
            diff = crop_h - crop_w
            x1 = max(0, x1 - diff // 2)
            x2 = min(w, x1 + crop_h)
            if x2 - x1 < crop_h:
                x1 = max(0, x2 - crop_h)

        # Minimum size guard
        if (x2 - x1) < 64 or (y2 - y1) < 64:
            return None

        return (int(x1), int(y1), int(x2), int(y2))

    def _extract_landmarks(self, face_crop_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 MediaPipe Face Mesh landmarks from a face crop."""
        if self.face_landmarker is None:
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_rgb)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks],
            dtype=np.float32,
        )[:468]

        return landmarks

    def detect(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detect face and extract landmarks from an image.

        Args:
            image: Input image in BGR format (as read by cv2.imread).
        """
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Stage 1: Get tight face bounding box from BlazeFace SSD
        bbox = self._detect_face_bbox(image_rgb)

        if bbox is None:
            return FaceDetectionResult(face_found=False, image_shape=(h, w))

        x1, y1, x2, y2 = bbox

        # Crop face
        face_crop_bgr = image[y1:y2, x1:x2].copy()
        if face_crop_bgr.size == 0:
            return FaceDetectionResult(face_found=False, image_shape=(h, w))

        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        try:
            face_crop_pil = Image.fromarray(face_crop_rgb)
        except Exception:
            face_crop_pil = None

        # Stage 2: Extract landmarks from the cropped face
        landmarks = self._extract_landmarks(face_crop_rgb)

        if landmarks is None:
            return FaceDetectionResult(
                face_found=True,
                landmarks=None,
                landmarks_pixel=None,
                face_crop=face_crop_bgr,
                face_crop_pil=face_crop_pil,
                bbox=bbox,
                image_shape=(h, w),
            )

        # Convert normalized landmarks to pixel coordinates within crop
        crop_h, crop_w = face_crop_bgr.shape[:2]
        landmarks_pixel = np.array(
            [(lm[0] * crop_w, lm[1] * crop_h) for lm in landmarks],
            dtype=np.float32,
        )

        return FaceDetectionResult(
            face_found=True,
            landmarks=landmarks,
            landmarks_pixel=landmarks_pixel,
            face_crop=face_crop_bgr,
            face_crop_pil=face_crop_pil,
            bbox=bbox,
            image_shape=(h, w),
        )

    def detect_from_path(self, image_path: str) -> FaceDetectionResult:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(image)

    def detect_from_pil(self, pil_image: Image.Image) -> FaceDetectionResult:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self.detect(image)

    def get_face_width(self, landmarks: np.ndarray) -> float:
        left_cheek = landmarks[234, :2]
        right_cheek = landmarks[454, :2]
        return np.linalg.norm(left_cheek - right_cheek)

    def close(self):
        if self.face_detector is not None:
            self.face_detector.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
