"""
End-to-End XAI Emotion Recognition Pipeline.

Orchestrates the full flow from input image to final output:
1. Face detection + landmark extraction (MediaPipe)
2. AU feature extraction (geometric distances → FACS)
3. Emotion classification (POSTER V2 / ResNet-50+CBAM)
4. Attention map generation (Grad-ECLIP)
5. Attention region parsing (heatmap → semantic regions)
6. [VRAM swap: unload classifier, load VLM]
7. Explanation generation (LLaVA-7B 4-bit)
8. Visualization generation

All designed for sequential model loading to stay under 6GB VRAM.
"""

import os
import gc
import json
import time
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

from src.face_detection.detector import FaceDetector, FaceDetectionResult
from src.face_detection.au_extractor import AUExtractor, AUExtractionResult
from src.emotion.model import build_model
from src.attention.grad_eclip import build_attention_generator
from src.attention.region_parser import RegionParser, AttentionRegion
from src.explainer.prompt_builder import PromptBuilder
from src.explainer.vlm_engine import VLMEngine
from src.visualization import (
    draw_landmarks,
    draw_heatmap_overlay,
    create_combined_panel,
)


@dataclass
class PredictionResult:
    """Complete prediction result from the pipeline."""
    # Image info
    image_path: str = ""
    processing_time: float = 0.0

    # Face detection
    face_found: bool = False

    # Emotion prediction
    emotion_label: str = ""
    confidence: float = 0.0
    top_predictions: List[Dict] = field(default_factory=list)

    # AU features
    active_aus: List[str] = field(default_factory=list)
    au_descriptions: str = ""

    # Attention
    attention_regions: List[str] = field(default_factory=list)
    attention_summary: str = ""

    # Explanation
    explanation: str = ""

    # Paths to generated visualizations
    visualization_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "image_path": self.image_path,
            "processing_time": round(self.processing_time, 2),
            "face_found": self.face_found,
            "emotion": {
                "label": self.emotion_label,
                "confidence": round(self.confidence, 4),
                "top_predictions": [
                    {"label": p["label"], "confidence": round(p["confidence"], 4)}
                    for p in self.top_predictions
                ],
            },
            "action_units": {
                "active": self.active_aus,
                "descriptions": self.au_descriptions,
            },
            "attention": {
                "regions": self.attention_regions,
                "summary": self.attention_summary,
            },
            "explanation": self.explanation,
            "visualizations": self.visualization_paths,
        }


class XAIEmotionPipeline:
    """
    End-to-end XAI Emotion Recognition Pipeline.

    Manages sequential model loading to keep VRAM under 6GB:
    Phase 1: Classifier + Grad-ECLIP (~2.5GB)
    Phase 2: LLaVA-7B 4-bit (~4.5GB)
    """

    # Emotion labels for FER2013
    EMOTION_LABELS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def __init__(
        self,
        model_name: str = "poster_v2",
        checkpoint_path: Optional[str] = None,
        attention_method: str = "grad_cam",
        vlm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        vlm_quantization: str = "none",
        device: str = "auto",
        output_dir: str = "outputs",
        emotion_labels: Optional[List[str]] = None,
    ):
        """
        Args:
            model_name: Classifier model ("poster_v2" or "resnet50_cbam").
            checkpoint_path: Path to trained classifier checkpoint.
            attention_method: "grad_eclip" or "grad_cam".
            vlm_model: HuggingFace model name for VLM.
            vlm_quantization: Quantization level ("4bit", "8bit", "none").
            device: Device to use.
            output_dir: Directory for saving outputs.
            emotion_labels: Custom emotion labels list.
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.attention_method = attention_method
        self.vlm_model_name = vlm_model
        self.vlm_quantization = vlm_quantization
        self.output_dir = output_dir
        self.emotion_labels = emotion_labels or self.EMOTION_LABELS

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize components (lightweight — no model loading yet)
        self.face_detector = FaceDetector()
        self.au_extractor = AUExtractor()
        self.region_parser = RegionParser()
        self.prompt_builder = PromptBuilder()

        # Models loaded on-demand
        self.classifier = None
        self.attention_gen = None
        self.vlm_engine = None

        os.makedirs(output_dir, exist_ok=True)

    def _load_classifier(self):
        """Load the emotion classifier and attention generator."""
        if self.classifier is not None:
            return

        print("[Pipeline] Loading classifier...")
        num_classes = len(self.emotion_labels)
        self.classifier = build_model(self.model_name, num_classes=num_classes)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Pipeline] Loaded checkpoint from {self.checkpoint_path}")

        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        # Build attention generator
        self.attention_gen = build_attention_generator(
            self.classifier,
            method=self.attention_method,
            device=self.device,
        )

        if torch.cuda.is_available():
            print(f"[Pipeline] Classifier VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def _unload_classifier(self):
        """Unload classifier to free VRAM for VLM."""
        if self.classifier is not None:
            del self.classifier
            self.classifier = None
        if self.attention_gen is not None:
            del self.attention_gen
            self.attention_gen = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Pipeline] Classifier unloaded, VRAM freed.")

    def _load_vlm(self):
        """Load the VLM engine."""
        if self.vlm_engine is not None:
            return

        self.vlm_engine = VLMEngine(
            model_name=self.vlm_model_name,
            quantization=self.vlm_quantization,
        )
        self.vlm_engine.load()

    def _unload_vlm(self):
        """Unload VLM to free VRAM."""
        if self.vlm_engine is not None:
            self.vlm_engine.unload()
            self.vlm_engine = None

    def predict(
        self,
        image_path: str,
        generate_explanation: bool = True,
        save_output: bool = True,
        skip_face_detection: bool = False,
    ) -> PredictionResult:
        """
        Run the full pipeline on a single image.

        Args:
            image_path: Path to the input image.
            generate_explanation: Whether to generate VLM explanation.
            save_output: Whether to save results to disk.
            skip_face_detection: If True, skip BlazeFace + landmark steps and
                feed the full image directly to the classifier.

        Returns:
            PredictionResult with all pipeline outputs.
        """
        import cv2  # local import — already a project dependency
        start_time = time.time()
        result = PredictionResult(image_path=image_path)

        # ===== STEP 1: Face Detection + Landmarks =====
        if skip_face_detection:
            print(f"\n[Step 1] Skipping face detection — using full image as input.")
            raw_bgr = cv2.imread(image_path)
            if raw_bgr is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            h, w = raw_bgr.shape[:2]
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            full_pil = Image.fromarray(raw_rgb)
            # Build a minimal FaceDetectionResult that looks like a valid detection
            face_result = FaceDetectionResult(
                face_found=True,
                landmarks=None,          # No landmarks → AU step will produce nothing
                landmarks_pixel=None,
                face_crop=raw_bgr,
                face_crop_pil=full_pil,
                bbox=(0, 0, w, h),
                image_shape=(h, w),
            )
        else:
            print(f"\n[Step 1] Detecting face in {image_path}...")
            face_result = self.face_detector.detect_from_path(image_path)

        if not face_result.face_found:
            result.face_found = False
            result.processing_time = time.time() - start_time
            print("[Step 1] No face detected!")
            return result

        result.face_found = True
        print(f"[Step 1] Face detected — bbox: {face_result.bbox}")

        # ===== STEP 2: AU Feature Extraction =====
        print("[Step 2] Extracting Action Unit features...")
        if face_result.landmarks is not None:
            au_result = self.au_extractor.extract(face_result.landmarks)
            result.active_aus = au_result.active_aus
            result.au_descriptions = self.au_extractor.format_for_prompt(au_result)
            print(f"[Step 2] Active AUs: {au_result.active_aus}")
        else:
            result.active_aus = []
            result.au_descriptions = "- No landmark data available (face detection skipped)"
            print("[Step 2] Skipped — no landmarks available.")

        # ===== STEP 3: Emotion Classification =====
        print("[Step 3] Classifying emotion...")
        self._load_classifier()

        # Preprocess face crop for classifier
        import torchvision.transforms as T
        preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(face_result.face_crop_pil).unsqueeze(0)

        # Build landmark tensor for POSTER V2 (shape: 1, 468, 2)
        # face_result.landmarks is (468, 3) normalized [x, y, z] — we use only x, y
        landmark_tensor = None
        if self.model_name == "poster_v2":
            if face_result.landmarks is not None:
                lm_xy = face_result.landmarks[:, :2]  # (468, 2) — x, y only
                landmark_tensor = torch.from_numpy(lm_xy).unsqueeze(0).to(self.device)  # (1, 468, 2)
                print("[Step 3] Passing live MediaPipe landmarks to POSTER V2.")
            else:
                # No landmarks detected — model will use fallback learned tokens
                landmark_tensor = torch.zeros(1, 468, 2, device=self.device)
                print("[Step 3] No landmarks found — POSTER V2 using fallback tokens.")

        with torch.no_grad():
            if landmark_tensor is not None:
                logits = self.classifier(input_tensor.to(self.device), landmarks=landmark_tensor)
            else:
                logits = self.classifier(input_tensor.to(self.device))
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

        # Get top predictions
        top_k = min(3, len(self.emotion_labels))
        top_probs, top_indices = probabilities.topk(top_k)

        result.emotion_label = self.emotion_labels[top_indices[0].item()]
        result.confidence = top_probs[0].item()
        result.top_predictions = [
            {"label": self.emotion_labels[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]
        print(f"[Step 3] Prediction: {result.emotion_label} ({result.confidence:.1%})")

        # ===== STEP 4: Attention Map (Grad-ECLIP) =====
        print(f"[Step 4] Generating attention map ({self.attention_method})...")
        raw_cam = self.attention_gen.generate(input_tensor, top_indices[0].item())

        # ===== STEP 5: Parse Attention Regions =====
        print("[Step 5] Parsing attention regions...")
        attention_regions = self.region_parser.parse(raw_cam)
        result.attention_regions = [r.name for r in attention_regions]
        result.attention_summary = self.region_parser.format_for_prompt(attention_regions)
        print(f"[Step 5] Focus: {self.region_parser.get_summary(attention_regions)}")

        # ===== STEP 6: Generate Explanation (optional) =====
        if generate_explanation:
            print("[Step 6] Generating textual explanation (Qwen-0.5B)...")

            # Swap models: unload classifier → load VLM
            self._unload_classifier()
            self._load_vlm()

            # Build prompt
            prompt = self.prompt_builder.build(
                emotion_label=result.emotion_label,
                confidence=result.confidence,
                au_descriptions=result.au_descriptions,
                attention_descriptions=result.attention_summary,
                top_alternatives=result.top_predictions[1:],
                model_name=self.model_name,
            )

            # Generate
            explanation = self.vlm_engine.generate(
                image=face_result.face_crop_pil,
                prompt=prompt,
            )
            result.explanation = explanation
            print(f"[Step 6] Explanation: {explanation[:100]}...")

            # Unload VLM
            self._unload_vlm()
        else:
            result.explanation = "[Explanation generation skipped]"

        # ===== STEP 7: Save Output =====
        result.processing_time = time.time() - start_time

        if save_output:
            self._save_result(result, raw_cam, face_result)

        print(f"\n[Done] Total time: {result.processing_time:.1f}s")
        return result

    def _save_result(
        self,
        result: PredictionResult,
        raw_cam: np.ndarray,
        face_result: FaceDetectionResult,
    ):
        """Save prediction results and visualizations to disk."""
        base_name = os.path.splitext(os.path.basename(result.image_path))[0]
        output_subdir = os.path.join(self.output_dir, base_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Save JSON result
        json_path = os.path.join(output_subdir, "result.json")
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        result.visualization_paths["json"] = json_path

        # Save face crop
        crop_path = os.path.join(output_subdir, "face_crop.jpg")
        face_result.face_crop_pil.save(crop_path)
        result.visualization_paths["face_crop"] = crop_path

        # Generate and save XAI Visualization Panel
        np_img = np.array(face_result.face_crop_pil)

        # Landmark overlay — use empty array when landmarks are unavailable (e.g. --no-crop mode)
        if face_result.landmarks_pixel is not None:
            lm_for_draw = (
                face_result.landmarks_pixel.cpu().numpy()
                if hasattr(face_result.landmarks_pixel, "cpu")
                else face_result.landmarks_pixel
            )
        else:
            lm_for_draw = np.empty((0, 2), dtype=np.float32)
        ldmk_img = draw_landmarks(np_img, lm_for_draw)
        
        # Heatmap overlay
        hm_img = draw_heatmap_overlay(np_img, raw_cam)
        
        # Combined Panel
        panel_path = os.path.join(output_subdir, "xai_panel.png")
        create_combined_panel(
            original_image=np_img,
            landmark_image=ldmk_img,
            heatmap_image=hm_img,
            emotion_label=result.emotion_label,
            confidence=result.confidence,
            explanation=result.explanation,
            active_aus=result.active_aus,
            attention_summary=result.attention_summary,
            output_path=panel_path
        )
        result.visualization_paths["xai_panel"] = panel_path

        print(f"[Save] Results and Heatmaps saved to {output_subdir}/")

    def predict_batch(
        self,
        image_paths: List[str],
        generate_explanations: bool = False,
    ) -> List[PredictionResult]:
        """
        Run pipeline on multiple images (for evaluation).

        For batch mode, explanations are optional (VRAM swapping is slow).
        """
        results = []
        for i, path in enumerate(image_paths):
            print(f"\n{'='*40}")
            print(f"Processing {i+1}/{len(image_paths)}: {path}")
            result = self.predict(
                path,
                generate_explanation=generate_explanations,
            )
            results.append(result)
        return results

    def close(self):
        """Clean up all resources."""
        self._unload_classifier()
        self._unload_vlm()
        self.face_detector.close()
