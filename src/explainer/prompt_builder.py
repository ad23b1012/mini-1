"""
Prompt Builder for the Explanation Generator.

Constructs structured prompts from pipeline outputs (emotion prediction,
AU features, attention regions) that are fed to LLaVA-7B to generate
grounded natural language explanations.
"""

from typing import List, Optional, Dict


# =============================================================================
# Emotion → AU Mapping (ground truth correlations from FACS literature)
# Used to cross-validate model predictions against known AU-emotion patterns.
# =============================================================================

EMOTION_AU_MAP = {
    "anger": {
        "expected_aus": ["AU4", "AU5", "AU17", "AU24"],
        "key_features": [
            "eyebrows furrowed and lowered",
            "eyes wide or narrowed",
            "lips pressed or tightened",
            "chin raised",
        ],
    },
    "contempt": {
        "expected_aus": ["AU12", "AU14"],
        "key_features": [
            "unilateral lip corner pull (asymmetric smile)",
            "one-sided dimple",
            "slight head tilt",
            "lips may be slightly pressed",
        ],
    },
    "disgust": {
        "expected_aus": ["AU4", "AU6", "AU9", "AU15", "AU25"],
        "key_features": [
            "nose wrinkled",
            "upper lip raised",
            "eyebrows lowered",
            "lip corners pulled down",
        ],
    },
    "fear": {
        "expected_aus": ["AU1", "AU2", "AU4", "AU5", "AU20", "AU26"],
        "key_features": [
            "inner eyebrows raised",
            "eyes wide open",
            "mouth open / jaw dropped",
            "lips stretched horizontally",
        ],
    },
    "happy": {
        "expected_aus": ["AU6", "AU12", "AU25"],
        "key_features": [
            "lip corners pulled up (smile)",
            "cheeks raised",
            "crow's feet wrinkles around eyes",
            "teeth may be visible",
        ],
    },
    "sad": {
        "expected_aus": ["AU1", "AU4", "AU15", "AU17"],
        "key_features": [
            "inner eyebrows raised",
            "eyebrows furrowed",
            "lip corners pulled down",
            "chin raised/wrinkled",
        ],
    },
    "surprise": {
        "expected_aus": ["AU1", "AU2", "AU5", "AU26"],
        "key_features": [
            "eyebrows raised (both inner and outer)",
            "eyes wide open",
            "mouth open / jaw dropped",
        ],
    },
    "neutral": {
        "expected_aus": [],
        "key_features": [
            "relaxed facial muscles",
            "no prominent AU activations",
            "calm expression",
        ],
    },
}


class PromptBuilder:
    """
    Constructs structured prompts for the LLaVA-7B explanation generator.

    The prompt includes:
    1. Predicted emotion and confidence
    2. Detected facial action units (AUs)
    3. Model attention regions (from Grad-ECLIP)
    4. Top alternative predictions
    5. FACS-based expected features for cross-validation
    """

    def __init__(
        self,
        max_explanation_words: int = 100,
        include_facs_reference: bool = True,
    ):
        """
        Args:
            max_explanation_words: Target maximum words for the explanation.
            include_facs_reference: Whether to include FACS reference features.
        """
        self.max_explanation_words = max_explanation_words
        self.include_facs_reference = include_facs_reference

    def build(
        self,
        emotion_label: str,
        confidence: float,
        au_descriptions: str,
        attention_descriptions: str,
        top_alternatives: Optional[List[Dict[str, float]]] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Build the full prompt for LLaVA-7B.

        Args:
            emotion_label: Predicted emotion (e.g., "happy").
            confidence: Prediction confidence (0-1).
            au_descriptions: Formatted AU feature descriptions (from AUExtractor).
            attention_descriptions: Formatted attention region descriptions (from RegionParser).
            top_alternatives: List of dicts [{"label": str, "confidence": float}, ...].
            model_name: Name of the classification model used.

        Returns:
            Complete prompt string for the VLM.
        """
        # Format alternatives
        alt_text = "None"
        if top_alternatives:
            alt_lines = []
            for alt in top_alternatives[:3]:
                alt_lines.append(f"  - {alt['label']}: {alt['confidence']:.1%}")
            alt_text = "\n".join(alt_lines)

        # FACS reference features
        facs_ref = ""
        if self.include_facs_reference and emotion_label.lower() in EMOTION_AU_MAP:
            expected = EMOTION_AU_MAP[emotion_label.lower()]
            if expected["key_features"]:
                facs_ref = "\n\n**Expected FACS Features for this emotion:**\n"
                for feat in expected["key_features"]:
                    facs_ref += f"  - {feat}\n"

        prompt = f"""You are an expert in facial expression analysis and emotion recognition using the Facial Action Coding System (FACS).

You are given evidence from a multimodal emotion recognition system. Your task is to provide a clear, clinical explanation of why the predicted emotion is correct, grounding your explanation in the observed facial features and model attention.

## Evidence

**Model Architecture:** {model_name if model_name else 'Emotion Classifier'}

**Predicted Emotion:** {emotion_label} (confidence: {confidence:.1%})

**Facial Action Units Detected:**
{au_descriptions}

**Model Attention Regions:**
{attention_descriptions}

**Top Alternative Predictions:**
{alt_text}{facs_ref}

## Task

Explain WHY this face shows "{emotion_label}". You MUST:
1. Reference specific facial features observed (e.g., "raised inner eyebrows", "lip corners pulled up")
2. Explain how these features correspond to the predicted emotion based on FACS
3. If the attention regions align with the detected AUs, mention this as supporting evidence
4. If confidence is low or alternatives are close, briefly note the ambiguity

Keep your explanation under {self.max_explanation_words} words. Be clinical, precise, and evidence-based. Do not speculate beyond the provided evidence."""

        return prompt

    def build_conversation(
        self,
        emotion_label: str,
        confidence: float,
        au_descriptions: str,
        attention_descriptions: str,
        top_alternatives: Optional[List[Dict[str, float]]] = None,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build conversation-format prompt for LLaVA-7B's chat interface.

        Returns a list of message dicts compatible with transformers chat template.
        """
        user_prompt = self.build(
            emotion_label, confidence, au_descriptions,
            attention_descriptions, top_alternatives, model_name
        )

        return [
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
