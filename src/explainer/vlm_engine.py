"""
VLM Engine — LLaVA-7B-based Explanation Generator.

Uses LLaVA-1.5-7B with 4-bit quantization (via bitsandbytes) to generate
natural language explanations for emotion predictions.

VRAM Usage: ~4.5GB at 4-bit quantization.
Strategy: Sequential loading — classifier is unloaded before VLM is loaded,
keeping peak VRAM under 5GB on the RTX 4050 (6GB).
"""

import torch
from typing import Optional, List, Dict
from PIL import Image
import gc


class VLMEngine:
    """
    Qwen-0.5B lightweight text model for explanation generation.

    Loads the model efficiently to easily fit on 16GB RAM and 6GB VRAM.
    Generates grounded natural language explanations from structured evidence prompts.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        quantization: str = "none",
        max_new_tokens: int = 150,
        temperature: float = 0.3,
        do_sample: bool = False,
        device: str = "auto",
    ):
        """
        Args:
            model_name: HuggingFace model ID for LLaVA.
            quantization: "4bit", "8bit", or "none".
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            do_sample: Whether to use sampling (False = greedy).
            device: Device to use.
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.device = device

        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """
        Load the LLaVA model with quantization.

        Call this AFTER freeing VRAM from the classifier
        (torch.cuda.empty_cache()).
        """
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM] Loading {self.model_name}...")

        # Configure loading
        model_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }

        self.processor = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        self._loaded = True
        print(f"[LLM] Model loaded successfully!")

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"[VLM] VRAM usage: {vram_used:.2f} GB")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Generate an explanation for the given face image and prompt.

        Args:
            image: Face image as PIL Image.
            prompt: Structured evidence prompt (from PromptBuilder).

        Returns:
            Generated explanation text.
        """
        if not self._loaded:
            self.load()

        # Format for Qwen Chat prompt
        messages = [
            {"role": "system", "content": "You are an expert Explainable AI (XAI) assistant. Write a short, single-paragraph explanation about why the AI predicted this emotion based on the evidence."},
            {"role": "user", "content": prompt}
        ]
        
        full_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs (Text only)
        inputs = self.processor(
            text=full_prompt,
            return_tensors="pt",
        )

        # Move to same device as model
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                use_cache=True,
            )

        # Decode — only get the newly generated tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        explanation = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean up weird markdown artifacts common in small text models
        explanation = explanation.replace("```text", "").replace("```", "").strip()

        return explanation

    def unload(self):
        """
        Unload the model and free VRAM.

        Call this before loading the classifier for the next image.
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        self._loaded = False
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[VLM] Model unloaded, VRAM freed.")

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
