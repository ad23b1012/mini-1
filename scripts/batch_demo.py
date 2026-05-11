"""
Batch XAI Demo — Test all models on all 8 emotion classes.

Creates organized output:
    outputs/
    ├── resnet18/
    │   ├── anger/  (xai_panel.png, result.json)
    │   ├── contempt/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    ├── efficientnet_b4/
    │   └── ...
    └── vit/
        └── ...

Usage:
    python scripts/batch_demo.py
"""

import argparse
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import XAIEmotionPipeline

EMOTION_CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

MODELS = {
    "resnet18": "checkpoints/resnet18_best.pth",
    "efficientnet_b4": "checkpoints/efficientnet_b4_best.pth",
    "convnext_tiny": "checkpoints/convnext_tiny_best.pth",
    "regnet_y_800mf": "checkpoints/regnet_y_800mf_best.pth",
}


def get_test_images(data_dir: str, num_per_class: int = 1, seed: int = 42) -> dict:
    """
    Pick test images from the TEST SPLIT ONLY (10% the model never saw).
    
    Uses the same stratified split logic as AffectNetDataset to ensure
    we only pick images from the held-out test set.
    """
    from src.emotion.dataset import AffectNetDataset, AFFECTNET_LABELS

    # Load test split (same seed=42 as training — deterministic)
    test_ds = AffectNetDataset(data_dir, split="test")

    # Group test images by class
    class_images = {cls: [] for cls in AFFECTNET_LABELS}
    for path, label_idx in test_ds.samples:
        cls_name = AFFECTNET_LABELS[label_idx]
        class_images[cls_name].append(path)

    # Pick random samples from test set
    random.seed(seed)
    test_images = {}
    for cls_name, paths in class_images.items():
        if paths:
            selected = random.sample(paths, min(num_per_class, len(paths)))
            test_images[cls_name] = selected
        else:
            print(f"  [WARN] No test images for class: {cls_name}")

    return test_images


def main():
    parser = argparse.ArgumentParser(description="Batch XAI Demo — all models × all classes")
    parser.add_argument("--data-path", type=str, default="dataset_cropped")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-per-class", type=int, default=1, help="Images per class")
    parser.add_argument("--no-explanation", action="store_true", help="Skip VLM explanations")
    parser.add_argument("--models", nargs="+", default=None, help="Models to test (default: all)")
    args = parser.parse_args()

    # Select models
    models_to_test = args.models if args.models else list(MODELS.keys())

    # Get test images
    print(f"\n{'='*60}")
    print(f"  XAI Batch Demo — {len(models_to_test)} Models × 8 Classes")
    print(f"{'='*60}")
    print(f"  Data:        {args.data_path}")
    print(f"  Models:      {models_to_test}")
    print(f"  Explanation: {'No' if args.no_explanation else 'Yes (Qwen-0.5B)'}")
    print(f"{'='*60}\n")

    test_images = get_test_images(args.data_path, num_per_class=args.num_per_class)
    total_tests = sum(len(imgs) for imgs in test_images.values()) * len(models_to_test)
    print(f"Total tests: {total_tests} ({len(test_images)} classes × {args.num_per_class} images × {len(models_to_test)} models)\n")

    for model_name in models_to_test:
        checkpoint = MODELS.get(model_name)
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"\n[SKIP] {model_name}: checkpoint not found at {checkpoint}")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        # Initialize pipeline for this model
        model_output_dir = os.path.join(args.output_dir, model_name)

        pipeline = XAIEmotionPipeline(
            model_name=model_name,
            checkpoint_path=checkpoint,
            attention_method="grad_cam",
            output_dir=model_output_dir,
        )

        for cls_name, image_paths in test_images.items():
            for img_path in image_paths:
                cls_output_dir = os.path.join(model_output_dir, cls_name)
                os.makedirs(cls_output_dir, exist_ok=True)

                print(f"\n  [{model_name}] {cls_name}: {os.path.basename(img_path)}")

                try:
                    # Override output dir for this specific class
                    pipeline.output_dir = cls_output_dir

                    result = pipeline.predict(
                        image_path=img_path,
                        generate_explanation=not args.no_explanation,
                    )

                    # Print summary
                    print(f"    Predicted: {result.emotion_label} ({result.confidence:.1%})")
                    print(f"    Active AUs: {', '.join(result.active_aus)}")
                    print(f"    Attention: {', '.join(result.attention_regions)}")
                    if result.explanation and result.explanation != "[Explanation generation skipped]":
                        print(f"    Explanation: {result.explanation[:100]}...")
                    correct = "[YES]" if result.emotion_label == cls_name else "[NO]"
                    print(f"    Match: {correct}")

                except Exception as e:
                    print(f"    [ERROR] {e}")

        # Close pipeline to free GPU memory before next model
        pipeline.close()
        print(f"\n  [{model_name}] Done — results in {model_output_dir}/")

    print(f"\n{'='*60}")
    print(f"  All tests complete! Results in {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
