"""
Demo script — Run the full XAI pipeline on a single image.

Usage:
    python scripts/demo.py --image path/to/face.jpg
    python scripts/demo.py --image path/to/face.jpg --no-explanation  # Skip VLM
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import XAIEmotionPipeline


def main():
    parser = argparse.ArgumentParser(description="XAI Emotion Recognition Demo")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input face image",
    )
    parser.add_argument(
        "--model", type=str, default="convnext_tiny",
        choices=["poster_v2", "resnet50_cbam", "resnet18", "efficientnet_b4",
                 "convnext_tiny", "regnet_y_800mf"],
        help="Classifier model",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--attention", type=str, default="grad_cam",
        choices=["grad_eclip", "grad_cam"],
        help="Attention map method",
    )
    parser.add_argument(
        "--no-explanation", action="store_true",
        help="Skip VLM explanation generation",
    )
    parser.add_argument(
        "--no-crop", action="store_true",
        help="Skip face detection — pass the full image directly to the model",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Output directory",
    )

    args = parser.parse_args()

    # Auto-detect checkpoint if not specified
    if args.checkpoint is None:
        auto_path = os.path.join("checkpoints", f"{args.model}_best.pth")
        if os.path.exists(auto_path):
            args.checkpoint = auto_path
            print(f"[Auto] Found checkpoint: {auto_path}")
        else:
            print(f"[WARNING] No checkpoint found at {auto_path} — model will use untrained weights!")

    print(f"\n{'='*60}")
    print(f"  XAI Emotion Recognition — Demo")
    print(f"{'='*60}")
    print(f"  Image:      {args.image}")
    print(f"  Model:      {args.model}")
    print(f"  Checkpoint: {args.checkpoint or 'NONE (untrained!)'}")
    print(f"  Attention:  {args.attention}")
    print(f"  VLM:        {'Disabled' if args.no_explanation else 'Qwen-0.5B'}")
    print(f"{'='*60}\n")

    # Initialize pipeline
    pipeline = XAIEmotionPipeline(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        attention_method=args.attention,
        output_dir=args.output_dir,
    )

    # Run prediction
    result = pipeline.predict(
        image_path=args.image,
        generate_explanation=not args.no_explanation,
        skip_face_detection=args.no_crop,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"\n  Emotion:      {result.emotion_label.upper()}")
    print(f"  Confidence:   {result.confidence:.1%}")
    print(f"\n  Active AUs:   {', '.join(result.active_aus) or 'None'}")
    print(f"\n  Attention:    {', '.join(result.attention_regions)}")
    print(f"\n  Explanation:")
    print(f"  {result.explanation}")
    print(f"\n  Time:         {result.processing_time:.1f}s")
    print(f"{'='*60}\n")

    # Save JSON
    print(json.dumps(result.to_dict(), indent=2))

    pipeline.close()


if __name__ == "__main__":
    main()
