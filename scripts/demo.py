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
        "--model", type=str, default="resnet18",
        choices=["poster_v2", "resnet50_cbam", "resnet18", "efficientnet_b4"],
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
        "--output-dir", type=str, default="outputs",
        help="Output directory",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  XAI Emotion Recognition — Demo")
    print(f"{'='*60}")
    print(f"  Image:      {args.image}")
    print(f"  Model:      {args.model}")
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
