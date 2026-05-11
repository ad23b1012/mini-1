"""
Evaluate a flat folder of unlabelled test images.

Usage:
    uv run python scripts/eval_test_folder.py --data-path Test --output-dir outputs
"""

import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import XAIEmotionPipeline

MODELS = {
    "resnet18": "checkpoints/resnet18_best.pth",
    "efficientnet_b4": "checkpoints/efficientnet_b4_best.pth",
    "convnext_tiny": "checkpoints/convnext_tiny_best.pth",
    "regnet_y_800mf": "checkpoints/regnet_y_800mf_best.pth",
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate unseen test folder")
    parser.add_argument("--data-path", type=str, default="Test", help="Path to folder with raw images")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    # Find all images in the folder
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.data_path, ext)))
        image_paths.extend(glob.glob(os.path.join(args.data_path, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {args.data_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Evaluating {len(image_paths)} Unseen Images from '{args.data_path}'")
    print(f"{'='*60}\n")

    for model_name, checkpoint_path in MODELS.items():
        if not os.path.exists(checkpoint_path):
            print(f"[SKIP] Checkpoint not found for {model_name} at {checkpoint_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Running Pipeline: {model_name}")
        print(f"{'='*60}")

        model_output_dir = os.path.join(args.output_dir, model_name, "test_unseen")
        os.makedirs(model_output_dir, exist_ok=True)

        pipeline = XAIEmotionPipeline(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            attention_method="grad_cam",
            output_dir=model_output_dir,
        )

        for img_path in image_paths:
            print(f"\n  Processing: {os.path.basename(img_path)}")
            try:
                result = pipeline.predict(
                    image_path=img_path,
                    generate_explanation=True,
                    save_output=True
                )
                
                # Move the saved folder to be organized by predicted class
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                current_dir = os.path.join(model_output_dir, base_name)
                
                if result.emotion_label:
                    predicted_class_dir = os.path.join(model_output_dir, result.emotion_label)
                    os.makedirs(predicted_class_dir, exist_ok=True)
                    new_dir = os.path.join(predicted_class_dir, base_name)
                    
                    # Rename/move directory to organize by predicted emotion
                    if os.path.exists(current_dir):
                        if os.path.exists(new_dir):
                            import shutil
                            shutil.rmtree(new_dir)
                        os.rename(current_dir, new_dir)
                        print(f"    -> Saved to {model_name}/test_unseen/{result.emotion_label}/{base_name}/")

            except Exception as e:
                print(f"    [ERROR] Failed to process {img_path}: {e}")

        pipeline.close()

    print(f"\n{'='*60}")
    print(f"  All tests complete! Results structured by predicted class in {args.output_dir}/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
