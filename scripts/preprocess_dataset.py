"""
Preprocess the dataset: YOLO face crop + MediaPipe landmark extraction.

This script must be run BEFORE training to ensure:
1. All images are tightly cropped to face-only (no hair, neck, background)
2. MediaPipe landmarks are pre-extracted and saved for POSTER V2 / AU analysis
3. Train-test consistency: model sees the same type of input during training and inference

Usage:
    python scripts/preprocess_dataset.py --input dataset --output dataset_cropped
"""

import argparse
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detection.detector import FaceDetector


def preprocess_dataset(input_dir: str, output_dir: str, save_landmarks: bool = True):
    """
    Process the entire dataset:
    1. Detect face with YOLO → crop tightly (face only)
    2. Extract MediaPipe landmarks from the crop
    3. Save cropped face image + landmarks JSON

    Args:
        input_dir: Path to raw dataset (e.g., dataset/)
        output_dir: Path to save processed dataset (e.g., dataset_cropped/)
        save_landmarks: Whether to extract and save MediaPipe landmarks
    """
    # Initialize face detector (YOLO + MediaPipe)
    detector = FaceDetector(
        use_yolo=True,
        face_crop_padding=0.05,  # Tight face-only crop
    )

    # Get all class folders
    classes = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    print(f"Found {len(classes)} classes: {classes}")

    # Stats tracking
    total_images = 0
    successful_crops = 0
    failed_crops = 0
    landmarks_dict = {}  # filename → landmarks list

    for class_name in classes:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Get all image files
        image_files = sorted([
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ])

        print(f"\n[{class_name}] Processing {len(image_files)} images...")

        for fname in tqdm(image_files, desc=f"  {class_name}", leave=False):
            total_images += 1
            input_path = os.path.join(class_input_dir, fname)
            output_path = os.path.join(class_output_dir, fname)

            try:
                # Read image
                image = cv2.imread(input_path)
                if image is None:
                    failed_crops += 1
                    continue

                # Detect and crop face
                result = detector.detect(image)

                if result.face_found and result.face_crop is not None:
                    # Save the YOLO-cropped face
                    # Resize to a standard size (224x224) for consistency
                    face_crop = cv2.resize(result.face_crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Save as PNG for lossless quality, or JPG to match original
                    if fname.lower().endswith('.png'):
                        cv2.imwrite(output_path, face_crop)
                    else:
                        cv2.imwrite(output_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # Save landmarks if available
                    if save_landmarks and result.landmarks is not None:
                        landmarks_dict[fname] = result.landmarks[:, :2].tolist()

                    successful_crops += 1
                else:
                    # Fallback: if no face detected, copy original image resized
                    # (some images may already be tight crops)
                    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                    if fname.lower().endswith('.png'):
                        cv2.imwrite(output_path, img_resized)
                    else:
                        cv2.imwrite(output_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    successful_crops += 1  # Still usable, just not YOLO-cropped

            except Exception as e:
                print(f"\n  [ERROR] {fname}: {e}")
                failed_crops += 1
                continue

    # Save landmarks JSON
    if save_landmarks and landmarks_dict:
        landmarks_path = os.path.join(output_dir, "landmarks.json")
        with open(landmarks_path, "w") as f:
            json.dump(landmarks_dict, f)
        print(f"\n[Landmarks] Saved {len(landmarks_dict)} landmark sets to {landmarks_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Preprocessing Complete")
    print(f"{'='*60}")
    print(f"  Total images:     {total_images}")
    print(f"  Successful crops: {successful_crops}")
    print(f"  Failed:           {failed_crops}")
    print(f"  Success rate:     {100*successful_crops/max(total_images,1):.1f}%")
    print(f"  Output dir:       {output_dir}")
    print(f"{'='*60}")

    detector.close()


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset with YOLO face cropping")
    parser.add_argument("--input", type=str, default="dataset",
                        help="Path to raw dataset folder")
    parser.add_argument("--output", type=str, default="dataset_cropped",
                        help="Path to save preprocessed dataset")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Skip landmark extraction")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  XAI Dataset Preprocessor")
    print(f"{'='*60}")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Landmarks:  {'No' if args.no_landmarks else 'Yes'}")
    print(f"{'='*60}\n")

    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        save_landmarks=not args.no_landmarks,
    )


if __name__ == "__main__":
    main()
