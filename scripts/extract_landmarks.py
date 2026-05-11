"""
Extract MediaPipe landmarks for the entire dataset offline.

Saves a JSON file mapping image filenames to 468x2 facial landmark coordinates.
This allows the dataset loader to feed exact geometric landmarks into POSTER V2
without slowing down the training loop.
"""

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detection.detector import FaceDetector


def extract_landmarks(data_dir: str, output_path: str):
    """
    Traverse dataset, extract landmarks, and save to a JSON dict.
    
    Structure:
    {
        "Training_10118481.jpg": [[x1, y1], [x2, y2], ...],
        ...
    }
    """
    print(f"Initializing FaceDetector for {data_dir}...")
    # Use static_image_mode=True for processing uncorrelated images
    detector = FaceDetector(static_image_mode=True, refine_landmarks=False)

    landmarks_dict = {}
    
    image_paths = []
    # Traverse all subdirectories (for train and test sets)
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                
    print(f"Found {len(image_paths)} images. Extracting landmarks...")
    
    for path in tqdm(image_paths, desc="Extracting MediaPipe Landmarks"):
        try:
            # We use detect_from_pil directly or read via PIL -> cv2
            img = Image.open(path).convert('RGB')
            # FaceDetector expects np array (BGR or RGB -> BGR handled automatically?)
            # Actually detect_from_pil converts RGB PIL to BGR cv2 under the hood.
            result = detector.detect_from_pil(img)
            
            fname = os.path.basename(path)
            if result.face_found and result.landmarks is not None:
                # result.landmarks is (468, 3) normalized [0, 1] x,y,z
                # We extract (468, 2) x,y coordinates
                lm_2d = result.landmarks[:, :2]
                landmarks_dict[fname] = lm_2d.tolist()
            else:
                # Keep track of failures but do not output a tensor here,
                # Dataset loader handles fallback to zeros
                pass
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    detector.close()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(landmarks_dict, f)
        
    print(f"\n✅ Extracted {len(landmarks_dict)} valid landmarks out of {len(image_paths)} images.")
    print(f"✅ Saved landmarks dictionary to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data split directory (e.g., FER/train)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path (e.g., data/fer2013_train_landmarks.json)")
    args = parser.parse_args()
    
    extract_landmarks(args.data_dir, args.output)
