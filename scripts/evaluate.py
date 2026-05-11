"""
Evaluate the trained classifier on test sets.

Usage:
    python scripts/evaluate.py --model resnet18 --dataset affectnet \
        --data-path dataset --checkpoint checkpoints/resnet18_best.pth
"""

import argparse
import os
import sys
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.emotion.model import build_model
from src.emotion.dataset import get_dataloader, get_num_classes, get_labels
from src.visualization import create_confusion_matrix_plot
from sklearn.metrics import classification_report

ALL_MODELS = ["poster_v2", "resnet50_cbam", "resnet18", "efficientnet_b4"]
ALL_DATASETS = ["affectnet", "fer2013", "rafdb"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier")
    parser.add_argument(
        "--model", type=str, default="resnet18",
        choices=ALL_MODELS,
    )
    parser.add_argument(
        "--dataset", type=str, default="affectnet",
        choices=ALL_DATASETS,
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = get_num_classes(args.dataset)
    labels = get_labels(args.dataset)

    # Load model
    model = build_model(args.model, num_classes=num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded {args.model} from {args.checkpoint}")
    print(f"Checkpoint accuracy: {checkpoint.get('best_accuracy', 'N/A')}%")

    # Create dataloader
    test_loader, _ = get_dataloader(
        args.dataset, args.data_path, split=args.split,
        batch_size=args.batch_size, augment=False, num_workers=0
    )

    # Evaluate
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # Only POSTER V2 accepts a landmarks tensor in its forward() method
    uses_landmarks = args.model == "poster_v2"

    with torch.no_grad():
        for images, targets, landmarks in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if uses_landmarks:
                landmarks = landmarks.to(device, non_blocking=True)
                outputs = model(images, landmarks=landmarks)
            else:
                outputs = model(images)

            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = 100.0 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=labels)
    print(report)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Confusion matrix
    create_confusion_matrix_plot(
        all_labels, all_preds, labels,
        output_path=os.path.join(args.output_dir, f"{args.model}_{args.dataset}_confusion.png"),
        title=f"{args.model} on {args.dataset} ({args.split}) — {accuracy:.2f}%",
    )

    # Save report
    report_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "accuracy": accuracy,
            "classification_report": classification_report(
                all_labels, all_preds, target_names=labels, output_dict=True
            ),
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
