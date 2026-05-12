"""
Phase 2: Train with FACS-Constrained CAM Attention Loss.

This script uses the XAIEmotionTrainer which adds a spatial attention
supervision loss during training. The model is penalised when its CAM
heatmap deviates from FACS-derived facial region priors.

Usage:
    # Default λ=0.1
    python scripts/train_xai.py --model convnext_tiny --dataset affectnet \
        --data-path dataset_cropped --epochs 50 --lr 1e-4

    # Ablation: try different λ values
    python scripts/train_xai.py --model convnext_tiny --dataset affectnet \
        --data-path dataset_cropped --xai-lambda 0.05

    python scripts/train_xai.py --model convnext_tiny --dataset affectnet \
        --data-path dataset_cropped --xai-lambda 0.3
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion.xai_train import XAIEmotionTrainer
from src.emotion.dataset import get_dataloader, get_num_classes, get_labels
from src.visualization import plot_training_history


BATCH_SIZE_RECOMMENDATIONS = {
    "convnext_tiny": 32,
    "regnet_y_800mf": 64,
    "resnet18": 64,
    "efficientnet_b4": 24,
}

ALL_MODELS = ["convnext_tiny", "regnet_y_800mf", "resnet18", "efficientnet_b4"]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: XAI-Guided Training with FACS-Constrained CAM Loss"
    )
    parser.add_argument("--model", type=str, default="convnext_tiny", choices=ALL_MODELS)
    parser.add_argument("--dataset", type=str, default="affectnet", choices=["affectnet"])
    parser.add_argument("--data-path", type=str, default="dataset_cropped")
    parser.add_argument("--landmarks-file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--xai-lambda", type=float, default=0.1,
                        help="Weight for FACS-CAM attention loss (ablate: 0.05, 0.1, 0.3)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Defaults to checkpoints/xai_lambda_<value>")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default=None)

    args = parser.parse_args()

    # Auto batch size
    if args.batch_size is None:
        args.batch_size = BATCH_SIZE_RECOMMENDATIONS.get(args.model, 32)

    # Auto checkpoint dir — separate per λ value for ablation
    if args.checkpoint_dir is None:
        lambda_str = f"{args.xai_lambda:.2f}".replace(".", "")
        args.checkpoint_dir = f"checkpoints/xai_lambda_{lambda_str}"

    num_classes = get_num_classes(args.dataset)
    labels = get_labels(args.dataset)

    # Auto-resume
    if args.resume_checkpoint is None:
        last_path = os.path.join(args.checkpoint_dir, f"{args.model}_xai_last.pth")
        if os.path.exists(last_path):
            args.resume_checkpoint = last_path

    print(f"\n{'='*60}")
    print("XAI Emotion Recognition — Phase 2: FACS-Guided Training")
    print(f"{'='*60}")
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset} ({num_classes} classes)")
    print(f"Classes:    {labels}")
    print(f"Data Dir:   {args.data_path}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learn Rate: {args.lr}")
    print(f"Weight Dec: {args.weight_decay}")
    print(f"XAI Lambda: {args.xai_lambda}")
    print(f"Ckpt Dir:   {args.checkpoint_dir}")
    if args.resume_checkpoint:
        print(f"Resuming:   {args.resume_checkpoint}")
    print(f"{'='*60}\n")

    import torch
    torch.backends.cudnn.benchmark = True

    # Create dataloaders
    train_loader, train_weights = get_dataloader(
        "affectnet", args.data_path, split="train",
        batch_size=args.batch_size, image_size=args.image_size,
        augment=True, num_workers=args.num_workers,
        landmarks_file=args.landmarks_file
    )
    val_loader, _ = get_dataloader(
        "affectnet", args.data_path, split="val",
        batch_size=args.batch_size, image_size=args.image_size,
        augment=False, num_workers=args.num_workers,
        landmarks_file=args.landmarks_file
    )

    print("Class weights assigned for Focal Loss balancing:")
    print(train_weights.cpu().numpy().round(4))

    trainer = XAIEmotionTrainer(
        model_name=args.model,
        num_classes=num_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        class_weights=train_weights,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
        xai_lambda=args.xai_lambda,
    )

    # Train
    history = trainer.train(train_loader, val_loader)

    # Plot
    os.makedirs("outputs", exist_ok=True)
    lambda_str = f"{args.xai_lambda:.2f}".replace(".", "")
    plot_training_history(
        history,
        output_path=f"outputs/{args.model}_xai_lambda{lambda_str}_training_history.png",
        title=f"{args.model} XAI-Guided (λ={args.xai_lambda})",
    )

    print(f"\n✅ XAI-Guided Training complete! Best accuracy: {trainer.best_accuracy:.2f}%")
    print(f"   Checkpoint: {args.checkpoint_dir}/{args.model}_xai_best.pth")
    print(f"   λ = {args.xai_lambda}")


if __name__ == "__main__":
    main()
