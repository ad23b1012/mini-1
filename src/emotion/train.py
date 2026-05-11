"""
Training loop for emotion classifiers.

Supports both POSTER V2 and ResNet-50+CBAM models with:
- Mixed precision training (AMP) for RTX 4050 efficiency
- AdamW optimizer with cosine annealing LR schedule
- Class-Balanced Focal Loss with label smoothing for imbalanced data
- Gradient accumulation for effective larger batch sizes
- Best checkpoint saving with validation accuracy tracking
- Early stopping
- Compatible with both landmark-aware (POSTER V2) and standard CNN models
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm
import numpy as np

from src.emotion.model import build_model


class FocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for imbalanced datasets.
    Reduces the relative loss for well-classified examples, putting more
    focus on hard, misclassified examples (like 'disgust' class in FER).
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce_loss(inputs, targets)
        pt = torch.exp(log_pt)
        # Apply Focal factor: (1 - pt)^gamma
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        return focal_loss.mean()


class EmotionTrainer:
    """
    Trainer for emotion classification models.

    Manages the full training lifecycle: training loop, validation,
    checkpointing, logging, and early stopping.
    """

    def __init__(
        self,
        model_name: str = "poster_v2",
        num_classes: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 50,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        warmup_epochs: int = 5,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 10,
        device: str = "auto",
        resume_checkpoint: Optional[str] = None,
    ):
        self.epochs = epochs
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.model_name = model_name

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Trainer] Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")

        # Build model
        self.model = build_model(model_name, num_classes).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Trainer] Model: {model_name} | Total params: {total_params:,}")

        # Determine if model uses landmarks — ONLY POSTER V2 has the landmark
        # projection layer and cross-attention that accepts a landmarks tensor.
        # ResNet50-CBAM, ResNet-18, and EfficientNet-B4 do NOT accept landmarks.
        self.uses_landmarks = model_name == "poster_v2"

        # SOTA Loss handling data imbalance
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print(f"[Trainer] Using Class-Balanced Focal Loss with weights: {class_weights.cpu().numpy().round(3)}")
        self.criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=label_smoothing)

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=learning_rate * 0.01,
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = learning_rate

        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.start_epoch = 0

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[Trainer] Resuming precisely from checkpoint {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt["epoch"]
            self.best_accuracy = ckpt.get("best_accuracy", 0.0)
            print(f"[Trainer] Resumed actively at epoch {self.start_epoch} with previous best_acc: {self.best_accuracy:.2f}%")
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _warmup_lr(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, labels, landmarks in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.uses_landmarks:
                landmarks = landmarks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    if self.uses_landmarks:
                        logits = self.model(images, landmarks=landmarks)
                    else:
                        logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.uses_landmarks:
                    logits = self.model(images, landmarks=landmarks)
                else:
                    logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        return {"loss": total_loss / total, "accuracy": 100.0 * correct / total}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels, landmarks in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.uses_landmarks:
                landmarks = landmarks.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    if self.uses_landmarks:
                        logits = self.model(images, landmarks=landmarks)
                    else:
                        logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                if self.uses_landmarks:
                    logits = self.model(images, landmarks=landmarks)
                else:
                    logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            "loss": total_loss / total,
            "accuracy": 100.0 * correct / total,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels),
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} for {self.epochs} epochs")
        print(f"{'='*60}\n")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            self._warmup_lr(epoch)

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if val_metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = val_metrics["accuracy"]
                self.best_epoch = epoch + 1
                patience_counter = 0

                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_accuracy": self.best_accuracy,
                    "model_name": self.model_name,
                }, checkpoint_path)
            else:
                patience_counter += 1

            # Save latest checkpoint iteratively every single epoch explicitly for pausing and resuming
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_accuracy": self.best_accuracy,
                "model_name": self.model_name,
            }
            
            last_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_last.pth")
            torch.save(checkpoint_state, last_path)
            
            # Save periodic checkpoints every 5 epochs to keep historical in-between states
            if (epoch + 1) % 5 == 0:
                epoch_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch_{epoch + 1}.pth")
                torch.save(checkpoint_state, epoch_path)

            if patience_counter >= self.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {self.early_stopping_patience} epochs.")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        
        history_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def load_best(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_accuracy = checkpoint["best_accuracy"]
        print(f"[Trainer] Loaded best model (accuracy: {self.best_accuracy:.2f}%)")
