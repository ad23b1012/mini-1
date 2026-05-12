"""
XAI-Guided Emotion Trainer (Phase 2).

Extends the standard EmotionTrainer with a FACS-Constrained CAM Attention
Loss that penalises the model during training when its spatial attention
deviates from FACS-derived facial region priors.

This is a NEW trainer — the original EmotionTrainer in train.py is untouched.

Key design decisions:
- Uses CAM (not Grad-CAM) to avoid a second backward pass (~5% overhead)
- Supports ConvNeXt-Tiny, RegNetY-800MF, ResNet-18, EfficientNet-B4
- Logs both classification loss and XAI loss separately for analysis
- Compatible with AMP mixed precision training
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
from src.emotion.train import FocalLoss
from src.emotion.facs_masks import build_facs_masks
from src.emotion.cam_loss import CAMAttentionLoss


def _get_feature_extractor(model, model_name: str):
    """Return a callable that extracts feature maps from the last conv stage."""
    if model_name == "convnext_tiny":
        return model.backbone.features       # → [B, 768, 7, 7]
    elif model_name == "regnet_y_800mf":
        def _extract(x):
            x = model.backbone.stem(x)
            x = model.backbone.trunk_output(x)
            return x
        return _extract                       # → [B, 784, 7, 7]
    elif model_name == "resnet18":
        def _extract(x):
            x = model.backbone.conv1(x)
            x = model.backbone.bn1(x)
            x = model.backbone.relu(x)
            x = model.backbone.maxpool(x)
            x = model.backbone.layer1(x)
            x = model.backbone.layer2(x)
            x = model.backbone.layer3(x)
            x = model.backbone.layer4(x)
            return x
        return _extract                       # → [B, 512, 7, 7]
    elif model_name == "efficientnet_b4":
        return model.backbone.features       # → [B, 1792, 7, 7]
    else:
        raise ValueError(f"XAI training not supported for model: {model_name}")


def _get_fc_weights(model, model_name: str) -> torch.Tensor:
    """Return the final classification Linear layer's weight matrix."""
    if model_name == "convnext_tiny":
        return model.backbone.classifier[3].weight   # [8, 768]
    elif model_name == "regnet_y_800mf":
        return model.backbone.fc[1].weight           # [8, 784]
    elif model_name == "resnet18":
        return model.backbone.fc[1].weight           # [8, 512]
    elif model_name == "efficientnet_b4":
        return model.backbone.classifier[1].weight   # [8, 1792]
    else:
        raise ValueError(f"XAI training not supported for model: {model_name}")


def _forward_from_features(model, model_name: str, feature_maps: torch.Tensor) -> torch.Tensor:
    """Complete the forward pass from feature maps to logits."""
    if model_name == "convnext_tiny":
        x = model.backbone.avgpool(feature_maps)
        x = model.backbone.classifier(x)
        return x
    elif model_name == "regnet_y_800mf":
        x = model.backbone.avgpool(feature_maps)
        x = x.flatten(start_dim=1)
        x = model.backbone.fc(x)
        return x
    elif model_name == "resnet18":
        x = model.backbone.avgpool(feature_maps)
        x = torch.flatten(x, 1)
        x = model.backbone.fc(x)
        return x
    elif model_name == "efficientnet_b4":
        x = model.backbone.avgpool(feature_maps)
        x = model.backbone.classifier(x)
        return x
    else:
        raise ValueError(f"XAI training not supported for model: {model_name}")


class XAIEmotionTrainer:
    """
    Phase 2 Trainer with FACS-Constrained CAM Attention Loss.

    Training loss: L = L_focal + λ × L_cam_facs

    Where:
      - L_focal:    Class-Weighted Focal Loss (classification)
      - L_cam_facs: MSE between CAM heatmap and FACS region mask (explanation)
      - λ:          Weighting factor (default 0.1, ablated from 0.05 to 0.3)
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
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
        xai_lambda: float = 0.1,
    ):
        self.epochs = epochs
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.model_name = model_name
        self.xai_lambda = xai_lambda

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[XAI-Trainer] Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"[XAI-Trainer] GPU: {torch.cuda.get_device_name(0)}")

        # Build model
        self.model = build_model(model_name, num_classes).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[XAI-Trainer] Model: {model_name} | Params: {total_params:,}")
        print(f"[XAI-Trainer] FACS-Constrained CAM Loss: λ = {xai_lambda}")

        # Feature extractor & FC weight accessor for CAM
        self.feature_extractor = _get_feature_extractor(self.model, model_name)

        # Losses
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=label_smoothing)
        self.cam_loss = CAMAttentionLoss()

        # Optimizer & Scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01)
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = learning_rate

        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.start_epoch = 0

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[XAI-Trainer] Resuming from {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt["epoch"]
            self.best_accuracy = ckpt.get("best_accuracy", 0.0)
            print(f"[XAI-Trainer] Resumed at epoch {self.start_epoch}, best_acc: {self.best_accuracy:.2f}%")

        self.history = {
            "train_loss": [], "train_cls_loss": [], "train_xai_loss": [],
            "train_acc": [], "val_loss": [], "val_acc": [], "lr": [],
        }
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _warmup_lr(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr * (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_xai_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [XAI-Train]", leave=False)
        for images, labels, _landmarks in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    # Step 1: Extract feature maps from last conv stage
                    feature_maps = self.feature_extractor(images)

                    # Step 2: Complete forward pass → logits
                    logits = _forward_from_features(self.model, self.model_name, feature_maps)

                    # Step 3: Classification loss
                    cls_loss = self.criterion(logits, labels)

                    # Step 4: CAM attention loss
                    fc_weights = _get_fc_weights(self.model, self.model_name)
                    facs_masks = build_facs_masks(labels, grid_size=feature_maps.shape[-1])
                    xai_loss = self.cam_loss(feature_maps.detach(), fc_weights, labels, facs_masks)

                    # Step 5: Combined loss
                    loss = cls_loss + self.xai_lambda * xai_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                feature_maps = self.feature_extractor(images)
                logits = _forward_from_features(self.model, self.model_name, feature_maps)
                cls_loss = self.criterion(logits, labels)

                fc_weights = _get_fc_weights(self.model, self.model_name)
                facs_masks = build_facs_masks(labels, grid_size=feature_maps.shape[-1])
                xai_loss = self.cam_loss(feature_maps.detach(), fc_weights, labels, facs_masks)

                loss = cls_loss + self.xai_lambda * xai_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_cls_loss += cls_loss.item() * images.size(0)
            total_xai_loss += xai_loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                xai=f"{xai_loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        return {
            "loss": total_loss / total,
            "cls_loss": total_cls_loss / total,
            "xai_loss": total_xai_loss / total,
            "accuracy": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _landmarks in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return {"loss": total_loss / total, "accuracy": 100.0 * correct / total}

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        print(f"\n{'='*60}")
        print(f"XAI-Guided Training: {self.model_name} for {self.epochs} epochs (λ={self.xai_lambda})")
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
            self.history["train_cls_loss"].append(train_metrics["cls_loss"])
            self.history["train_xai_loss"].append(train_metrics["xai_loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"CLS: {train_metrics['cls_loss']:.4f} | "
                f"XAI: {train_metrics['xai_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if val_metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = val_metrics["accuracy"]
                self.best_epoch = epoch + 1
                patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_accuracy": self.best_accuracy,
                    "model_name": self.model_name,
                    "xai_lambda": self.xai_lambda,
                }, os.path.join(self.checkpoint_dir, f"{self.model_name}_xai_best.pth"))
            else:
                patience_counter += 1

            # Save latest checkpoint
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_accuracy": self.best_accuracy,
                "model_name": self.model_name,
                "xai_lambda": self.xai_lambda,
            }
            torch.save(checkpoint_state, os.path.join(self.checkpoint_dir, f"{self.model_name}_xai_last.pth"))

            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint_state, os.path.join(self.checkpoint_dir, f"{self.model_name}_xai_epoch_{epoch+1}.pth"))

            if patience_counter >= self.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {self.early_stopping_patience} epochs.")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"XAI-Guided Training complete in {total_time:.1f}s")
        print(f"Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")

        history_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_xai_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
