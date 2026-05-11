"""
Visualization module for XAI Emotion Recognition.

Generates publication-quality visualizations:
1. Face with landmark overlay
2. Attention heatmap overlay
3. Combined panel (original | landmarks | heatmap | explanation text)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
from PIL import Image
from typing import Optional, List, Tuple
import os


def draw_landmarks(
    image: np.ndarray,
    landmarks_pixel: np.ndarray,
    active_au_indices: Optional[List[int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 1,
) -> np.ndarray:
    """
    Draw facial landmarks on an image.

    Args:
        image: Input image (BGR format).
        landmarks_pixel: Pixel coordinates of landmarks (N, 2).
        active_au_indices: Landmark indices involved in active AUs (highlighted).
        color: Default landmark color (BGR).
        radius: Landmark circle radius.

    Returns:
        Image with landmarks drawn.
    """
    output = image.copy()

    for i, pt in enumerate(landmarks_pixel):
        x, y = int(pt[0]), int(pt[1])
        if active_au_indices and i in active_au_indices:
            cv2.circle(output, (x, y), radius + 1, (0, 0, 255), -1)  # Red for active AUs
        else:
            cv2.circle(output, (x, y), radius, color, -1)

    return output


def draw_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
    add_colorbar: bool = True,
) -> np.ndarray:
    """
    Overlay attention heatmap on an image with a publication-quality colorbar.

    Args:
        image: Original image (RGB format, uint8).
        heatmap: Attention map (any size, values in [0, 1]).
        colormap: OpenCV colormap.
        alpha: Heatmap overlay transparency.
        add_colorbar: Whether to add a vertical colorbar scale.

    Returns:
        Image with heatmap overlay and optional colorbar (RGB, uint8).
    """
    h, w = image.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 255]
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (alpha * heatmap_colored.astype(float) +
               (1 - alpha) * image.astype(float)).astype(np.uint8)

    if add_colorbar:
        overlay = _add_colorbar(overlay, colormap)

    return overlay


def _add_colorbar(
    image: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    bar_width: int = 25,
    padding: int = 10,
) -> np.ndarray:
    """
    Add a vertical colorbar to the right side of an image.
    Shows Low (blue) → High (red) intensity scale for publication.
    """
    h, w = image.shape[:2]
    
    # Create the colorbar gradient (vertical, bottom=0 to top=255)
    gradient = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    
    # Apply the same colormap
    colorbar = cv2.applyColorMap(gradient, colormap)
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB)
    
    # Create a white background strip for padding + bar + labels
    label_width = 40
    strip_width = padding + bar_width + label_width
    strip = np.ones((h, strip_width, 3), dtype=np.uint8) * 255
    
    # Place colorbar in the strip
    strip[:, padding:padding + bar_width] = colorbar
    
    # Add border around colorbar
    cv2.rectangle(strip, (padding - 1, 0), (padding + bar_width, h - 1), (0, 0, 0), 1)
    
    # Add text labels using cv2 (no matplotlib dependency for this)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text_color = (0, 0, 0)
    
    # "High" at top
    cv2.putText(strip, "High", (padding + bar_width + 3, 15),
                font, font_scale, text_color, thickness, cv2.LINE_AA)
    # "Low" at bottom  
    cv2.putText(strip, "Low", (padding + bar_width + 3, h - 5),
                font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Concatenate image and colorbar strip
    result = np.concatenate([image, strip], axis=1)
    return result


def create_combined_panel(
    original_image: np.ndarray,
    landmark_image: np.ndarray,
    heatmap_image: np.ndarray,
    emotion_label: str,
    confidence: float,
    explanation: str,
    active_aus: List[str],
    attention_summary: str,
    output_path: str,
    figsize: Tuple[int, int] = (20, 6),
):
    """
    Create a publication-quality combined visualization panel.

    Layout: [Original | Landmarks | Heatmap | Text]

    Args:
        original_image: Original face image (RGB).
        landmark_image: Face with landmark overlay (RGB).
        heatmap_image: Face with heatmap overlay (RGB).
        emotion_label: Predicted emotion.
        confidence: Prediction confidence.
        explanation: VLM explanation text.
        active_aus: List of active AU codes.
        attention_summary: Attention region summary.
        output_path: Path to save the panel.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Panel 1: Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Landmarks
    axes[1].imshow(landmark_image)
    axes[1].set_title("Face Landmarks + AUs", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Heatmap
    axes[2].imshow(heatmap_image)
    axes[2].set_title(f"Attention Map\n(Grad-CAM)", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Text explanation
    axes[3].axis("off")
    text_content = (
        f"Emotion: {emotion_label.upper()}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Active AUs: {', '.join(active_aus) if active_aus else 'None'}\n\n"
        f"Attention:\n{attention_summary}\n\n"
        f"Explanation:\n{explanation}"
    )
    axes[3].text(
        0.05, 0.95, text_content,
        transform=axes[3].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )
    axes[3].set_title("XAI Explanation", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Viz] Combined panel saved to {output_path}")


def create_confusion_matrix_plot(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
):
    """
    Create a publication-quality confusion matrix plot.

    Args:
        labels: Ground truth labels.
        predictions: Model predictions.
        class_names: List of class name strings.
        output_path: Path to save the plot.
        title: Plot title.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap="Blues", aspect="auto")

    # Labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    # Annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Confusion matrix saved to {output_path}")


def plot_training_history(
    history: dict,
    output_path: str,
    title: str = "Training History",
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys: train_loss, train_acc, val_loss, val_acc, lr.
        output_path: Path to save the plot.
        title: Plot title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(epochs, history["lr"], "g-")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Training history saved to {output_path}")
