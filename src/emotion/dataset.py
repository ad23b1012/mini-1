"""
Dataset loaders for Facial Expression Recognition.

Supports:
- AffectNet (primary — 8 classes with 80/10/10 stratified split)
- FER2013 (legacy — from directory structure: FER/train, FER/test)
- RAF-DB (secondary — from extracted image directory)

Images are resized to 224x224 RGB for model input. Optionally supports loading pre-extracted 
MediaPipe landmarks for explicit structural representation.
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


# ============================================================================
# Label mappings
# ============================================================================

# New 8-class dataset (alphabetical order matching folder names)
AFFECTNET_LABELS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Legacy FER2013 Label mapping (0-6)
FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
RAFDB_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


# ============================================================================
# AffectNet Dataset (Primary — 8 classes, 80/10/10 split)
# ============================================================================

class AffectNetDataset(Dataset):
    """
    AffectNet-style Dataset Loader (8-class, flat directory structure).
    
    Loads images from a flat directory of class folders and creates a reproducible
    80/10/10 stratified train/val/test split using sklearn.
    
    Directory structure expected:
        data_dir/
        ├── anger/
        ├── contempt/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
        landmarks_file: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Args:
            data_dir: Path to dataset root (containing emotion class folders).
            split: "train", "val", or "test".
            image_size: Target image size (default 224 for model input).
            transform: Custom transform pipeline.
            augment: Whether to apply data augmentation (only for training).
            landmarks_file: Path to pre-extracted `.json` landmarks dictionary.
            random_state: Random seed for reproducible splitting.
        """
        self.split = split
        self.image_size = image_size
        self.data_dir = data_dir

        # Map class names to indices (alphabetical)
        self.class_to_idx = {name: i for i, name in enumerate(AFFECTNET_LABELS)}
        self.num_classes = len(AFFECTNET_LABELS)

        # Collect all samples
        all_samples = []
        for class_name in AFFECTNET_LABELS:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            if os.path.isdir(class_dir):
                for fname in sorted(os.listdir(class_dir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        all_samples.append((os.path.join(class_dir, fname), class_idx))

        if len(all_samples) == 0:
            raise FileNotFoundError(
                f"No images found in {data_dir}. Expected subfolders: {AFFECTNET_LABELS}"
            )

        # Stratified 80/10/10 split
        all_paths = [s[0] for s in all_samples]
        all_labels = [s[1] for s in all_samples]

        # First split: 80% train, 20% temp
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_paths, all_labels,
            test_size=0.20,
            stratify=all_labels,
            random_state=random_state,
        )
        # Second split: 50/50 of the 20% temp → 10% val, 10% test
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=0.50,
            stratify=temp_labels,
            random_state=random_state,
        )

        # Select the requested split
        if split == "train":
            self.samples = list(zip(train_paths, train_labels))
        elif split == "val":
            self.samples = list(zip(val_paths, val_labels))
        elif split == "test":
            self.samples = list(zip(test_paths, test_labels))
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

        # Build transforms
        if transform is not None:
            self.transform = transform
        elif augment and split == "train":
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_eval_transform()

        # Load optional landmarks
        self.landmarks_dict = {}
        if landmarks_file and os.path.exists(landmarks_file):
            print(f"[Dataset] Loading landmarks from {landmarks_file}")
            with open(landmarks_file, "r") as f:
                self.landmarks_dict = json.load(f)

        # Print split stats
        label_counts = np.bincount([s[1] for s in self.samples], minlength=self.num_classes)
        print(f"[AffectNet] Loaded {len(self.samples)} images for split='{split}'")
        for i, name in enumerate(AFFECTNET_LABELS):
            print(f"  {name}: {label_counts[i]}")

    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for balanced Focal Loss."""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        # Add epsilon to prevent div by zero for absent classes
        weights = 1.0 / (counts + 1e-6)
        # Normalize weights
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def _get_train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.13)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_eval_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get base filename for landmark lookup
        fname = os.path.basename(path)
        if fname in self.landmarks_dict:
            lm_array = np.array(self.landmarks_dict[fname], dtype=np.float32)
            landmarks = torch.from_numpy(lm_array)
        else:
            # Fallback to zeros (indicating no landmark detected)
            landmarks = torch.zeros((468, 2), dtype=torch.float32)

        return image, label, landmarks


# ============================================================================
# FER2013 Dataset (Legacy)
# ============================================================================

class MultiModalFER2013Dataset(Dataset):
    """
    FER2013 Dataset Loader (ImageFolder structure).
    
    Loads images from a typical directory structure (class folders).
    Optionally loads MediaPipe landmarks if a pre-extracted JSON/NPY file is specified.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
        landmarks_file: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Path to FER root (containing train/test dirs).
            split: "train" or "test".
            image_size: Target image size (default 224 for model input).
            transform: Custom transform pipeline.
            augment: Whether to apply data augmentation (only for training).
            landmarks_file: Path to pre-extracted `.json` landmarks dictionary.
        """
        self.split = split
        self.image_size = image_size
        self.data_dir = os.path.join(data_dir, split)
        
        # We manually map classes to match original FER2013 integer mapping
        self.class_to_idx = {name: i for i, name in enumerate(FER2013_LABELS)}
        
        # Load all image paths
        self.samples = []
        for class_name in FER2013_LABELS:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, fname), class_idx))

        # Build transforms
        if transform is not None:
            self.transform = transform
        elif augment and split == "train":
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_eval_transform()

        # Load optional landmarks
        self.landmarks_dict = {}
        if landmarks_file and os.path.exists(landmarks_file):
            print(f"[Dataset] Loading landmarks from {landmarks_file}")
            with open(landmarks_file, "r") as f:
                self.landmarks_dict = json.load(f)

        print(f"[FER2013] Loaded {len(self.samples)} images for split='{split}'")

    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for balanced Focal Loss."""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=len(FER2013_LABELS))
        # Add epsilon to prevent div by zero for absent classes
        weights = 1.0 / (counts + 1e-6)
        # Normalize weights
        weights = weights / weights.sum() * len(FER2013_LABELS)
        return torch.tensor(weights, dtype=torch.float32)

    def _get_train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.13)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_eval_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Get base filename for landmark lookup
        fname = os.path.basename(path)
        if fname in self.landmarks_dict:
            lm_array = np.array(self.landmarks_dict[fname], dtype=np.float32)
            landmarks = torch.from_numpy(lm_array)
        else:
            # Fallback to zeros (indicating no landmark detected)
            landmarks = torch.zeros((468, 2), dtype=torch.float32)

        return image, label, landmarks


# ============================================================================
# RAF-DB Dataset (Secondary)
# ============================================================================

class RAFDBDataset(Dataset):
    """
    RAF-DB Dataset Loader.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.split = split

        label_file = os.path.join(data_dir, "basic", "EmoLabel", "list_pathdatalabel.txt")
        self.image_paths = []
        self.labels = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                img_name = parts[0]
                label = int(parts[1]) - 1  # 1-indexed to 0-indexed

                if split == "train" and img_name.startswith("train"):
                    img_path = os.path.join(data_dir, "basic", "Image", "aligned", img_name.replace(".jpg", "_aligned.jpg"))
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                elif split == "test" and img_name.startswith("test"):
                    img_path = os.path.join(data_dir, "basic", "Image", "aligned", img_name.replace(".jpg", "_aligned.jpg"))
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)

        if transform is not None:
            self.transform = transform
        elif augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        print(f"[RAF-DB] Loaded {len(self)} images for split='{split}'")

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=7)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * 7
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # Dummy landmarks for RAF-DB until extraction is run
        landmarks = torch.zeros((468, 2), dtype=torch.float32)
        return image, label, landmarks


# ============================================================================
# Factory function
# ============================================================================

def get_num_classes(dataset_name: str) -> int:
    """Return the number of classes for a given dataset."""
    if dataset_name.lower() == "affectnet":
        return len(AFFECTNET_LABELS)
    elif dataset_name.lower() == "fer2013":
        return len(FER2013_LABELS)
    elif dataset_name.lower() == "rafdb":
        return len(RAFDB_LABELS)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_labels(dataset_name: str) -> List[str]:
    """Return the label list for a given dataset."""
    if dataset_name.lower() == "affectnet":
        return AFFECTNET_LABELS
    elif dataset_name.lower() == "fer2013":
        return FER2013_LABELS
    elif dataset_name.lower() == "rafdb":
        return RAFDB_LABELS
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloader(
    dataset_name: str,
    data_path: str,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    augment: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    landmarks_file: Optional[str] = None,
) -> Tuple[DataLoader, torch.Tensor]:
    """
    Returns (DataLoader, class_weights).
    """
    if dataset_name.lower() == "affectnet":
        dataset = AffectNetDataset(
            data_dir=data_path,
            split=split,
            image_size=image_size,
            augment=augment,
            landmarks_file=landmarks_file,
        )
    elif dataset_name.lower() == "fer2013":
        dataset = MultiModalFER2013Dataset(
            data_dir=data_path,
            split=split,
            image_size=image_size,
            augment=augment,
            landmarks_file=landmarks_file,
        )
    elif dataset_name.lower() == "rafdb":
        dataset = RAFDBDataset(
            data_dir=data_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    shuffle = (split == "train")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return loader, dataset.get_class_weights()
