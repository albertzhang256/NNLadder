"""
BCCD Dataset Loader for Blood Cell Count and Detection.

This module loads microscopic blood smear images and their Supervisely JSON annotations
for train, val, and test splits from /home/albert/data/bccd.

Supported classes:
- RBC (Red Blood Cells) -> index 0
- WBC (White Blood Cells) -> index 1
- Platelets -> index 2
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

# Standard BCCD class mapping
CLASS_NAMES = ["RBC", "WBC", "Platelets"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Default ImageNet normalization constants used by pretrained ResNet backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transform(img_size: int = 448) -> T.Compose:
    """
    Returns the standard image transformation pipeline:
    1. Resize image to (img_size, img_size)
    2. Convert PIL Image to PyTorch Tensor in range [0.0, 1.0]
    3. Normalize with ImageNet mean and standard deviation
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BCCDDataset(Dataset):
    """
    PyTorch Dataset for BCCD blood cell detection.
    
    Args:
        data_dir: Path to dataset root containing train/, val/, test/ folders.
        split: 'train', 'val', or 'test'.
        img_size: Target square image dimension (e.g., 448 or 416).
        transform: Optional custom torchvision transform. If None, uses default.
    """

    def __init__(
        self,
        data_dir: str = "/home/albert/data/bccd",
        split: str = "train",
        img_size: int = 448,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform if transform is not None else get_default_transform(img_size)

        self.img_dir = self.data_dir / split / "img"
        self.ann_dir = self.data_dir / split / "ann"

        if not self.img_dir.is_dir() or not self.ann_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory '{split}' not found at {self.data_dir}. "
                f"Expected {self.img_dir} and {self.ann_dir}."
            )

        # Collect and pair image files with their matching JSON annotations
        self.image_files = sorted(list(self.img_dir.glob("*.jpeg")) + list(self.img_dir.glob("*.jpg")))
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Loads the image and bounding box annotations for the given index.
        
        Returns:
            image_tensor: Normalized FloatTensor of shape (3, img_size, img_size).
            target: Dictionary containing:
                - 'boxes': [N, 4] normalized [xmin, ymin, xmax, ymax] in range [0, 1].
                - 'boxes_xywh': [N, 4] normalized [cx, cy, w, h] in range [0, 1].
                - 'labels': [N] LongTensor containing class indices (0, 1, 2).
                - 'orig_size': [2] LongTensor with original [height, width].
                - 'image_name': String filename.
        """
        img_path = self.image_files[idx]
        ann_path = self.ann_dir / f"{img_path.name}.json"

        # 1. Load image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # 2. Parse JSON annotation
        boxes_xyxy_norm = []
        boxes_xywh_norm = []
        labels = []

        if ann_path.is_file():
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            for obj in ann_data.get("objects", []):
                class_title = obj.get("classTitle")
                if class_title not in CLASS_TO_IDX:
                    continue

                points = obj.get("points", {}).get("exterior", [])
                if len(points) != 2:
                    continue

                (xmin, ymin), (xmax, ymax) = points

                # Filter out degenerate or zero-area boxes
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Clamp box coordinates to original image bounds
                xmin = max(0.0, min(float(xmin), float(orig_w)))
                ymin = max(0.0, min(float(ymin), float(orig_h)))
                xmax = max(0.0, min(float(xmax), float(orig_w)))
                ymax = max(0.0, min(float(ymax), float(orig_h)))

                # Calculate normalized [xmin, ymin, xmax, ymax] in [0, 1]
                x1_norm = xmin / orig_w
                y1_norm = ymin / orig_h
                x2_norm = xmax / orig_w
                y2_norm = ymax / orig_h

                # Calculate normalized YOLO format [cx, cy, w, h] in [0, 1]
                cx_norm = (x1_norm + x2_norm) / 2.0
                cy_norm = (y1_norm + y2_norm) / 2.0
                w_norm = x2_norm - x1_norm
                h_norm = y2_norm - y1_norm

                boxes_xyxy_norm.append([x1_norm, y1_norm, x2_norm, y2_norm])
                boxes_xywh_norm.append([cx_norm, cy_norm, w_norm, h_norm])
                labels.append(CLASS_TO_IDX[class_title])

        # 3. Apply transformations to image
        img_tensor = self.transform(img)

        # 4. Convert targets to tensors (handle edge case of zero objects)
        if len(boxes_xyxy_norm) > 0:
            boxes_tensor = torch.tensor(boxes_xyxy_norm, dtype=torch.float32)
            boxes_xywh_tensor = torch.tensor(boxes_xywh_norm, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            boxes_xywh_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "boxes_xywh": boxes_xywh_tensor,
            "labels": labels_tensor,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "image_name": img_path.name,
        }

        return img_tensor, target


def detection_collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for object detection DataLoader.
    
    Stacks images into a single (B, 3, H, W) tensor, while keeping the
    variable-length target dictionaries in a Python list of length B.
    """
    images, targets = zip(*batch)
    stacked_images = torch.stack(images, dim=0)
    return stacked_images, list(targets)


def get_dataloaders(
    data_dir: str = "/home/albert/data/bccd",
    batch_size: int = 8,
    img_size: int = 448,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds training, validation, and test DataLoaders for BCCD.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = BCCDDataset(data_dir=data_dir, split="train", img_size=img_size)
    val_dataset = BCCDDataset(data_dir=data_dir, split="val", img_size=img_size)
    test_dataset = BCCDDataset(data_dir=data_dir, split="test", img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=== Testing BCCD Dataset and DataLoaders ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="/home/albert/data/bccd",
        batch_size=4,
        img_size=448,
        num_workers=0,  # 0 for single-thread debug inspection
    )

    print(f"Train batches: {len(train_loader)} (total images: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (total images: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (total images: {len(test_loader.dataset)})")

    # Fetch one sample batch from training split
    sample_images, sample_targets = next(iter(train_loader))
    print(f"\nSample batch image tensor shape: {sample_images.shape}")
    print(f"Image tensor min: {sample_images.min():.3f}, max: {sample_images.max():.3f}")

    print("\nTargets in sample batch:")
    for i, tgt in enumerate(sample_targets):
        num_boxes = tgt["boxes"].shape[0]
        label_names = [CLASS_NAMES[lbl.item()] for lbl in tgt["labels"]]
        label_summary = {name: label_names.count(name) for name in set(label_names)}
        print(
            f"  Image {i} ({tgt['image_name']}): {num_boxes} boxes, "
            f"classes: {label_summary}, orig size: {tgt['orig_size'].tolist()}"
        )

