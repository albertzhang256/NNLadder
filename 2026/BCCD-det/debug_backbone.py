"""
Interactive Debugging Script: ResNet-18 Feature Extraction on BCCD Data.

This script demonstrates and verifies:
1. Loading real blood smear images & annotations via BCCD DataLoaders (train/val/test).
2. Initializing ResNet-18 with pretrained ImageNet weights as a feature extractor.
3. Running a forward computation step through the backbone.
4. Inspecting output feature shapes, numerical statistics, and receptive field grid mapping.

Run with:
    python debug_backbone.py
Or debug interactively in VS Code by setting breakpoints and pressing F5.
"""

import argparse
import time
import torch

from dataset import CLASS_NAMES, get_dataloaders
from backbone import build_resnet18_backbone


def run_feature_extraction_debug(
    data_dir: str = "/home/albert/data/bccd",
    split: str = "train",
    batch_size: int = 4,
    img_size: int = 448,
    freeze_stages: int = 0,
    device_str: str = "auto",
):
    print("=" * 70)
    print("Step 1: Selecting Computation Device")
    print("=" * 70)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Active Device: {device} (CUDA available: {torch.cuda.is_available()})")
    if device.type == "cuda":
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 70)
    print(f"Step 2: Loading BCCD Dataset ({split} split, batch size = {batch_size})")
    print("=" * 70)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=0,  # Single-process for seamless interactive debugging
    )

    split_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    loader = split_loaders.get(split, train_loader)
    print(f"Total batches in {split} loader: {len(loader)} ({len(loader.dataset)} images)")

    # Fetch first batch of real images and ground-truth targets
    # >>> PLACE BREAKPOINT HERE TO INSPECT RAW BATCH DATA <<<
    images, targets = next(iter(loader))

    print(f"\nBatch Image Tensor:")
    print(f"  Shape:       {images.shape} (Batch, Channels, Height, Width)")
    print(f"  Dtype:       {images.dtype}")
    print(f"  Value Range: [{images.min().item():.3f}, {images.max().item():.3f}]")

    print("\nBatch Target Annotations:")
    for idx, tgt in enumerate(targets):
        labels = [CLASS_NAMES[l.item()] for l in tgt["labels"]]
        counts = {name: labels.count(name) for name in set(labels)}
        print(
            f"  Sample {idx}: {tgt['image_name']} | "
            f"Original Size: {tgt['orig_size'].tolist()} | "
            f"Boxes: {len(tgt['boxes'])} -> {counts}"
        )

    print("\n" + "=" * 70)
    print("Step 3: Initializing ResNet-18 Backbone (Pretrained on ImageNet)")
    print("=" * 70)
    backbone = build_resnet18_backbone(pretrained=True, freeze_stages=freeze_stages)
    backbone = backbone.to(device)
    backbone.eval()  # Set to evaluation mode for deterministic debugging

    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,} (~{total_params * 4 / 1024**2:.2f} MB)")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Downsampling Factor:  {backbone.downsample_ratio}x")
    print(f"Feature Channels:     {backbone.out_channels}")

    print("\n" + "=" * 70)
    print("Step 4: Executing Forward Pass for Feature Extraction")
    print("=" * 70)
    # Move images to the computation device
    images = images.to(device)

    # >>> PLACE BREAKPOINT HERE TO STEP INTO FORWARD COMPUTATION <<<
    start_time = time.perf_counter()
    with torch.no_grad():
        features = backbone(images)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Forward pass completed in {elapsed_ms:.2f} ms")
    print(f"\nExtracted Feature Map Tensor:")
    print(f"  Shape:       {features.shape} (Batch, Channels, Grid_H, Grid_W)")
    print(f"  Dtype:       {features.dtype}")
    print(f"  Mean:        {features.mean().item():.4f}")
    print(f"  Std:         {features.std().item():.4f}")
    print(f"  Min:         {features.min().item():.4f}")
    print(f"  Max:         {features.max().item():.4f}")

    # Spatial grid interpretation
    b, c, grid_h, grid_w = features.shape
    stride = img_size // grid_h
    print("\n" + "=" * 70)
    print("Step 5: Spatial Detection Grid Mapping")
    print("=" * 70)
    print(f"Input image resolution:       {img_size} x {img_size}")
    print(f"Output feature map grid:      {grid_h} x {grid_w} (S = {grid_h})")
    print(f"Each grid cell receptive field: {stride} x {stride} pixels on input image")
    print(f"Each spatial grid cell contains: {c} feature channels for detection head")
    print("=" * 70)
    print("Feature extraction forward pass verified successfully!")
    print("=" * 70)

    return images, targets, features, backbone


def parse_args():
    parser = argparse.ArgumentParser(description="Debug ResNet-18 Feature Extraction on BCCD")
    parser.add_argument("--data-dir", type=str, default="/home/albert/data/bccd", help="Path to BCCD data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--img-size", type=int, default=448, help="Square image size (e.g. 448)")
    parser.add_argument("--freeze-stages", type=int, default=0, help="Number of stages to freeze (0-4)")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_extraction_debug(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        img_size=args.img_size,
        freeze_stages=args.freeze_stages,
        device_str=args.device,
    )

