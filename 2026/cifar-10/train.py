import argparse
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model import BasicCNN


CIFAR10_CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Download (if necessary) and load CIFAR-10 training and test datasets.
    Stores data in the specified directory.
    """
    expanded_dir = os.path.expanduser(data_dir)
    os.makedirs(expanded_dir, exist_ok=True)
    print(f"[Data] Using data directory: {os.path.abspath(expanded_dir)}")

    # Use fast, reliable mirror for downloading CIFAR-10
    torchvision.datasets.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

    # CIFAR-10 dataset statistics
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    # Standard data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # Normalization only for test evaluation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    print("[Data] Downloading / loading CIFAR-10 training set...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=expanded_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    print("[Data] Downloading / loading CIFAR-10 test set...")
    test_dataset = torchvision.datasets.CIFAR10(
        root=expanded_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_per_class: bool = False
) -> Tuple[float, float]:
    """Evaluate the model on a test/validation dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0.0] * 10
    class_total = [0.0] * 10

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            c = predicted.eq(targets)
            correct += c.sum().item()

            if compute_per_class:
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    if compute_per_class:
        print("\n" + "=" * 45)
        print("Per-Class Test Accuracy:")
        print("-" * 45)
        for i in range(10):
            acc = 100.0 * class_correct[i] / max(class_total[i], 1)
            print(f"{CIFAR10_CLASSES[i]:>10}: {acc:5.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        print("=" * 45 + "\n")

    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train a Basic CNN on CIFAR-10")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/data/cifar-10",
        help="Path to store CIFAR-10 dataset (default: ~/data/cifar-10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Input batch size for training and testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to train (default: 15)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: 1e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="cifar10_basic_cnn.pth",
        help="File path to save the best model weights",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers (default: 2)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA training even if GPU is available",
    )

    args = parser.parse_args()

    # Seed setting
    set_seed(args.seed)

    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[Device] Using device: {device}")
    if use_cuda:
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build model, criterion, optimizer, and scheduler
    model = BasicCNN(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] BasicCNN initialized with {total_params:,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    best_test_acc = 0.0
    start_time = time.time()

    print("\n" + "=" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Test Loss':^10} | {'Test Acc':^9} | {'Time':^7}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        print(
            f"{epoch:^7d} | "
            f"{train_loss:^11.4f} | "
            f"{train_acc:^9.2f}% | "
            f"{test_loss:^10.4f} | "
            f"{test_acc:^8.2f}% | "
            f"{epoch_duration:^6.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
            }, args.save_path)

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (Saved to {args.save_path})")

    # Load best model for per-class evaluation
    if os.path.exists(args.save_path):
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("\nEvaluating Best Saved Model on Test Set:")
        evaluate(model, test_loader, criterion, device, compute_per_class=True)


if __name__ == "__main__":
    main()
