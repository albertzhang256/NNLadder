"""
Overarching Training, Validation, and Testing Script for BCCD YOLO Detector.

This script manages:
1. YOLOLoss: Loss function combining coordinate regression (MSE), objectness
   confidence (BCEWithLogits), background confidence (no-object BCEWithLogits),
   and cell classification (CrossEntropy).
2. train_one_epoch: Training step with gradient clipping and loss component logging.
3. validate: Evaluation step on the validation set.
4. test_model: Final evaluation on the test set with NMS predictions.
5. Model checkpointing: Saves 'best_model.pth' and 'last_model.pth'.

Usage:
    # Full training for 20 epochs
    python train.py --epochs 20 --batch-size 8

    # Quick dry run (1 epoch)
    python train.py --epochs 1 --batch-size 4

    # Run test evaluation only using existing weights
    python train.py --test-only --weights checkpoints/best_model.pth
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, get_dataloaders
from yolo_head import BCCDYOLO, build_bccd_yolo


# ==============================================================================
# 1. YOLO Loss Function
# ==============================================================================

class YOLOLoss(nn.Module):
    """
    Computes YOLO detection loss against ground-truth bounding boxes.

    Loss components:
    1. Coordinate loss (MSE on center offsets and square-root box dimensions)
    2. Objectness confidence loss (BCEWithLogits for responsible box)
    3. No-object confidence loss (BCEWithLogits for empty background boxes)
    4. Classification loss (CrossEntropy for grid cells containing objects)

    Args:
        num_classes: Number of object classes (3 for BCCD).
        num_boxes: Number of predicted boxes per grid cell (default: 2).
        lambda_coord: Weight for bounding box coordinate loss (default: 5.0).
        lambda_noobj: Weight for background / no-object confidence loss (default: 2.0).
        lambda_obj: Weight for foreground objectness confidence loss (default: 1.0).
        lambda_class: Weight for classification loss (default: 1.0).
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_boxes: int = 2,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 2.0,
        lambda_obj: float = 1.0,
        lambda_class: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class

    def forward(
        self,
        predictions: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates loss between raw predictions and target annotations.

        Args:
            predictions: Tensor of shape (B, S, S, num_boxes * 5 + num_classes).
            targets: List of length B with target dictionaries ('boxes_xywh', 'labels').

        Returns:
            total_loss: Scalar torch.Tensor with gradients enabled.
            loss_components: Dict with float values of each loss component.
        """
        device = predictions.device
        batch_size, grid_h, grid_w, _ = predictions.shape
        S = grid_h

        # ------------------------------------------------------------------
        # A. Build Ground-Truth Target Tensors
        # ------------------------------------------------------------------
        target_coord = torch.zeros(batch_size, S, S, self.num_boxes, 4, device=device)
        target_conf = torch.zeros(batch_size, S, S, self.num_boxes, device=device)
        obj_mask = torch.zeros(batch_size, S, S, self.num_boxes, dtype=torch.bool, device=device)

        target_class = torch.zeros(batch_size, S, S, dtype=torch.int64, device=device)
        class_mask = torch.zeros(batch_size, S, S, dtype=torch.bool, device=device)

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes_xywh"]  # [N, 4]: cx, cy, w, h in [0, 1]
            gt_labels = targets[b]["labels"]     # [N]

            for idx in range(len(gt_boxes)):
                cx, cy, w, h = gt_boxes[idx].tolist()
                label = gt_labels[idx].item()

                # Determine which grid cell (gy, gx) contains the center of the cell
                gx = min(int(cx * S), S - 1)
                gy = min(int(cy * S), S - 1)

                # Local offset within grid cell [0.0, 1.0]
                tx = cx * S - gx
                ty = cy * S - gy

                # Assign to Box 0 or Box 1 in this cell
                box_idx = 0
                if obj_mask[b, gy, gx, 0]:
                    box_idx = 1  # If Box 0 already claimed, assign to Box 1

                obj_mask[b, gy, gx, box_idx] = True
                target_conf[b, gy, gx, box_idx] = 1.0
                target_coord[b, gy, gx, box_idx] = torch.tensor(
                    [tx, ty, w, h], dtype=torch.float32, device=device
                )

                target_class[b, gy, gx] = label
                class_mask[b, gy, gx] = True

        # ------------------------------------------------------------------
        # B. Slice and Transform Model Predictions
        # ------------------------------------------------------------------
        pred_boxes = []
        pred_confs = []

        for b in range(self.num_boxes):
            start = b * 5
            tx = torch.sigmoid(predictions[..., start + 0])
            ty = torch.sigmoid(predictions[..., start + 1])
            tw = torch.sigmoid(predictions[..., start + 2])
            th = torch.sigmoid(predictions[..., start + 3])
            tconf_logit = predictions[..., start + 4]  # Keep raw logit for BCEWithLogits

            pred_boxes.append(torch.stack([tx, ty, tw, th], dim=-1))
            pred_confs.append(tconf_logit)

        pred_boxes = torch.stack(pred_boxes, dim=3)  # (B, S, S, num_boxes, 4)
        pred_confs = torch.stack(pred_confs, dim=3)  # (B, S, S, num_boxes) logits
        pred_class_logits = predictions[..., self.num_boxes * 5 :]  # (B, S, S, num_classes)

        # ------------------------------------------------------------------
        # C. Compute Individual Loss Terms
        # ------------------------------------------------------------------
        num_objects = obj_mask.sum().clamp(min=1).item()

        # 1. Coordinate Loss: MSE on (tx, ty) and square-root (w, h)
        if obj_mask.any():
            loss_xy = F.mse_loss(
                pred_boxes[..., :2][obj_mask], target_coord[..., :2][obj_mask], reduction="sum"
            )
            # Sqrt on w, h ensures smaller deviations in small boxes are penalized appropriately
            loss_wh = F.mse_loss(
                torch.sqrt(pred_boxes[..., 2:][obj_mask] + 1e-6),
                torch.sqrt(target_coord[..., 2:][obj_mask] + 1e-6),
                reduction="sum",
            )
            loss_coord = (loss_xy + loss_wh) / num_objects
        else:
            loss_coord = torch.tensor(0.0, device=device)

        # 2. Foreground Objectness Confidence Loss
        if obj_mask.any():
            loss_obj = F.binary_cross_entropy_with_logits(
                pred_confs[obj_mask], target_conf[obj_mask], reduction="sum"
            ) / num_objects
        else:
            loss_obj = torch.tensor(0.0, device=device)

        # 3. Background (No-Object) Confidence Loss
        noobj_mask = ~obj_mask
        loss_noobj = F.binary_cross_entropy_with_logits(
            pred_confs[noobj_mask], target_conf[noobj_mask], reduction="sum"
        ) / noobj_mask.sum().clamp(min=1).item()

        # 4. Classification Loss
        if class_mask.any():
            loss_class = F.cross_entropy(
                pred_class_logits[class_mask], target_class[class_mask], reduction="sum"
            ) / class_mask.sum().clamp(min=1).item()
        else:
            loss_class = torch.tensor(0.0, device=device)

        # Weighted Total Loss
        total_loss = (
            self.lambda_coord * loss_coord
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
            + self.lambda_class * loss_class
        )

        loss_dict = {
            "total": total_loss.item(),
            "coord": loss_coord.item(),
            "obj": loss_obj.item(),
            "noobj": loss_noobj.item(),
            "class": loss_class.item(),
        }

        return total_loss, loss_dict


# ==============================================================================
# 2. Training Loop for One Epoch
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: YOLOLoss,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Runs a single training epoch through the dataset."""
    model.train()
    running_loss = 0.0
    running_coord = 0.0
    running_obj = 0.0
    running_noobj = 0.0
    running_class = 0.0

    num_batches = len(dataloader)
    start_time = time.perf_counter()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Loss calculation
        loss, loss_dict = criterion(predictions, targets)

        # Backpropagation
        loss.backward()

        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Optimizer step
        optimizer.step()

        running_loss += loss_dict["total"]
        running_coord += loss_dict["coord"]
        running_obj += loss_dict["obj"]
        running_noobj += loss_dict["noobj"]
        running_class += loss_dict["class"]

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(
                f"  Epoch [{epoch:2d}] Batch [{batch_idx + 1:2d}/{num_batches:2d}] | "
                f"Loss: {loss_dict['total']:.4f} "
                f"(Coord: {loss_dict['coord']:.3f}, Obj: {loss_dict['obj']:.3f}, "
                f"NoObj: {loss_dict['noobj']:.4f}, Cls: {loss_dict['class']:.3f})"
            )

    elapsed = time.perf_counter() - start_time
    avg_metrics = {
        "loss": running_loss / num_batches,
        "coord": running_coord / num_batches,
        "obj": running_obj / num_batches,
        "noobj": running_noobj / num_batches,
        "class": running_class / num_batches,
        "time": elapsed,
    }
    return avg_metrics


# ==============================================================================
# 3. Validation Step
# ==============================================================================

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: YOLOLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Runs validation evaluation over the val split without gradient updates."""
    model.eval()
    running_loss = 0.0
    running_coord = 0.0
    running_obj = 0.0
    running_noobj = 0.0
    running_class = 0.0

    num_batches = len(dataloader)
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            predictions = model(images)
            _, loss_dict = criterion(predictions, targets)

            running_loss += loss_dict["total"]
            running_coord += loss_dict["coord"]
            running_obj += loss_dict["obj"]
            running_noobj += loss_dict["noobj"]
            running_class += loss_dict["class"]

    return {
        "loss": running_loss / num_batches,
        "coord": running_coord / num_batches,
        "obj": running_obj / num_batches,
        "noobj": running_noobj / num_batches,
        "class": running_class / num_batches,
    }


# ==============================================================================
# 4. Testing & Inference Evaluation
# ==============================================================================

def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: YOLOLoss,
    device: torch.device,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, float]:
    """
    Evaluates the model on the held-out test split, reporting loss
    and running NMS detection inference to count predicted classes.
    """
    print("\n" + "=" * 70)
    print("=== Running Final Evaluation on Test Split ===")
    print("=" * 70)

    val_metrics = validate(model, dataloader, criterion, device)
    print(f"Test Loss: {val_metrics['loss']:.4f}")
    print(
        f"  Coord Loss: {val_metrics['coord']:.4f} | "
        f"Obj Loss: {val_metrics['obj']:.4f} | "
        f"NoObj Loss: {val_metrics['noobj']:.4f} | "
        f"Cls Loss: {val_metrics['class']:.4f}"
    )

    # Run NMS prediction on sample test batches
    model.eval()
    total_detected_cells = 0
    class_detections = {name: 0 for name in CLASS_NAMES}

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            batch_dets = model.predict(
                images, conf_threshold=conf_threshold, iou_threshold=iou_threshold
            )

            for det in batch_dets:
                total_detected_cells += len(det)
                if len(det) > 0:
                    for class_idx in det[:, 5].int().tolist():
                        class_detections[CLASS_NAMES[class_idx]] += 1

    print("\nTest Split NMS Detections Summary:")
    print(f"  Total Images Evaluated: {len(dataloader.dataset)}")
    print(f"  Total Cells Detected:   {total_detected_cells}")
    for name, count in class_detections.items():
        print(f"    - {name:10s}: {count:5d} detections")
    print("=" * 70)

    return val_metrics


# ==============================================================================
# 5. Overarching Training Pipeline Orchestrator
# ==============================================================================

def run_pipeline(
    data_dir: str = "/home/albert/data/bccd",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    img_size: int = 448,
    freeze_stages: int = 0,
    lambda_noobj: float = 2.0,
    augment: bool = True,
    save_dir: str = "./checkpoints",
    device_str: str = "auto",
    test_only: bool = False,
    weights_path: Optional[str] = None,
):
    # Device configuration
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("=== BCCD YOLO Detector Pipeline ===")
    print("=" * 70)
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU Model:        {torch.cuda.get_device_name(0)}")
    print(f"Dataset Root:     {data_dir}")
    print(f"Image Resolution: {img_size} x {img_size}")
    print(f"Batch Size:       {batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"Initial LR:       {lr}")
    print(f"Lambda No-Object: {lambda_noobj}")
    print(f"Augmentation:     {'Enabled (Flips, 90-deg Rotations, ColorJitter)' if augment else 'Disabled'}")

    # Build DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=2,
        augment=augment,
    )
    print(
        f"Data Loaded:      {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test images"
    )

    # Build Model
    model = build_bccd_yolo(
        num_classes=3,
        num_boxes=2,
        pretrained_backbone=True,
        freeze_stages=freeze_stages,
    ).to(device)

    criterion = YOLOLoss(num_classes=3, num_boxes=2, lambda_noobj=lambda_noobj)

    # Handle Test-Only mode
    if test_only:
        ckpt_path = weights_path or os.path.join(save_dir, "best_model.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        print(f"\nLoading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_model(model, test_loader, criterion, device)
        return

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    print("\n" + "=" * 70)
    print("=== Commencing Model Training ===")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        print(f"\nEpoch [{epoch:2d}/{epochs:2d}] (LR: {scheduler.get_last_lr()[0]:.2e})")

        # 1. Train
        train_res = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # 2. Validate
        val_res = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"--> Epoch [{epoch:2d}/{epochs:2d}] Summary ({epoch_time:.1f}s) | "
            f"Train Loss: {train_res['loss']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} "
            f"(Coord: {val_res['coord']:.3f}, Obj: {val_res['obj']:.3f}, Cls: {val_res['class']:.3f})"
        )

        # 3. Checkpoint Saving (Best Model)
        is_best = val_res["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_res["loss"],
                },
                best_path,
            )
            print(f"  *** Best model saved to {best_path} (Val Loss: {best_val_loss:.4f}) ***")

        # Always save latest checkpoint
        last_path = os.path.join(save_dir, "last_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_res["loss"],
            },
            last_path,
        )

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
    print("=" * 70)

    # 4. Final Test Set Evaluation using the Best Model
    best_path = os.path.join(save_dir, "best_model.pth")
    if os.path.isfile(best_path):
        print(f"Loading best checkpoint for test evaluation: {best_path}")
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

    test_model(model, test_loader, criterion, device)


# ==============================================================================
# 6. CLI Arguments Parser
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train, Validate, and Test BCCD YOLO Detector")
    parser.add_argument("--data-dir", type=str, default="/home/albert/data/bccd", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DataLoaders")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--img-size", type=int, default=448, help="Square input image dimension")
    parser.add_argument("--freeze-stages", type=int, default=0, help="Backbone stages to freeze (0 to 4)")
    parser.add_argument("--lambda-noobj", type=float, default=2.0, help="Weight for background / no-object loss (default: 2.0)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation during training")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run evaluation on test set")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        freeze_stages=args.freeze_stages,
        lambda_noobj=args.lambda_noobj,
        augment=not args.no_augment,
        save_dir=args.save_dir,
        device_str=args.device,
        test_only=args.test_only,
        weights_path=args.weights,
    )

