"""
Overarching Training, Validation, and Testing Script for BCCD YOLO Detector.

This script manages:
1. YOLOLoss: Loss function combining CIoU box regression, dynamic IoU-based
   box responsibility matching, IoU-aware objectness confidence (BCEWithLogits),
   background confidence (no-object BCEWithLogits), and cell classification.
2. train_one_epoch: Training step with gradient clipping and loss component logging.
3. validate: Evaluation step on the validation set.
4. test_model: Final evaluation on the test set with NMS predictions.
5. Model checkpointing: Saves 'best_model.pth' and 'last_model.pth'.

Usage:
    # Two-stage training with differential LRs & selective early freezing (default)
    python train.py --epochs 20 --batch-size 8

    # Full end-to-end unfreezing in Stage 2
    python train.py --epochs 20 --warmup-epochs 5 --freeze-stages 0

    # Manual Stage 2 resume from Stage 1 checkpoint
    python train.py --resume checkpoints/last_model.pth --freeze-stages 0 --epochs 15

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
import torchvision.ops as ops
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, get_dataloaders
from yolo_head import BCCDYOLO, box_iou, build_bccd_yolo


# ==============================================================================
# 1. YOLO Loss Function
# ==============================================================================

class YOLOLoss(nn.Module):
    """
    Computes YOLO detection loss against ground-truth bounding boxes.

    Loss components:
    1. Coordinate loss (Complete IoU / CIoU loss on IoU-matched boxes)
    2. Objectness confidence loss (IoU-aware BCEWithLogits for responsible box)
    3. No-object confidence loss (BCEWithLogits for competing & background boxes)
    4. Classification loss (CrossEntropy for grid cells containing objects)

    Args:
        num_classes: Number of object classes (3 for BCCD).
        num_boxes: Number of predicted boxes per grid cell (default: 2).
        lambda_coord: Weight for bounding box CIoU loss (default: 5.0).
        lambda_noobj: Weight for background / no-object confidence loss (default: 5.0).
        lambda_obj: Weight for foreground objectness confidence loss (default: 1.0).
        lambda_class: Weight for classification loss (default: 1.0).
        iou_target_weight: Blend weight alpha in [0, 1] for IoU-aware objectness
            targets: target_conf = (1 - alpha) + alpha * IoU (default: 0.5).
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_boxes: int = 2,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_class: float = 1.0,
        iou_target_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.iou_target_weight = iou_target_weight

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
        # A. Decode Model Predictions to Normalized [x1, y1, x2, y2] Boxes
        # ------------------------------------------------------------------
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=predictions.dtype),
            torch.arange(grid_w, device=device, dtype=predictions.dtype),
            indexing="ij",
        )
        grid_x = grid_x.unsqueeze(0)  # (1, S, S)
        grid_y = grid_y.unsqueeze(0)  # (1, S, S)

        pred_boxes_list = []
        pred_confs_list = []

        for b in range(self.num_boxes):
            start = b * 5
            tx = torch.sigmoid(predictions[..., start + 0])
            ty = torch.sigmoid(predictions[..., start + 1])
            tw = torch.sigmoid(predictions[..., start + 2])
            th = torch.sigmoid(predictions[..., start + 3])
            tconf_logit = predictions[..., start + 4]  # Keep raw logit for BCEWithLogits

            # Decode center relative to grid cell and convert [cx, cy, w, h] -> [x1, y1, x2, y2]
            cx = (tx + grid_x) / float(grid_w)
            cy = (ty + grid_y) / float(grid_h)
            box_xyxy = torch.stack(
                [cx - 0.5 * tw, cy - 0.5 * th, cx + 0.5 * tw, cy + 0.5 * th],
                dim=-1,
            )

            pred_boxes_list.append(box_xyxy)
            pred_confs_list.append(tconf_logit)

        pred_boxes_xyxy = torch.stack(pred_boxes_list, dim=3)  # (B, S, S, num_boxes, 4)
        pred_confs = torch.stack(pred_confs_list, dim=3)       # (B, S, S, num_boxes) logits
        pred_class_logits = predictions[..., self.num_boxes * 5 :]  # (B, S, S, num_classes)

        # ------------------------------------------------------------------
        # B. Dynamic IoU-Based Box Matching & Target Construction
        # ------------------------------------------------------------------
        target_boxes_xyxy = torch.zeros(batch_size, S, S, self.num_boxes, 4, device=device)
        target_conf = torch.zeros(batch_size, S, S, self.num_boxes, device=device)
        obj_mask = torch.zeros(batch_size, S, S, self.num_boxes, dtype=torch.bool, device=device)

        target_class = torch.zeros(batch_size, S, S, dtype=torch.int64, device=device)
        class_mask = torch.zeros(batch_size, S, S, dtype=torch.bool, device=device)

        pred_boxes_detached = pred_boxes_xyxy.detach()

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes_xywh"]  # [N, 4]: cx, cy, w, h in [0, 1]
            gt_labels = targets[b]["labels"]     # [N]

            for idx in range(len(gt_boxes)):
                cx, cy, w, h = gt_boxes[idx].tolist()
                label = int(gt_labels[idx].item())

                # Determine which grid cell (gy, gx) contains the object center
                gx = min(int(cx * grid_w), grid_w - 1)
                gy = min(int(cy * grid_h), grid_h - 1)

                target_class[b, gy, gx] = label
                class_mask[b, gy, gx] = True

                # Match to the unassigned predictor in (gy, gx) with highest IoU
                available = ~obj_mask[b, gy, gx]
                if not available.any():
                    continue

                gt_xyxy = torch.tensor(
                    [[cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]],
                    dtype=predictions.dtype,
                    device=device,
                )
                ious = box_iou(pred_boxes_detached[b, gy, gx], gt_xyxy).squeeze(1)
                best_box_idx = int(ious.masked_fill(~available, -1.0).argmax().item())
                best_iou = float(ious[best_box_idx].clamp(0.0, 1.0).item())

                obj_mask[b, gy, gx, best_box_idx] = True
                target_boxes_xyxy[b, gy, gx, best_box_idx] = gt_xyxy[0]
                target_conf[b, gy, gx, best_box_idx] = (
                    (1.0 - self.iou_target_weight) + self.iou_target_weight * best_iou
                )

        # ------------------------------------------------------------------
        # C. Compute Individual Loss Terms
        # ------------------------------------------------------------------
        num_objects = obj_mask.sum().clamp(min=1).item()

        # 1. Coordinate Loss: Complete IoU (CIoU) on responsible boxes
        if obj_mask.any():
            loss_coord = ops.complete_box_iou_loss(
                pred_boxes_xyxy[obj_mask],
                target_boxes_xyxy[obj_mask],
                reduction="mean",
            )
        else:
            loss_coord = torch.tensor(0.0, device=device)

        # 2. Foreground Objectness Confidence Loss (IoU-aware target)
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
    conf_threshold: float = 0.50,
    iou_threshold: float = 0.30,
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

def build_optimizer(
    model: BCCDYOLO,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    Creates an AdamW optimizer with differential learning rates:
    pretrained backbone parameters use backbone_lr (lower), while the
    randomly initialized YOLO head uses head_lr (higher).
    """
    param_groups = []
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})

    if not param_groups:
        raise RuntimeError("No trainable parameters found when building optimizer.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def format_lr_status(optimizer: torch.optim.Optimizer) -> str:
    """Formats per-group learning rates for epoch logging."""
    lrs = {g.get("name", f"group{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)}
    head_str = f"Head LR: {lrs['head']:.2e}" if "head" in lrs else "Head: frozen"
    bb_str = f"Backbone LR: {lrs['backbone']:.2e}" if "backbone" in lrs else "Backbone: frozen"
    return f"{head_str} | {bb_str}"


def log_trainable_params(model: BCCDYOLO, stage_label: str) -> None:
    """Logs trainable parameter counts across backbone and detection head."""
    bb_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    bb_total = sum(p.numel() for p in model.backbone.parameters())
    hd_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    hd_total = sum(p.numel() for p in model.head.parameters())
    print(
        f"[{stage_label}] Trainable Params -> "
        f"Backbone: {bb_trainable:,}/{bb_total:,} (freeze_stages={model.backbone.freeze_stages}) | "
        f"Head: {hd_trainable:,}/{hd_total:,}"
    )


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    best_val_loss: float,
) -> None:
    """Saves model weights, optimizer state, and validation tracking metrics."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def run_pipeline(
    data_dir: str = "/home/albert/data/bccd",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    img_size: int = 448,
    freeze_stages: int = 2,
    warmup_epochs: Optional[int] = None,
    lambda_noobj: float = 5.0,
    augment: bool = True,
    save_dir: str = "./checkpoints",
    device_str: str = "auto",
    test_only: bool = False,
    weights_path: Optional[str] = None,
    resume_path: Optional[str] = None,
):
    # Device configuration
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if warmup_epochs is None:
        warmup_epochs = 0 if resume_path else 5

    effective_warmup = max(0, min(warmup_epochs, epochs))
    use_two_stage = (effective_warmup > 0) and (effective_warmup < epochs) and (freeze_stages < 4)
    initial_freeze_stages = 4 if effective_warmup > 0 else freeze_stages

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
    print(f"Head LR:          {lr}")
    print(f"Backbone LR:      {backbone_lr}")
    print(f"Freeze Stages:    {freeze_stages} (Warmup epochs @ freeze=4: {effective_warmup})")
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
        freeze_stages=initial_freeze_stages,
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

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    # Optional resume from prior checkpoint (e.g. manual Stage 2 fine-tuning)
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found at: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        print(
            f"Resumed weights from: {resume_path} "
            f"(ckpt epoch: {checkpoint.get('epoch', '?')}, best_val_loss: {best_val_loss:.4f})"
        )

    # Initial Optimizer & LR Scheduler (Stage 1 Warmup or Single-Stage Fine-Tuning)
    optimizer = build_optimizer(model, head_lr=lr, backbone_lr=backbone_lr, weight_decay=weight_decay)
    if use_two_stage:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=effective_warmup, eta_min=max(lr * 0.1, 1e-6)
        )
        initial_stage_label = f"Stage 1 Warmup (Epochs 1-{effective_warmup})"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        initial_stage_label = (
            "Stage 1 Only (Head Warmup)" if initial_freeze_stages == 4 else "Fine-Tuning"
        )

    print("\n" + "=" * 70)
    print("=== Commencing Model Training ===")
    print("=" * 70)
    log_trainable_params(model, initial_stage_label)

    for epoch in range(1, epochs + 1):
        # Transition from Stage 1 (frozen backbone warmup) to Stage 2 (unfrozen differential LR)
        if use_two_stage and epoch == effective_warmup + 1:
            stage2_epochs = epochs - effective_warmup
            print("\n" + "-" * 70)
            print(
                f"=== Stage 2 Fine-Tuning (Epochs {epoch}-{epochs}): "
                f"Unfreezing backbone to freeze_stages={freeze_stages} ==="
            )
            print("-" * 70)
            model.set_freeze_stages(freeze_stages)
            log_trainable_params(model, f"Stage 2 Fine-Tuning (Epochs {epoch}-{epochs})")
            optimizer = build_optimizer(
                model, head_lr=lr, backbone_lr=backbone_lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage2_epochs, eta_min=1e-6
            )

        epoch_start = time.perf_counter()
        print(f"\nEpoch [{epoch:2d}/{epochs:2d}] ({format_lr_status(optimizer)})")

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
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_path = os.path.join(save_dir, "best_model.pth")
            save_checkpoint(best_path, epoch, model, optimizer, val_res["loss"], best_val_loss)
            print(f"  *** Best model saved to {best_path} (Val Loss: {best_val_loss:.4f}) ***")

        # Always save latest checkpoint
        last_path = os.path.join(save_dir, "last_model.pth")
        save_checkpoint(last_path, epoch, model, optimizer, val_res["loss"], best_val_loss)

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
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for detection head (default: 1e-3)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Initial learning rate for pretrained backbone (default: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--img-size", type=int, default=448, help="Square input image dimension")
    parser.add_argument("--freeze-stages", type=int, default=2, help="Backbone stages kept frozen during fine-tuning, 0 to 4 (default: 2)")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Stage 1 head-only warmup epochs with freeze_stages=4 (default: 5, or 0 with --resume)")
    parser.add_argument("--lambda-noobj", type=float, default=5.0, help="Weight for background / no-object loss (default: 5.0)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation during training")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run evaluation on test set")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file for testing")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        freeze_stages=args.freeze_stages,
        warmup_epochs=args.warmup_epochs,
        lambda_noobj=args.lambda_noobj,
        augment=not args.no_augment,
        save_dir=args.save_dir,
        device_str=args.device,
        test_only=args.test_only,
        weights_path=args.weights,
        resume_path=args.resume,
    )

