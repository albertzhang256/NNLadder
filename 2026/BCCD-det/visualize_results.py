"""
Visualize YOLO Blood Cell Detection Results from Trained Model Checkpoints.

This script loads a trained model checkpoint (e.g., best_model.pth or last_model.pth),
runs inference on images from the test, val, or train splits, and displays
a side-by-side visual comparison of Ground Truth vs. Model Predictions.

Features:
- Side-by-side comparison: Ground Truth (left) vs. YOLO Predictions (right).
- Color-coded bounding boxes matching cell types:
    - RBC: Red (#D0021B)
    - WBC: Blue (#4A90E2)
    - Platelets: Green (#7ED321)
- Displays confidence score on predicted bounding box badges (e.g., 'RBC 0.94').
- Configurable confidence threshold and NMS IoU threshold.
- Interactive GUI display (`plt.show()`) and/or saving to disk (`--save-path`).

Usage:
    # 1. Visualize first test image with best_model.pth
    python visualize_results.py

    # 2. View a random image from the test split
    python visualize_results.py --random

    # 3. View using last_model.pth
    python visualize_results.py --weights checkpoints/last_model.pth

    # 4. View a specific image with custom confidence threshold
    python visualize_results.py --image-name BloodImage_00385.jpeg --conf-threshold 0.35

    # 5. Save comparison to file without opening GUI window
    python visualize_results.py --index 2 --save-path result.png --no-show
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import CLASS_NAMES, CLASS_TO_IDX, get_default_transform
from yolo_head import BCCDYOLO, build_bccd_yolo

# Color scheme matching dataset annotations and visualize_data.py
CLASS_COLORS = {
    "RBC": "#D0021B",       # Red
    "WBC": "#4A90E2",       # Blue
    "Platelets": "#7ED321", # Green
}
DEFAULT_COLOR = "#F5A623"   # Orange fallback


def load_model_from_checkpoint(
    weights_path: str,
    device: torch.device,
) -> Tuple[BCCDYOLO, Dict]:
    """
    Loads the BCCDYOLO detector and restores weights from a saved checkpoint.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {weights_path}\n"
            f"Please run 'python train.py' first to train and generate checkpoints."
        )

    model = build_bccd_yolo(num_classes=3, num_boxes=2, pretrained_backbone=False)
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle both full checkpoint dict and bare state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "epoch": checkpoint.get("epoch", "N/A"),
            "val_loss": checkpoint.get("val_loss", None),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {"epoch": "N/A", "val_loss": None}

    model = model.to(device)
    model.eval()
    return model, metadata


def load_ground_truth(ann_path: Path) -> List[Dict]:
    """Parses Supervisely JSON format ground-truth bounding boxes."""
    if not ann_path.is_file():
        return []
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_objects = []
    for obj in data.get("objects", []):
        class_title = obj.get("classTitle", "Unknown")
        points = obj.get("points", {}).get("exterior", [])
        if len(points) == 2:
            (xmin, ymin), (xmax, ymax) = points
            if xmax > xmin and ymax > ymin:
                gt_objects.append({
                    "class": class_title,
                    "box": [xmin, ymin, xmax, ymax],
                })
    return gt_objects


def run_inference_on_image(
    model: BCCDYOLO,
    image_path: Path,
    device: torch.device,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 448,
) -> Tuple[Image.Image, List[Dict]]:
    """
    Loads an image, applies normalization, runs YOLO inference + NMS,
    and returns detected bounding boxes scaled to the original pixel resolution.
    """
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    transform = get_default_transform(img_size=img_size)
    img_tensor = transform(orig_img).unsqueeze(0).to(device)  # (1, 3, 448, 448)

    with torch.no_grad():
        # detections is a list of length 1 containing Tensor of shape (K, 6):
        # [x1, y1, x2, y2, score, class_id] in normalized [0, 1] coordinates
        detections = model.predict(
            img_tensor,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )[0]

    detected_objects = []
    if detections.numel() > 0:
        for det in detections:
            x1_norm, y1_norm, x2_norm, y2_norm, score, class_idx = det.tolist()
            class_name = CLASS_NAMES[int(class_idx)]

            # Convert normalized [0, 1] coordinates back to original image pixels
            xmin = x1_norm * orig_w
            ymin = y1_norm * orig_h
            xmax = x2_norm * orig_w
            ymax = y2_norm * orig_h

            detected_objects.append({
                "class": class_name,
                "score": score,
                "box": [xmin, ymin, xmax, ymax],
            })

    return orig_img, detected_objects


def draw_bounding_boxes(
    ax,
    objects: List[Dict],
    is_prediction: bool = False,
):
    """Draws color-coded bounding boxes and text badges onto a matplotlib axis."""
    for obj in objects:
        class_name = obj["class"]
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        xmin, ymin, xmax, ymax = obj["box"]
        width = xmax - xmin
        height = ymax - ymin

        # Draw bounding box rectangle
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Label badge text (include confidence score for predictions)
        if is_prediction:
            label_text = f"{class_name} {obj['score']:.2f}"
        else:
            label_text = f"{class_name}"

        ax.text(
            xmin,
            max(0, ymin - 3),
            label_text,
            color="white",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(
                facecolor=color,
                edgecolor="none",
                boxstyle="round,pad=0.2",
                alpha=0.85,
            ),
        )


def plot_comparison(
    img: Image.Image,
    gt_objects: List[Dict],
    pred_objects: List[Dict],
    image_name: str,
    weights_name: str,
    metadata: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
    single_panel: bool = False,
):
    """Renders visual detection results."""
    # Count occurrences of classes
    gt_counts = {name: 0 for name in CLASS_NAMES}
    for obj in gt_objects:
        gt_counts[obj["class"]] = gt_counts.get(obj["class"], 0) + 1

    pred_counts = {name: 0 for name in CLASS_NAMES}
    for obj in pred_objects:
        pred_counts[obj["class"]] = pred_counts.get(obj["class"], 0) + 1

    if single_panel:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        ax.imshow(img)
        draw_bounding_boxes(ax, pred_objects, is_prediction=True)
        ax.set_title(
            f"YOLO Predictions: {image_name}\n"
            f"Detected ({len(pred_objects)}): RBC={pred_counts['RBC']}, WBC={pred_counts['WBC']}, Platelets={pred_counts['Platelets']}",
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")
    else:
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(18, 7.5))

        # 1. Left Panel: Ground Truth
        ax_gt.imshow(img)
        draw_bounding_boxes(ax_gt, gt_objects, is_prediction=False)
        ax_gt.set_title(
            f"Ground Truth ({len(gt_objects)} cells)\n"
            f"RBC: {gt_counts['RBC']} | WBC: {gt_counts['WBC']} | Platelets: {gt_counts['Platelets']}",
            fontsize=11,
            fontweight="bold",
            color="#2C3E50",
        )
        ax_gt.axis("off")

        # 2. Right Panel: Model Predictions
        ax_pred.imshow(img)
        draw_bounding_boxes(ax_pred, pred_objects, is_prediction=True)
        ax_pred.set_title(
            f"YOLO Predictions ({len(pred_objects)} detected)\n"
            f"RBC: {pred_counts['RBC']} | WBC: {pred_counts['WBC']} | Platelets: {pred_counts['Platelets']}",
            fontsize=11,
            fontweight="bold",
            color="#2C3E50",
        )
        ax_pred.axis("off")

        # Top overarching title
        epoch_str = f"Epoch {metadata['epoch']}" if metadata["epoch"] != "N/A" else ""
        loss_str = f"Val Loss: {metadata['val_loss']:.4f}" if metadata["val_loss"] is not None else ""
        meta_info = f" ({epoch_str}, {loss_str})".replace(" (, ", " (").replace(", )", ")")
        if meta_info == " ()":
            meta_info = ""

        plt.suptitle(
            f"BCCD Blood Cell Detection | File: {image_name} | Weights: {weights_name}{meta_info}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO Detection Results from Checkpoint")
    parser.add_argument(
        "--weights",
        type=str,
        default="./checkpoints/best_model.pth",
        help="Path to trained checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/albert/data/bccd",
        help="Path to BCCD dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--index", type=int, default=0, help="Image index within the split")
    parser.add_argument("--random", action="store_true", help="Pick a random image from the split")
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="Specific image filename (e.g. BloodImage_00054.jpeg)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="NMS confidence score cutoff (default: 0.25)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="NMS IoU suppression threshold (default: 0.45)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the rendered plot image (e.g. output.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive matplotlib window",
    )
    parser.add_argument(
        "--single-panel",
        action="store_true",
        help="Show only model predictions instead of side-by-side comparison",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'cuda', 'cpu', or 'auto'",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Load trained model
    weights_path = os.path.abspath(args.weights)
    print(f"Loading checkpoint: {weights_path}")
    model, metadata = load_model_from_checkpoint(weights_path, device)
    weights_name = os.path.basename(weights_path)

    # 2. Resolve image and annotation paths
    img_dir = Path(args.data_dir) / args.split / "img"
    ann_dir = Path(args.data_dir) / args.split / "ann"

    image_files = sorted(list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")

    if args.image_name:
        img_path = img_dir / args.image_name
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
    elif args.random:
        img_path = random.choice(image_files)
    else:
        index = max(0, min(args.index, len(image_files) - 1))
        img_path = image_files[index]

    ann_path = ann_dir / f"{img_path.name}.json"
    print(f"Processing image: {img_path.name} (from '{args.split}' split)")

    # 3. Load Ground Truth
    gt_objects = load_ground_truth(ann_path)

    # 4. Run Model Prediction
    img, pred_objects = run_inference_on_image(
        model=model,
        image_path=img_path,
        device=device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    print(f"Ground-truth cells: {len(gt_objects)}")
    print(f"Predicted detections: {len(pred_objects)}")

    # 5. Render Plot
    plot_comparison(
        img=img,
        gt_objects=gt_objects,
        pred_objects=pred_objects,
        image_name=img_path.name,
        weights_name=weights_name,
        metadata=metadata,
        save_path=args.save_path,
        show=not args.no_show,
        single_panel=args.single_panel,
    )


if __name__ == "__main__":
    main()

