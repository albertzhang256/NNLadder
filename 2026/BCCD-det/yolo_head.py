"""
YOLO Detection Head and Non-Maximum Suppression (NMS) for Blood Cell Detection (BCCD).

This module defines:
1. YOLOHead: A convolutional head that maps backbone features to grid predictions.
2. Box conversion & IoU utilities (box_cxcywh_to_xyxy, box_iou).
3. Non-Maximum Suppression (NMS): Both pure-PyTorch reference (py_nms) and
   hardware-accelerated multi-class NMS (non_max_suppression).
4. BCCDYOLO: End-to-end detector integrating ResNet-18 backbone, YOLO head, and NMS prediction.

Default Configuration for BCCD:
- Classes (C = 3): RBC (0), WBC (1), Platelets (2)
- Bounding boxes per grid cell (B_boxes = 2)
- Grid output channels per cell: 2 * 5 + 3 = 13 channels
    - Box 1: [x, y, w, h, conf]      (indices 0..4)
    - Box 2: [x, y, w, h, conf]      (indices 5..9)
    - Classes: [p_RBC, p_WBC, p_Plt] (indices 10..12)
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.ops as ops

from backbone import ResNet18Backbone, build_resnet18_backbone


# ==============================================================================
# 1. Bounding Box Geometry & IoU Utilities
# ==============================================================================

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2].
    Clamps coordinates to [0.0, 1.0].

    Args:
        boxes: Tensor of shape (..., 4) containing [cx, cy, w, h].

    Returns:
        Tensor of shape (..., 4) containing [x1, y1, x2, y2].
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(min=0.0, max=1.0)
    y1 = (cy - 0.5 * h).clamp(min=0.0, max=1.0)
    x2 = (cx + 0.5 * w).clamp(min=0.0, max=1.0)
    y2 = (cy + 0.5 * h).clamp(min=0.0, max=1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Intersection-over-Union (IoU) between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes2: Tensor of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        iou: Tensor of shape (N, M) with values in [0.0, 1.0].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Intersection coordinates
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-7)


# ==============================================================================
# 2. Non-Maximum Suppression (NMS) Algorithms
# ==============================================================================

def py_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.30) -> torch.Tensor:
    """
    Pure PyTorch single-class Non-Maximum Suppression (for learning and debugging).

    Args:
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Tensor of shape (N,) confidence scores.
        iou_threshold: IoU overlap threshold for suppression.

    Returns:
        keep: 1D LongTensor of indices of boxes to retain.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # Sort descending by score
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0]
        keep.append(idx.item())

        if order.numel() == 1:
            break

        # Compute IoU between best box and all other remaining boxes
        cur_box = boxes[idx : idx + 1]
        rem_boxes = boxes[order[1:]]
        ious = box_iou(cur_box, rem_boxes).squeeze(0)

        # Retain only boxes whose overlap is below threshold
        rem_mask = ious <= iou_threshold
        order = order[1:][rem_mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def non_max_suppression(
    decoded_preds: Dict[str, torch.Tensor],
    conf_threshold: float = 0.50,
    iou_threshold: float = 0.30,
    max_detections: int = 100,
) -> List[torch.Tensor]:
    """
    Performs multi-class Non-Maximum Suppression (NMS) on decoded YOLO predictions.

    Args:
        decoded_preds: Dictionary from YOLOHead.decode() containing:
            - 'boxes': (B, S, S, B_boxes, 4) in [cx, cy, w, h] format.
            - 'confidences': (B, S, S, B_boxes) objectness confidences.
            - 'class_probs': (B, S, S, num_classes) class probabilities.
        conf_threshold: Candidate box score cutoff (score = conf * class_prob).
        iou_threshold: IoU suppression cutoff for overlapping boxes.
        max_detections: Maximum detections to return per image.

    Returns:
        List of length B, where each element is a Tensor of shape (K, 6):
            [x1, y1, x2, y2, score, class_id]
        where coordinates are normalized in [0, 1].
    """
    boxes = decoded_preds["boxes"]          # (B, S, S, B_boxes, 4)
    confs = decoded_preds["confidences"]    # (B, S, S, B_boxes)
    class_probs = decoded_preds["class_probs"]  # (B, S, S, num_classes)

    batch_size, grid_h, grid_w, num_boxes, _ = boxes.shape
    device = boxes.device

    batch_detections = []

    for i in range(batch_size):
        img_boxes = boxes[i]          # (S, S, num_boxes, 4)
        img_confs = confs[i]          # (S, S, num_boxes)
        img_class_probs = class_probs[i]  # (S, S, num_classes)

        # 1. Best class and its probability for each grid cell
        best_class_prob, best_class_id = img_class_probs.max(dim=-1)  # (S, S)

        # 2. Flatten all grid cells and box candidates for this image
        # Total candidate boxes M = S * S * num_boxes (e.g. 14 * 14 * 2 = 392)
        flat_boxes_cxcywh = img_boxes.reshape(-1, 4)
        flat_boxes_xyxy = box_cxcywh_to_xyxy(flat_boxes_cxcywh)
        flat_confs = img_confs.reshape(-1)

        # Broadcast cell class predictions to each box in the cell
        class_prob_expanded = (
            best_class_prob.unsqueeze(-1).expand(-1, -1, num_boxes).reshape(-1)
        )
        class_id_expanded = (
            best_class_id.unsqueeze(-1).expand(-1, -1, num_boxes).reshape(-1)
        )

        # 3. Final detection score = objectness confidence * class probability
        flat_scores = flat_confs * class_prob_expanded

        # 4. Filter by confidence score threshold
        candidate_mask = flat_scores >= conf_threshold
        if not candidate_mask.any():
            batch_detections.append(torch.zeros((0, 6), device=device, dtype=torch.float32))
            continue

        cand_boxes = flat_boxes_xyxy[candidate_mask]
        cand_scores = flat_scores[candidate_mask]
        cand_classes = class_id_expanded[candidate_mask]

        # 5. Multi-class NMS (independent suppression per class)
        keep = ops.batched_nms(cand_boxes, cand_scores, cand_classes, iou_threshold)
        keep = keep[:max_detections]

        # 6. Assemble output tensor: [x1, y1, x2, y2, score, class_id]
        kept_boxes = cand_boxes[keep]
        kept_scores = cand_scores[keep, None]
        kept_classes = cand_classes[keep, None].float()

        detections = torch.cat([kept_boxes, kept_scores, kept_classes], dim=-1)
        batch_detections.append(detections)

    return batch_detections


# ==============================================================================
# 3. YOLO Detection Head
# ==============================================================================

class YOLOHead(nn.Module):
    """
    YOLO detection head.

    Maps backbone feature maps to bounding box coordinates, objectness confidences,
    and class probabilities. Includes built-in coordinate decoding and NMS filtering.

    Args:
        in_channels: Backbone output channels (512 for ResNet-18 layer4).
        num_classes: Number of object classes (3 for BCCD).
        num_boxes: Number of bounding boxes per grid cell (default: 2).
        hidden_channels: Intermediate channels for convolutional refinement (default: 256).
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 3,
        num_boxes: int = 2,
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        self.box_attr = 5  # [tx, ty, tw, th, conf]
        self.out_channels = num_boxes * self.box_attr + num_classes

        # 3x3 Conv refinement followed by 1x1 projection
        self.conv_refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv_out = nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the detection head.

        Args:
            x: Feature map tensor of shape (B, in_channels, S, S).

        Returns:
            predictions: Tensor of shape (B, S, S, out_channels).
        """
        feat = self.conv_refine(x)
        out = self.conv_out(feat)  # (B, out_channels, S, S)

        # Transpose to (B, S, S, out_channels) for spatial grid indexing
        return out.permute(0, 2, 3, 1).contiguous()

    def decode(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decodes raw model logits into normalized bounding box coordinates and probabilities.

        Args:
            predictions: Raw output tensor of shape (B, S, S, out_channels).

        Returns:
            Dictionary with 'boxes' (cx, cy, w, h in [0, 1]), 'confidences', and 'class_probs'.
        """
        batch_size, grid_h, grid_w, _ = predictions.shape
        device = predictions.device

        # Create coordinate grid of cell offsets
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=torch.float32),
            torch.arange(grid_w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid_x = grid_x.view(1, grid_h, grid_w, 1)
        grid_y = grid_y.view(1, grid_h, grid_w, 1)

        decoded_boxes = []
        confidences = []

        for b in range(self.num_boxes):
            start_idx = b * self.box_attr
            tx = predictions[..., start_idx + 0 : start_idx + 1]
            ty = predictions[..., start_idx + 1 : start_idx + 2]
            tw = predictions[..., start_idx + 2 : start_idx + 3]
            th = predictions[..., start_idx + 3 : start_idx + 4]
            tconf = predictions[..., start_idx + 4 : start_idx + 5]

            # 1. Box center (cx, cy): sigmoid offset relative to cell top-left corner
            cx = (torch.sigmoid(tx) + grid_x) / float(grid_w)
            cy = (torch.sigmoid(ty) + grid_y) / float(grid_h)

            # 2. Box dimensions (w, h): sigmoid ensures width/height are bounded in [0, 1]
            w = torch.sigmoid(tw)
            h = torch.sigmoid(th)

            box = torch.cat([cx, cy, w, h], dim=-1)
            decoded_boxes.append(box)

            # 3. Objectness confidence
            conf = torch.sigmoid(tconf).squeeze(-1)
            confidences.append(conf)

        boxes_tensor = torch.stack(decoded_boxes, dim=3)
        confs_tensor = torch.stack(confidences, dim=3)

        # 4. Class probabilities: softmax over C classes
        class_logits = predictions[..., self.num_boxes * self.box_attr :]
        class_probs = torch.softmax(class_logits, dim=-1)

        return {
            "boxes": boxes_tensor,
            "confidences": confs_tensor,
            "class_probs": class_probs,
        }

    def decode_and_nms(
        self,
        predictions: torch.Tensor,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.30,
        max_detections: int = 100,
    ) -> List[torch.Tensor]:
        """Decodes raw predictions and applies Non-Maximum Suppression."""
        decoded = self.decode(predictions)
        return non_max_suppression(
            decoded,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    def predict(
        self,
        features: torch.Tensor,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.30,
        max_detections: int = 100,
    ) -> List[torch.Tensor]:
        """Forward pass, decode, and apply NMS directly from backbone features."""
        raw_preds = self.forward(features)
        return self.decode_and_nms(
            raw_preds,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )


# ==============================================================================
# 4. Complete BCCDYOLO Detector
# ==============================================================================

class BCCDYOLO(nn.Module):
    """
    Complete YOLO detector combining ResNet-18 backbone with YOLOHead and NMS.

    Args:
        num_classes: Number of classes (default: 3).
        num_boxes: Number of boxes per grid cell (default: 2).
        pretrained_backbone: Whether to initialize ResNet-18 with ImageNet weights.
        freeze_stages: Number of backbone stages to freeze (0 to 4).
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_boxes: int = 2,
        pretrained_backbone: bool = True,
        freeze_stages: int = 0,
    ):
        super().__init__()
        self.backbone = build_resnet18_backbone(
            pretrained=pretrained_backbone, freeze_stages=freeze_stages
        )
        self.head = YOLOHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            num_boxes=num_boxes,
        )

        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.out_channels = self.head.out_channels

    def set_freeze_stages(self, freeze_stages: int) -> None:
        """Dynamically updates which early ResNet-18 backbone stages are frozen (0 to 4)."""
        self.backbone.set_freeze_stages(freeze_stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and detection head returning raw predictions."""
        features = self.backbone(x)
        return self.head(features)

    def decode(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Delegate decoding to head."""
        return self.head.decode(predictions)

    def predict(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.30,
        max_detections: int = 100,
    ) -> List[torch.Tensor]:
        """
        End-to-end inference: Images -> Backbone -> Head -> Decode -> NMS.

        Args:
            images: Input images tensor of shape (B, 3, H, W).
            conf_threshold: Candidate score cutoff.
            iou_threshold: NMS overlap cutoff.
            max_detections: Max detected boxes to return per image.

        Returns:
            List of length B, where each element is a Tensor of shape (K, 6):
                [x1, y1, x2, y2, score, class_id]
        """
        raw_preds = self.forward(images)
        return self.head.decode_and_nms(
            raw_preds,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )


def build_yolo_head(
    in_channels: int = 512,
    num_classes: int = 3,
    num_boxes: int = 2,
    hidden_channels: int = 256,
) -> YOLOHead:
    """Helper function to instantiate a standalone YOLOHead."""
    return YOLOHead(
        in_channels=in_channels,
        num_classes=num_classes,
        num_boxes=num_boxes,
        hidden_channels=hidden_channels,
    )


def build_bccd_yolo(
    num_classes: int = 3,
    num_boxes: int = 2,
    pretrained_backbone: bool = True,
    freeze_stages: int = 0,
) -> BCCDYOLO:
    """Helper function to instantiate the full BCCDYOLO detector."""
    return BCCDYOLO(
        num_classes=num_classes,
        num_boxes=num_boxes,
        pretrained_backbone=pretrained_backbone,
        freeze_stages=freeze_stages,
    )


# ==============================================================================
# 5. Testing & Verification Entrypoint
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("=== Testing YOLOHead, NMS, and Full BCCDYOLO Detector ===")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Standalone NMS Unit Test (Verify Suppression of Overlapping Boxes)
    print("\n--- 1. NMS Algorithm Unit Test ---")
    # Two overlapping boxes for the same class (RBC = 0):
    # Box A: [0.10, 0.10, 0.30, 0.30], score = 0.90
    # Box B: [0.11, 0.11, 0.31, 0.31], score = 0.80 (IoU ~0.85 with Box A)
    # Box C: [0.60, 0.60, 0.80, 0.80], score = 0.95 (Disjoint box)
    test_boxes = torch.tensor(
        [[0.10, 0.10, 0.30, 0.30], [0.11, 0.11, 0.31, 0.31], [0.60, 0.60, 0.80, 0.80]],
        device=device,
    )
    test_scores = torch.tensor([0.90, 0.80, 0.95], device=device)
    test_classes = torch.tensor([0, 0, 0], device=device)

    kept_indices = ops.batched_nms(test_boxes, test_scores, test_classes, iou_threshold=0.45)
    print(f"Initial candidate boxes: {len(test_boxes)}")
    print(f"Boxes kept after NMS:    {len(kept_indices)} (expected: 2, duplicate suppressed)")
    assert len(kept_indices) == 2, "NMS should suppress the lower scoring overlapping box!"
    print("NMS unit test passed successfully!")

    # 2. Standalone YOLOHead Test
    print("\n--- 2. Standalone YOLOHead Forward & Decode Test ---")
    head = YOLOHead(in_channels=512, num_classes=3, num_boxes=2).to(device)
    dummy_features = torch.randn(2, 512, 14, 14, device=device)
    with torch.no_grad():
        raw_preds = head(dummy_features)
        decoded = head.decode(raw_preds)
    print(f"Head output tensor shape: {raw_preds.shape} (B, S, S, Channels)")
    print(f"Decoded boxes shape:       {decoded['boxes'].shape} (B, S, S, num_boxes, 4)")

    # 3. End-to-End BCCDYOLO Model with Prediction & NMS
    print("\n--- 3. End-to-End BCCDYOLO Inference with NMS ---")
    model = BCCDYOLO(num_classes=3, num_boxes=2, pretrained_backbone=True).to(device)
    model.eval()

    dummy_images = torch.randn(2, 3, 448, 448, device=device)
    with torch.no_grad():
        # Predict with confidence threshold 0.1 to inspect output structure
        detections = model.predict(
            dummy_images, conf_threshold=0.1, iou_threshold=0.45, max_detections=20
        )

    print(f"Batch size: {len(detections)}")
    for img_idx, det in enumerate(detections):
        print(f"  Image {img_idx}: {det.shape[0]} final detections after NMS")
        if det.shape[0] > 0:
            sample_det = det[0]
            print(
                f"    Top detection: box=[{sample_det[0]:.2f}, {sample_det[1]:.2f}, "
                f"{sample_det[2]:.2f}, {sample_det[3]:.2f}], score={sample_det[4]:.3f}, "
                f"class_id={int(sample_det[5].item())}"
            )

    print("\n" + "=" * 70)
    print("YOLO head and NMS verified successfully!")
    print("=" * 70)
