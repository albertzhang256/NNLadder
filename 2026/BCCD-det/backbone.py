"""
ResNet-18 Feature Extractor Backbone for Object Detection.

This module provides a ResNet-18 backbone initialized with ImageNet pretrained weights.
The classification head (AdaptiveAvgPool2d and Linear FC layer) is removed,
leaving only the convolutional feature extraction stages.

Spatial downsampling:
- Input image size (B, 3, H, W), e.g. (B, 3, 448, 448)
- Stem (conv1 + bn1 + relu + maxpool): stride 4 -> (B, 64, H/4, W/4)
- Stage 1 (layer1): stride 4 -> (B, 64, H/4, W/4)
- Stage 2 (layer2): stride 8 -> (B, 128, H/8, W/8)
- Stage 3 (layer3): stride 16 -> (B, 256, H/16, W/16)
- Stage 4 (layer4): stride 32 -> (B, 512, H/32, W/32)

For an input of (B, 3, 448, 448), the output feature map is (B, 512, 14, 14),
which naturally serves as a 14x14 spatial detection grid.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 feature extractor backbone.

    Args:
        pretrained: If True, loads weights pretrained on ImageNet-1K.
        freeze_stages: Number of initial stages to freeze (0 to 4).
                       0: all parameters are trainable.
                       1: freeze stem and layer1.
                       2: freeze stem, layer1, and layer2.
                       3: freeze stem, layer1, layer2, and layer3.
                       4: freeze all backbone parameters.
    """

    def __init__(self, pretrained: bool = True, freeze_stages: int = 0):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # 1. Stem: Conv2d (7x7, stride 2) -> BatchNorm -> ReLU -> MaxPool (3x3, stride 2)
        # Total stride = 4, output channels = 64
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 2. Residual stages
        self.layer1 = resnet.layer1  # 64 channels, stride 4
        self.layer2 = resnet.layer2  # 128 channels, stride 8
        self.layer3 = resnet.layer3  # 256 channels, stride 16
        self.layer4 = resnet.layer4  # 512 channels, stride 32

        # Metadata for downstream detector heads
        self.out_channels = 512
        self.downsample_ratio = 32
        self.pretrained = pretrained
        self.freeze_stages = 0

        # Optional parameter freezing
        self.set_freeze_stages(freeze_stages)

    def _stage_modules(self) -> List[List[nn.Module]]:
        """Returns ordered list of module groups corresponding to stages 1..4."""
        return [
            [self.conv1, self.bn1, self.layer1],
            [self.layer2],
            [self.layer3],
            [self.layer4],
        ]

    def set_freeze_stages(self, freeze_stages: int) -> None:
        """
        Freezes stages 1..freeze_stages and unfreezes any remaining stages.
        Frozen BatchNorm layers are kept in eval mode so running stats do not drift.
        """
        if not 0 <= freeze_stages <= 4:
            raise ValueError(f"freeze_stages must be between 0 and 4, got {freeze_stages}")

        self.freeze_stages = freeze_stages

        # Reset all parameters to trainable first so dynamic unfreezing works
        for p in self.parameters():
            p.requires_grad = True

        for stage_idx, modules in enumerate(self._stage_modules(), start=1):
            is_frozen = freeze_stages >= stage_idx
            for mod in modules:
                for p in mod.parameters():
                    p.requires_grad = not is_frozen
                if is_frozen:
                    mod.eval()
                elif self.training:
                    mod.train()

    def train(self, mode: bool = True):
        """Sets training mode while keeping frozen backbone stages in eval mode."""
        super().train(mode)
        if mode and self.freeze_stages > 0:
            for stage_idx, modules in enumerate(self._stage_modules(), start=1):
                if self.freeze_stages >= stage_idx:
                    for mod in modules:
                        mod.eval()
        return self

    def forward(
        self, x: torch.Tensor, return_stages: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward computation through ResNet-18 convolutional layers.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            return_stages: If True, returns a dict of intermediate feature maps
                           {"c2": layer1, "c3": layer2, "c4": layer3, "c5": layer4}.
                           If False, returns only the final stage 4 feature map (c5).

        Returns:
            Feature map tensor of shape (batch_size, 512, height/32, width/32),
            or dictionary of multi-stage feature tensors.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        c2 = self.layer1(x)   # (B, 64,  H/4,  W/4)
        c3 = self.layer2(c2)  # (B, 128, H/8,  W/8)
        c4 = self.layer3(c3)  # (B, 256, H/16, W/16)
        c5 = self.layer4(c4)  # (B, 512, H/32, W/32)

        if return_stages:
            return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
        return c5


def build_resnet18_backbone(pretrained: bool = True, freeze_stages: int = 0) -> ResNet18Backbone:
    """Helper function to instantiate the ResNet-18 feature extractor."""
    return ResNet18Backbone(pretrained=pretrained, freeze_stages=freeze_stages)


if __name__ == "__main__":
    print("=== Testing ResNet-18 Feature Extractor Backbone ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = build_resnet18_backbone(pretrained=True).to(device)
    backbone.eval()

    # Synthetic test input
    sample_input = torch.randn(2, 3, 448, 448, device=device)
    with torch.no_grad():
        features = backbone(sample_input)

    print(f"Input shape:              {sample_input.shape}")
    print(f"Extracted feature shape:  {features.shape}")
    print(f"Downsample ratio:         {backbone.downsample_ratio}x")
    print(f"Output channels:          {backbone.out_channels}")
    print(f"Feature tensor mean:      {features.mean().item():.4f}")
    print(f"Feature tensor std:       {features.std().item():.4f}")

    # Test multi-stage output
    with torch.no_grad():
        stages = backbone(sample_input, return_stages=True)
    print("\nIntermediate stage shapes:")
    for stage_name, stage_tensor in stages.items():
        print(f"  {stage_name}: {stage_tensor.shape}")

