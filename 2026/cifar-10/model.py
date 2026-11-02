import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Standard residual block with two 3x3 convolutions, batch normalization,
    and a shortcut connection for downsampling or dimension matching.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNet14(nn.Module):
    """
    A 14-layer ResNet tailored for CIFAR-10 (doubling the depth of the initial 7-layer CNN).

    Architecture:
      - Initial Conv: 3 -> 32 channels (3x3 conv)
      - Stage 1 (32x32): 2 Residual Blocks (32 channels)  -> 4 conv layers
      - Stage 2 (16x16): 2 Residual Blocks (64 channels)  -> 4 conv layers
      - Stage 3 (8x8):   2 Residual Blocks (128 channels) -> 4 conv layers
      - Head: AdaptiveAvgPool2d((1, 1)) + Linear(128, num_classes)
      Total: 1 + 4 + 4 + 4 = 13 Conv layers + 1 Linear layer = 14 weighted layers.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Stage 1: 32x32 -> 32 channels (2 blocks = 4 conv layers)
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
        )
        # Stage 2: 32x32 -> 16x16 -> 64 channels (2 blocks = 4 conv layers)
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        )
        # Stage 3: 16x16 -> 8x8 -> 128 channels (2 blocks = 4 conv layers)
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# Aliases for backward compatibility
CIFARResNet14 = ResNet14
BasicCNN = ResNet14
