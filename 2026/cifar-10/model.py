import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    """
    A basic Convolutional Neural Network for CIFAR-10 image classification.

    Architecture:
      - 3 Convolutional blocks with BatchNorm, ReLU, and MaxPooling
      - Dropout for regularization to mitigate overfitting on CIFAR-10
      - Fully connected classification head
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super().__init__()
        
        # Block 1: Input 32x32x3 -> 16x16x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Block 2: Input 16x16x32 -> 8x8x64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Block 3: Input 8x8x64 -> 4x4x128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Classification Head: 128 * 4 * 4 = 2048 -> 512 -> num_classes (10)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits
