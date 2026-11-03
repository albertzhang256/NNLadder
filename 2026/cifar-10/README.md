# CIFAR-10 Classification with ResNet-14

This directory contains PyTorch code for training and evaluating a 14-layer Residual Network (ResNet-14) tailored for the CIFAR-10 dataset.

## Dataset Storage
By default, the dataset is automatically downloaded and stored in:
```
~/data/cifar-10
```
This is configured via the `--data-dir` argument in `train.py`.

## Model Architecture (`model.py`)
`ResNet14` is a 14-layer residual neural network designed specifically for $32\times 32$ CIFAR-10 images:
- **Input / Stem**: 1x Conv2d (32 channels, 3x3) + BatchNorm2d + ReLU
- **Stage 1 (32x32)**: 2 Residual Blocks (32 channels, 4 Conv layers)
- **Stage 2 (16x16)**: 2 Residual Blocks (64 channels, 4 Conv layers, stride 2 downsampling)
- **Stage 3 (8x8)**: 2 Residual Blocks (128 channels, 4 Conv layers, stride 2 downsampling)
- **Classifier**: AdaptiveAvgPool2d (1x1) + Flatten + Linear (128 -> 10)
- **Total weighted layers**: 13 Conv + 1 Linear = 14 layers (~697k trainable parameters)

## Requirements
- Python 3.10+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy

Install dependencies using `uv` or `pip`:
```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

## Running Training
To train the model:
```bash
python train.py
```

### Command-line Options
- `--data-dir`: Directory where CIFAR-10 is stored (default: `~/data/cifar-10`).
- `--batch-size`: Batch size for training and testing (default: `64`).
- `--epochs`: Number of epochs to train (default: `15`).
- `--lr`: Learning rate for Adam optimizer (default: `0.001`).
- `--weight-decay`: Weight decay for L2 regularization (default: `1e-4`).
- `--seed`: Random seed for reproducibility (default: `42`).
- `--save-path`: File path to save the best model weights (default: `cifar10_resnet14.pth`).
- `--num-workers`: DataLoader workers (default: `2`).
- `--no-cuda`: Disable GPU training and force CPU.
- `--no-trivial-augment`: Disables TrivialAugmentWide data augmentation (enabled by default).
