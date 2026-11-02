# CIFAR-10 Classification with Basic CNN

This directory contains PyTorch code for training and evaluating a basic Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

## Dataset Storage
By default, the dataset is automatically downloaded and stored in:
```
~/data/cifar-10
```
This is configured via the `--data-dir` argument in `train.py`.

## Model Architecture (`model.py`)
`BasicCNN` is a 3-block convolutional neural network designed for CIFAR-10:
- **Block 1**: 2x Conv2d (32 channels, 3x3) + BatchNorm2d + ReLU + MaxPool2d (2x2) + Dropout (0.25)
- **Block 2**: 2x Conv2d (64 channels, 3x3) + BatchNorm2d + ReLU + MaxPool2d (2x2) + Dropout (0.25)
- **Block 3**: Conv2d (128 channels, 3x3) + BatchNorm2d + ReLU + MaxPool2d (2x2) + Dropout (0.25)
- **Classifier**: Flatten + Linear (2048 -> 512) + BatchNorm1d + ReLU + Dropout (0.5) + Linear (512 -> 10)

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
- `--save-path`: File path to save the best model weights (default: `cifar10_basic_cnn.pth`).
- `--num-workers`: DataLoader workers (default: `2`).
- `--no-cuda`: Disable GPU training and force CPU.
