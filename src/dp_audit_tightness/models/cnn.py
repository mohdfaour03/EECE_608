"""Simple CNN for CIFAR-10 DP-SGD experiments.

Architecture follows the standard used in the DP auditing literature
(Nasr et al. 2023, Lu & Groth NeurIPS 2024): two convolutional layers
with max-pooling followed by two fully-connected layers.  Kept
deliberately simple so that Opacus can wrap it without modification.
"""

from __future__ import annotations


def build_cnn_cifar10(*, num_classes: int = 10):
    """Return a small CNN compatible with Opacus DP-SGD.

    The architecture avoids batch normalization (incompatible with
    per-sample gradient clipping) and uses only operations that Opacus
    supports out of the box.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for CNN construction.") from exc

    class SimpleCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Conv block 1: 3 -> 32 channels
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)       # 32x32 -> 16x16

            # Conv block 2: 32 -> 64 channels
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)       # 16x16 -> 8x8

            # Classifier: 64*8*8 = 4096 -> 128 -> num_classes
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleCNN()
