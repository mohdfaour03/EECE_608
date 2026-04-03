"""Tabular MLP for Purchase-100 and Adult Census experiments.

A simple multi-layer perceptron for tabular (non-image) data.
Fully compatible with Opacus DP-SGD — no batch norm, no operations
that break per-sample gradient clipping.
"""

from __future__ import annotations


def build_tabular_mlp(*, input_dim: int, hidden_dim: int, num_classes: int):
    """Return a 2-hidden-layer MLP for tabular classification.

    Architecture: input_dim -> hidden_dim -> hidden_dim // 2 -> num_classes
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for TabularMLP construction.") from exc

    mid_dim = max(16, hidden_dim // 2)

    class TabularMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, num_classes),
            )

        def forward(self, x):
            return self.network(x)

    return TabularMLP()
