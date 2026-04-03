from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dp_audit_tightness.config import DatasetConfig
from dp_audit_tightness.data.preprocessing import build_transform


@dataclass(slots=True)
class DatasetBundle:
    train_dataset: object
    eval_dataset: object
    input_dim: int
    num_classes: int
    train_size: int
    eval_size: int


def load_dataset_bundle(config: DatasetConfig, split_seed: int) -> DatasetBundle:
    name = config.name.lower()
    if name == "mnist":
        return _load_mnist(config, split_seed)
    if name == "cifar10":
        return _load_cifar10(config, split_seed)
    if name == "purchase100":
        return _load_purchase100(config, split_seed)
    if name == "adult":
        return _load_adult(config, split_seed)
    raise NotImplementedError(f"Dataset not supported: {config.name}")


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def _load_mnist(config: DatasetConfig, split_seed: int) -> DatasetBundle:
    import torch
    from torch.utils.data import random_split
    from torchvision.datasets import MNIST

    data_root = Path(config.data_dir)
    transform = build_transform("mnist")
    full_dataset = MNIST(root=data_root, train=True, download=True, transform=transform)
    eval_size = int(len(full_dataset) * config.validation_fraction)
    train_size = len(full_dataset) - eval_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)
    return DatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        input_dim=28 * 28,
        num_classes=10,
        train_size=train_size,
        eval_size=eval_size,
    )


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def _load_cifar10(config: DatasetConfig, split_seed: int) -> DatasetBundle:
    import torch
    from torch.utils.data import random_split
    from torchvision.datasets import CIFAR10

    data_root = Path(config.data_dir)
    transform = build_transform("cifar10")
    full_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    eval_size = int(len(full_dataset) * config.validation_fraction)
    train_size = len(full_dataset) - eval_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)
    return DatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        input_dim=3 * 32 * 32,      # 3072 (used by tabular fallbacks; CNN ignores this)
        num_classes=10,
        train_size=train_size,
        eval_size=eval_size,
    )


# ---------------------------------------------------------------------------
# Purchase-100  (Shokri et al. 2017 — classic MIA benchmark)
# ---------------------------------------------------------------------------

def _load_purchase100(config: DatasetConfig, split_seed: int) -> DatasetBundle:
    import torch
    from torch.utils.data import TensorDataset, random_split

    data_root = Path(config.data_dir)
    data_path = data_root / "purchase100.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Purchase-100 data not found at {data_path}. "
            "Download the preprocessed dataset and save it as an .npz file with "
            "keys 'features' (float32, shape [N, 600]) and 'labels' (int64, shape [N,])."
        )
    import numpy as np

    data = np.load(data_path)
    features = torch.tensor(data["features"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    full_dataset = TensorDataset(features, labels)
    eval_size = int(len(full_dataset) * config.validation_fraction)
    train_size = len(full_dataset) - eval_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)
    num_classes = int(labels.max().item()) + 1
    return DatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        input_dim=600,
        num_classes=num_classes,
        train_size=train_size,
        eval_size=eval_size,
    )


# ---------------------------------------------------------------------------
# Adult Census Income  (UCI — policy-relevant tabular benchmark)
# ---------------------------------------------------------------------------

def _load_adult(config: DatasetConfig, split_seed: int) -> DatasetBundle:
    import torch
    from torch.utils.data import TensorDataset, random_split

    data_root = Path(config.data_dir)
    data_path = data_root / "adult.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Adult Census data not found at {data_path}. "
            "Download from UCI ML Repository, preprocess with one-hot encoding, "
            "and save as .npz with keys 'features' (float32) and 'labels' (int64)."
        )
    import numpy as np

    data = np.load(data_path)
    features = torch.tensor(data["features"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    full_dataset = TensorDataset(features, labels)
    eval_size = int(len(full_dataset) * config.validation_fraction)
    train_size = len(full_dataset) - eval_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)
    input_dim = int(features.shape[1])
    num_classes = int(labels.max().item()) + 1
    return DatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        input_dim=input_dim,
        num_classes=num_classes,
        train_size=train_size,
        eval_size=eval_size,
    )

