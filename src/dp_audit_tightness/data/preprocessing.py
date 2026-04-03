from __future__ import annotations


def build_transform(dataset_name: str):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for dataset preprocessing. Install project dependencies first."
        ) from exc

    name = dataset_name.lower()

    if name == "mnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    if name == "cifar10":
        # Standard CIFAR-10 normalization (per-channel mean and std).
        # No data augmentation — consistent with DP-SGD auditing literature
        # (Nasr et al. 2023, Lu & Groth 2024).
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    if name == "purchase100":
        # Purchase-100 is already numeric; no image transforms needed.
        # The dataset loader handles tensor conversion directly.
        return None

    if name == "adult":
        # Adult Census Income is tabular; no image transforms needed.
        # The dataset loader handles preprocessing directly.
        return None

    raise NotImplementedError(f"No transform defined for dataset={dataset_name}")

