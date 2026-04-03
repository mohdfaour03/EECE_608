from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

from dp_audit_tightness.config import CanaryConfig


@dataclass(slots=True)
class CanaryPayload:
    identifier: str
    target_label: int
    inserted_image: object
    reference_image: object
    descriptor: str


@dataclass(slots=True)
class CanaryInsertionResult:
    augmented_train_dataset: object
    inserted_canaries: list[CanaryPayload]
    reference_canaries: list[CanaryPayload]
    inserted_example_count: int
    unique_inserted_count: int


def generate_canaries(
    config: CanaryConfig,
    seed: int,
    *,
    num_classes: int,
    image_shape: tuple[int, ...] = (1, 28, 28),
) -> list[CanaryPayload]:
    """Generate canary payloads for evaluator-controlled auditing.

    Parameters
    ----------
    image_shape : tuple
        (C, H, W) of the target dataset.  Defaults to MNIST (1, 28, 28).
        Pass (3, 32, 32) for CIFAR-10.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for canary generation.") from exc

    rng = random.Random(seed)
    canaries: list[CanaryPayload] = []
    for index in range(config.num_canaries):
        target_label = rng.randrange(num_classes)
        inserted_image, inserted_descriptor = _build_canary_image(
            strategy=config.strategy,
            target_label=target_label,
            seed=seed + index,
            inserted=True,
            image_shape=image_shape,
        )
        reference_image, reference_descriptor = _build_canary_image(
            strategy=config.strategy,
            target_label=target_label,
            seed=seed + index + 10_000,
            inserted=False,
            image_shape=image_shape,
        )
        descriptor = f"inserted={inserted_descriptor};reference={reference_descriptor}"
        canaries.append(
            CanaryPayload(
                identifier=f"{config.strategy}_{index}",
                target_label=target_label,
                inserted_image=inserted_image.to(dtype=torch.float32),
                reference_image=reference_image.to(dtype=torch.float32),
                descriptor=descriptor,
            )
        )
    return canaries


def insert_canaries_into_dataset(
    dataset: object,
    canaries: list[CanaryPayload],
    *,
    insertion_rate: float,
    seed: int | None = None,
) -> CanaryInsertionResult:
    try:
        import torch
        from torch.utils.data import ConcatDataset, TensorDataset
    except ImportError as exc:
        raise RuntimeError("torch is required for canary insertion.") from exc

    if not canaries:
        raise ValueError("At least one canary is required for canary insertion.")

    inserted_example_count = max(1, int(round(len(dataset) * insertion_rate)))
    unique_inserted_count = min(len(canaries), inserted_example_count)
    selected_canaries = list(canaries)
    if seed is not None:
        random.Random(seed).shuffle(selected_canaries)
    selected_canaries = selected_canaries[:unique_inserted_count]
    cycle = itertools.cycle(selected_canaries)
    inserted_images = []
    inserted_labels = []
    for _ in range(inserted_example_count):
        payload = next(cycle)
        inserted_images.append(payload.inserted_image.clone())
        inserted_labels.append(payload.target_label)

    tensor_dataset = TensorDataset(
        torch.stack(inserted_images),
        torch.tensor(inserted_labels, dtype=torch.long),
    )
    augmented_train_dataset = ConcatDataset([dataset, tensor_dataset])
    return CanaryInsertionResult(
        augmented_train_dataset=augmented_train_dataset,
        inserted_canaries=selected_canaries,
        reference_canaries=selected_canaries,
        inserted_example_count=inserted_example_count,
        unique_inserted_count=unique_inserted_count,
    )


def _build_canary_image(
    *,
    strategy: str,
    target_label: int,
    seed: int,
    inserted: bool,
    image_shape: tuple[int, ...] = (1, 28, 28),
) -> tuple[object, str]:
    """Build a synthetic canary image.

    Supports arbitrary ``image_shape`` (C, H, W) so the same canary
    strategies work for both MNIST (1, 28, 28) and CIFAR-10 (3, 32, 32).
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for canary generation.") from exc

    channels, height, width = image_shape
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand((channels, height, width), generator=generator) * 0.04
    patch_size = {"random_canaries": 3, "improved_canaries": 4, "optimized_canaries": 5}.get(
        strategy,
        3,
    )
    intensity = {"random_canaries": 0.8, "improved_canaries": 0.95, "optimized_canaries": 1.0}.get(
        strategy,
        0.8,
    )
    base_row = 2 + ((target_label * 3 + seed) % (height - patch_size - 2))
    base_col = 2 + ((target_label * 5 + seed // 2) % (width - patch_size - 2))
    if not inserted:
        base_row = (base_row + 7) % (height - patch_size - 1)
        base_col = (base_col + 11) % (width - patch_size - 1)
        intensity *= 0.75

    image[:, base_row : base_row + patch_size, base_col : base_col + patch_size] = intensity
    stripe_end = min(height - 2, width - 2)
    if strategy in {"improved_canaries", "optimized_canaries"}:
        stripe_row = min(target_label, height - 3)
        image[:, stripe_row : stripe_row + 2, 2 : stripe_end] = max(intensity - 0.1, 0.55)
    if strategy == "optimized_canaries":
        stripe_col = min(target_label + 2, width - 3)
        image[:, 2 : stripe_end, stripe_col : stripe_col + 2] = max(intensity - 0.05, 0.6)
        mid_r, mid_c = height // 2 - 2, width // 2 - 2
        image[:, mid_r : mid_r + 4, mid_c : mid_c + 4] = 1.0

    return image.clamp(0.0, 1.0), f"label={target_label},row={base_row},col={base_col},inserted={inserted}"
