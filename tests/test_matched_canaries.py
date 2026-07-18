"""Tests for the matched (exchangeable) canary design.

The matched design exists to eliminate the appearance-difference artifact:
inserted and held-out canaries must be drawn from the SAME distribution, with
membership decided only by a random split. These tests verify that property.
"""

from __future__ import annotations

import statistics

import pytest

torch = pytest.importorskip("torch")

from dp_audit_tightness.auditing.canary.generation import (
    generate_matched_canaries,
    insert_matched_canaries_into_dataset,
)
from dp_audit_tightness.config import CanaryConfig


def _make_config(num_canaries: int = 200) -> CanaryConfig:
    return CanaryConfig(
        strategy="matched_random_canaries",
        num_canaries=num_canaries,
        insertion_rate=0.01,
    )


def test_generates_two_groups_worth_of_canaries():
    canaries = generate_matched_canaries(_make_config(50), seed=7, num_classes=10)
    assert len(canaries) == 100


def test_inserted_and_reference_images_are_identical_per_payload():
    canaries = generate_matched_canaries(_make_config(10), seed=7, num_classes=10)
    for payload in canaries:
        assert payload.inserted_image is payload.reference_image


def test_split_is_half_half_and_disjoint():
    canaries = generate_matched_canaries(_make_config(100), seed=7, num_classes=10)
    dataset = torch.utils.data.TensorDataset(
        torch.zeros(500, 1, 28, 28), torch.zeros(500, dtype=torch.long)
    )
    result = insert_matched_canaries_into_dataset(dataset, canaries, seed=11)
    inserted_ids = {p.identifier for p in result.inserted_canaries}
    holdout_ids = {p.identifier for p in result.reference_canaries}
    assert len(inserted_ids) == 100
    assert len(holdout_ids) == 100
    assert not inserted_ids & holdout_ids
    assert result.inserted_example_count == 100


def test_split_depends_on_seed_not_appearance():
    canaries = generate_matched_canaries(_make_config(100), seed=7, num_classes=10)
    dataset = torch.utils.data.TensorDataset(
        torch.zeros(500, 1, 28, 28), torch.zeros(500, dtype=torch.long)
    )
    split_a = insert_matched_canaries_into_dataset(dataset, canaries, seed=11)
    split_b = insert_matched_canaries_into_dataset(dataset, canaries, seed=12)
    ids_a = {p.identifier for p in split_a.inserted_canaries}
    ids_b = {p.identifier for p in split_b.inserted_canaries}
    assert ids_a != ids_b  # different seeds -> different membership assignment


def test_groups_have_matching_intensity_distributions():
    """The old design gave references 0.75x intensity; matched must not."""
    canaries = generate_matched_canaries(_make_config(300), seed=7, num_classes=10)
    dataset = torch.utils.data.TensorDataset(
        torch.zeros(500, 1, 28, 28), torch.zeros(500, dtype=torch.long)
    )
    result = insert_matched_canaries_into_dataset(dataset, canaries, seed=11)
    inserted_means = [float(p.inserted_image.mean()) for p in result.inserted_canaries]
    holdout_means = [float(p.reference_image.mean()) for p in result.reference_canaries]
    # Identically distributed groups: means differ by far less than the 25%
    # intensity gap the old design baked in.
    assert abs(statistics.fmean(inserted_means) - statistics.fmean(holdout_means)) < 0.005
    assert abs(max(inserted_means) - max(holdout_means)) < 0.05


def test_patch_positions_are_not_label_deterministic():
    """Two canaries with the same label must be able to land on different patches."""
    canaries = generate_matched_canaries(_make_config(200), seed=7, num_classes=10)
    positions_by_label: dict[int, set[str]] = {}
    for payload in canaries:
        row_col = payload.descriptor.split("row=")[1]
        positions_by_label.setdefault(payload.target_label, set()).add(row_col)
    assert any(len(positions) > 1 for positions in positions_by_label.values())
