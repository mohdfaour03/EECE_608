"""Immutable evaluation harness for autoresearch.

DO NOT MODIFY THIS FILE. The agent only modifies experiment.py.

This script:
1. Ensures a pre-trained DP-SGD model exists (trains one if missing).
2. Provides the evaluation function that experiment.py calls.
3. Provides helpers for training shadow/reference models on GPU.
4. Prints the final metric line that the agent loop parses.
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

AUTORESEARCH_DIR = Path(__file__).resolve().parent
CACHE_DIR = AUTORESEARCH_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed experiment constants — the agent cannot change these
# ---------------------------------------------------------------------------
DATASET = "mnist"
MODEL_NAME = "simple_mlp"
INPUT_DIM = 784
HIDDEN_DIM = 128          # bigger than the old 64 — more capacity to memorize
NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)
BATCH_SIZE = 256
EPOCHS = 3                # enough overfitting for signal, not so much that the gap vanishes
CLIPPING_NORM = 1.0
NOISE_MULTIPLIER = 1.1
DELTA = 1e-5
LEARNING_RATE = 0.15
SPLIT_SEED = 11
TRAINING_SEED = 123


@dataclass
class EvalResult:
    tightness_ratio: float
    epsilon_lower: float
    epsilon_upper: float
    privacy_loss_gap: float
    member_score_mean: float
    nonmember_score_mean: float
    score_gap: float
    num_member_samples: int
    num_nonmember_samples: int
    member_favoring: bool
    selected_tpr: float | None
    selected_fpr: float | None
    wall_seconds: float


def _build_train_config(epochs=EPOCHS, hidden_dim=HIDDEN_DIM, seed=TRAINING_SEED,
                        split_seed=SPLIT_SEED, noise_multiplier=NOISE_MULTIPLIER):
    """Build a TrainExperimentConfig programmatically."""
    from dp_audit_tightness.config import (
        TrainExperimentConfig, DatasetConfig, ModelConfig,
        TrainingConfig, OptimizerConfig, PrivacyConfig, RunConfig,
    )
    return TrainExperimentConfig(
        experiment_name=f"autoresearch_mnist_e{epochs}_h{hidden_dim}_s{seed}",
        dataset=DatasetConfig(name=DATASET, data_dir="data/raw",
                              validation_fraction=0.05, num_workers=0),
        model=ModelConfig(name=MODEL_NAME, input_dim=INPUT_DIM,
                          hidden_dim=hidden_dim, num_classes=NUM_CLASSES),
        training=TrainingConfig(
            batch_size=BATCH_SIZE, eval_batch_size=512, epochs=epochs,
            clipping_norm=CLIPPING_NORM, noise_multiplier=noise_multiplier,
            optimizer=OptimizerConfig(name="sgd", learning_rate=LEARNING_RATE,
                                     weight_decay=0.0, momentum=0.0),
        ),
        privacy=PrivacyConfig(delta=DELTA, accountant="rdp"),
        run=RunConfig(split_seed=split_seed, training_seed=seed,
                      results_root="results", save_checkpoint=True),
    )


def get_training_record_and_config():
    """Return (training_record, training_config) from cache, training if needed."""
    from dp_audit_tightness.training.dp_sgd import train_dp_sgd
    from dp_audit_tightness.utils.results import (
        save_training_result, save_json, load_training_result,
    )
    from dp_audit_tightness.utils.seeds import set_global_seed
    from dp_audit_tightness.config import config_to_dict, TrainExperimentConfig

    result_path = CACHE_DIR / "training_result.json"
    config_snapshot_path = CACHE_DIR / "training_config.json"
    checkpoint_path = CACHE_DIR / "model.pt"

    if result_path.exists() and config_snapshot_path.exists():
        record = load_training_result(result_path)
        config = TrainExperimentConfig.from_dict(
            json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        )
        return record, config

    print(f">>> Training DP-SGD model (hidden={HIDDEN_DIM}, epochs={EPOCHS}) ...")
    train_config = _build_train_config()

    set_global_seed(TRAINING_SEED)
    _logger = logging.getLogger("autoresearch")
    outcome = train_dp_sgd(config=train_config, logger=_logger)

    outcome.record.model_artifact_path = str(checkpoint_path)
    if outcome.checkpoint_path and Path(outcome.checkpoint_path).exists():
        import shutil
        shutil.copy2(outcome.checkpoint_path, checkpoint_path)
    outcome.record.config_snapshot_path = str(config_snapshot_path)

    save_json(config_snapshot_path, config_to_dict(train_config))

    result_path.write_text(
        json.dumps(asdict(outcome.record), indent=2, default=str), encoding="utf-8"
    )

    print(f">>> Done. epsilon_upper={outcome.record.epsilon_upper_theory:.4f}, "
          f"accuracy={outcome.record.utility_metrics.get('accuracy', 0):.4f}")
    return outcome.record, train_config


def evaluate_audit(member_scores: list[float], nonmember_scores: list[float]) -> EvalResult:
    """Evaluate an audit attempt. Returns EvalResult."""
    from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound
    from dp_audit_tightness.evaluation.metrics import compute_privacy_tightness_metrics

    record, _ = get_training_record_and_config()
    epsilon_upper = record.epsilon_upper_theory

    estimate = estimate_empirical_lower_bound(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=DELTA,
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
    )

    metrics = compute_privacy_tightness_metrics(
        epsilon_upper_theory=epsilon_upper,
        epsilon_lower_empirical=estimate.epsilon_lower_empirical,
    )

    member_mean = statistics.fmean(member_scores) if member_scores else 0.0
    nonmember_mean = statistics.fmean(nonmember_scores) if nonmember_scores else 0.0

    return EvalResult(
        tightness_ratio=metrics.tightness_ratio,
        epsilon_lower=estimate.epsilon_lower_empirical,
        epsilon_upper=epsilon_upper,
        privacy_loss_gap=metrics.privacy_loss_gap,
        member_score_mean=member_mean,
        nonmember_score_mean=nonmember_mean,
        score_gap=member_mean - nonmember_mean,
        num_member_samples=estimate.num_member_samples,
        num_nonmember_samples=estimate.num_nonmember_samples,
        member_favoring=bool(estimate.member_favoring),
        selected_tpr=estimate.selected_tpr,
        selected_fpr=estimate.selected_fpr,
        wall_seconds=0.0,
    )


def load_model_and_data():
    """Load the cached target model and dataset for experiment.py."""
    import torch
    from dp_audit_tightness.data.datasets import load_dataset_bundle
    from dp_audit_tightness.models.io import load_model_for_inference

    record, config = get_training_record_and_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(config.model, record.model_artifact_path, device=device)
    bundle = load_dataset_bundle(config.dataset, split_seed=record.split_seed)
    return model, bundle, device, record


def train_shadow_model(train_dataset, seed: int, epochs: int = EPOCHS):
    """Train a shadow model on the given dataset. For reference/LiRA attacks.

    Returns the trained model (eval mode, on device).
    This is the expensive helper that justifies GPU.
    """
    import torch
    from dp_audit_tightness.utils.seeds import set_global_seed
    from dp_audit_tightness.training.dp_sgd import train_dp_sgd

    shadow_cache = CACHE_DIR / f"shadow_s{seed}_e{epochs}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check cache
    if shadow_cache.exists():
        from dp_audit_tightness.models.io import load_model_for_inference
        _, config = get_training_record_and_config()
        model = load_model_for_inference(config.model, str(shadow_cache), device=device)
        return model

    # Train fresh shadow model with same architecture + DP params
    config = _build_train_config(epochs=epochs, seed=seed)
    set_global_seed(seed)
    _logger = logging.getLogger("autoresearch.shadow")
    outcome = train_dp_sgd(config=config, logger=_logger)

    # Cache the checkpoint
    if outcome.checkpoint_path and Path(outcome.checkpoint_path).exists():
        import shutil
        shutil.copy2(outcome.checkpoint_path, shadow_cache)

    from dp_audit_tightness.models.io import load_model_for_inference
    model = load_model_for_inference(config.model, str(shadow_cache), device=device)
    return model


def print_results(result: EvalResult):
    """Print results in the format the agent loop parses."""
    print(f"tightness_ratio: {result.tightness_ratio:.6f}")
    print(f"epsilon_lower: {result.epsilon_lower:.6f}")
    print(f"epsilon_upper: {result.epsilon_upper:.6f}")
    print(f"privacy_loss_gap: {result.privacy_loss_gap:.6f}")
    print(f"score_gap: {result.score_gap:.6f}")
    print(f"member_favoring: {result.member_favoring}")
    print(f"wall_seconds: {result.wall_seconds:.1f}")
