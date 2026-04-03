"""Immutable evaluation harness for autoresearch.

DO NOT MODIFY THIS FILE. The agent only modifies experiment.py.

This script:
1. Ensures a pre-trained DP-SGD model exists (trains one if missing).
2. Provides the evaluation function that experiment.py calls.
3. Prints the final metric line that the agent loop parses.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
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
HIDDEN_DIM = 64
NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)
BATCH_SIZE = 256
EPOCHS = 1
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


def get_training_record_and_config():
    """Return (training_record, training_config) from cache, training if needed."""
    from dp_audit_tightness.config import load_train_config, TrainExperimentConfig
    from dp_audit_tightness.training.dp_sgd import train_dp_sgd
    from dp_audit_tightness.utils.results import (
        TrainingRunRecord,
        save_training_result,
        save_json,
        load_training_result,
    )
    from dp_audit_tightness.utils.seeds import set_global_seed
    from dp_audit_tightness.config import config_to_dict

    result_path = CACHE_DIR / "training_result.json"
    config_snapshot_path = CACHE_DIR / "training_config.json"
    checkpoint_path = CACHE_DIR / "model.pt"

    if result_path.exists() and config_snapshot_path.exists():
        record = load_training_result(result_path)
        config = TrainExperimentConfig.from_dict(
            json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        )
        return record, config

    print(">>> No cached model found. Training DP-SGD model (one-time cost)...")
    train_config = load_train_config(ROOT / "configs" / "train" / "mnist_mlp_dp_sgd_smoke.yaml")

    set_global_seed(TRAINING_SEED)
    outcome = train_dp_sgd(config=train_config)

    # Save to cache
    outcome.record.model_artifact_path = str(checkpoint_path)
    if outcome.checkpoint_path and Path(outcome.checkpoint_path).exists():
        import shutil
        shutil.copy2(outcome.checkpoint_path, checkpoint_path)
    outcome.record.config_snapshot_path = str(config_snapshot_path)

    save_json(config_snapshot_path, config_to_dict(train_config))
    save_training_result(outcome.record, results_root=str(CACHE_DIR))

    # Also save a simple JSON at the known path
    result_path.write_text(
        json.dumps(asdict(outcome.record), indent=2, default=str), encoding="utf-8"
    )

    print(f">>> Model trained. epsilon_upper={outcome.record.epsilon_upper_theory:.4f}")
    return outcome.record, train_config


def evaluate_audit(member_scores: list[float], nonmember_scores: list[float]) -> EvalResult:
    """Evaluate an audit attempt. Called by experiment.py with its scores.

    Returns an EvalResult. The agent loop reads the printed tightness_ratio line.
    """
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
    """Load the cached model and dataset bundle for experiment.py to use."""
    import torch
    from dp_audit_tightness.data.datasets import load_dataset_bundle
    from dp_audit_tightness.models.io import load_model_for_inference

    record, config = get_training_record_and_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(config.model, record.model_artifact_path, device=device)
    bundle = load_dataset_bundle(config.dataset, split_seed=record.split_seed)
    return model, bundle, device, record


def print_results(result: EvalResult):
    """Print results in the format the agent loop parses."""
    print(f"tightness_ratio: {result.tightness_ratio:.6f}")
    print(f"epsilon_lower: {result.epsilon_lower:.6f}")
    print(f"epsilon_upper: {result.epsilon_upper:.6f}")
    print(f"privacy_loss_gap: {result.privacy_loss_gap:.6f}")
    print(f"score_gap: {result.score_gap:.6f}")
    print(f"member_favoring: {result.member_favoring}")
    print(f"wall_seconds: {result.wall_seconds:.1f}")
