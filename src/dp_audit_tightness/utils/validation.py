from __future__ import annotations

from typing import Any, Mapping


_TRAINING_REQUIRED_KEYS = {
    "training_run_id",
    "experiment_name",
    "dataset",
    "split_seed",
    "training_seed",
    "model_name",
    "optimizer_name",
    "learning_rate",
    "weight_decay",
    "clipping_norm",
    "noise_multiplier",
    "batch_size",
    "epochs",
    "sampling_rate",
    "delta",
    "epsilon_upper_theory",
    "utility_metrics",
    "model_artifact_path",
    "created_at_utc",
}

_AUDIT_REQUIRED_KEYS = {
    "audit_run_id",
    "training_run_id",
    "dataset",
    "split_seed",
    "training_seed",
    "model_name",
    "optimizer_name",
    "clipping_norm",
    "noise_multiplier",
    "batch_size",
    "epochs",
    "sampling_rate",
    "delta",
    "epsilon_upper_theory",
    "utility_metrics",
    "audit_regime",
    "auditor_variant",
    "auditor_strength_rank",
    "repeated_seeds",
    "epsilon_lower_empirical",
    "privacy_loss_gap",
    "tightness_ratio",
    "saturation_detected",
    "created_at_utc",
}

_SUMMARY_REQUIRED_KEYS = {
    "summary_id",
    "dataset",
    "model_name",
    "audit_regime",
    "auditor_variant",
    "num_runs",
    "mean_epsilon_upper_theory",
    "mean_epsilon_lower_empirical",
    "std_epsilon_lower_empirical",
    "mean_privacy_loss_gap",
    "mean_tightness_ratio",
    "saturation_rate",
    "created_at_utc",
}


def validate_training_run_payload(payload: Mapping[str, Any]) -> None:
    _validate_required_keys(payload, _TRAINING_REQUIRED_KEYS, "training result")
    _validate_mapping_field(payload, "utility_metrics", "training result")


def validate_audit_run_payload(payload: Mapping[str, Any]) -> None:
    _validate_required_keys(payload, _AUDIT_REQUIRED_KEYS, "audit result")
    _validate_mapping_field(payload, "utility_metrics", "audit result")


def validate_summary_payload(payload: Mapping[str, Any]) -> None:
    _validate_required_keys(payload, _SUMMARY_REQUIRED_KEYS, "summary result")


def _validate_required_keys(
    payload: Mapping[str, Any],
    required_keys: set[str],
    artifact_name: str,
) -> None:
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise ValueError(
            f"{artifact_name} is missing required keys: {', '.join(missing)}"
        )


def _validate_mapping_field(payload: Mapping[str, Any], key: str, artifact_name: str) -> None:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{artifact_name} field '{key}' must be a mapping.")
