from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import csv
import json

from dp_audit_tightness.utils.paths import ensure_directory
from dp_audit_tightness.utils.validation import (
    validate_audit_run_payload,
    validate_summary_payload,
    validate_training_run_payload,
)


CANARY_AUDIT_REGIME = "evaluator_controlled_canary_stress_test"
PASSIVE_AUDIT_REGIME = "passive_final_model_only"


@dataclass(slots=True)
class TrainingRunRecord:
    training_run_id: str
    experiment_name: str
    dataset: str
    split_seed: int
    training_seed: int
    model_name: str
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    clipping_norm: float
    noise_multiplier: float
    batch_size: int
    epochs: int
    sampling_rate: float
    delta: float
    epsilon_upper_theory: float
    utility_metrics: dict[str, float]
    model_artifact_path: str | None
    epsilon_upper_pld: float | None = None
    pld_accounting_backend: str | None = None
    config_snapshot_path: str | None = None
    created_at_utc: str = field(default_factory=lambda: _utc_now_iso())
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingRunRecord":
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        flat = {key: value for key, value in asdict(self).items() if key != "utility_metrics"}
        for key, value in self.utility_metrics.items():
            flat[f"utility_{key}"] = value
        return flat


@dataclass(slots=True)
class AuditRunRecord:
    audit_run_id: str
    training_run_id: str
    dataset: str
    split_seed: int
    training_seed: int
    model_name: str
    optimizer_name: str
    clipping_norm: float
    noise_multiplier: float
    batch_size: int
    epochs: int
    sampling_rate: float
    delta: float
    epsilon_upper_theory: float
    utility_metrics: dict[str, float]
    audit_regime: str
    auditor_variant: str
    auditor_strength_rank: int
    repeated_seeds: list[int]
    epsilon_lower_empirical: float
    empirical_ci_lower: float | None
    empirical_ci_upper: float | None
    privacy_loss_gap: float
    tightness_ratio: float | None
    saturation_detected: bool
    saturation_reason: str | None
    audit_mode: str | None = None
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: _utc_now_iso())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditRunRecord":
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        flat = {
            key: value
            for key, value in asdict(self).items()
            if key not in {"utility_metrics", "audit_metadata", "repeated_seeds"}
        }
        for key, value in self.utility_metrics.items():
            flat[f"utility_{key}"] = value
        flat["repeated_seeds"] = json.dumps(self.repeated_seeds)
        flat["audit_metadata"] = json.dumps(self.audit_metadata, sort_keys=True)
        return flat


@dataclass(slots=True)
class AggregatedSummaryRecord:
    summary_id: str
    dataset: str
    model_name: str
    audit_regime: str
    auditor_variant: str
    num_runs: int
    mean_epsilon_upper_theory: float
    mean_epsilon_lower_empirical: float
    std_epsilon_lower_empirical: float
    mean_privacy_loss_gap: float
    mean_tightness_ratio: float | None
    saturation_rate: float
    created_at_utc: str = field(default_factory=lambda: _utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_training_result(record: TrainingRunRecord, results_root: str | Path) -> Path:
    root = Path(results_root)
    validate_training_run_payload(record.to_dict())
    json_path = root / "training" / f"{record.training_run_id}.json"
    save_json(json_path, record.to_dict())
    append_csv_row(root / "training" / "training_runs.csv", record.to_flat_dict())
    return json_path


def save_audit_result(record: AuditRunRecord, results_root: str | Path) -> Path:
    root = Path(results_root)
    validate_audit_run_payload(record.to_dict())
    subdirectory = _audit_subdirectory(record.audit_regime)
    json_path = root / "audits" / subdirectory / f"{record.audit_run_id}.json"
    save_json(json_path, record.to_dict())
    append_csv_row(root / "audits" / subdirectory / "audit_runs.csv", record.to_flat_dict())
    return json_path


def save_summary_result(record: AggregatedSummaryRecord, results_root: str | Path) -> Path:
    root = Path(results_root)
    validate_summary_payload(record.to_dict())
    json_path = root / "summaries" / f"{record.summary_id}.json"
    save_json(json_path, record.to_dict())
    append_csv_row(root / "summaries" / "aggregated_summaries.csv", record.to_flat_dict())
    return json_path


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2, sort_keys=True)
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_training_result(path: str | Path) -> TrainingRunRecord:
    payload = load_json(path)
    validate_training_run_payload(payload)
    return TrainingRunRecord.from_dict(payload)


def load_audit_result(path: str | Path) -> AuditRunRecord:
    payload = load_json(path)
    validate_audit_run_payload(payload)
    return AuditRunRecord.from_dict(payload)


def load_audit_results_for_training(
    results_root: str | Path,
    audit_regime: str,
    training_run_id: str,
) -> list[AuditRunRecord]:
    records: list[AuditRunRecord] = []
    audit_dir = Path(results_root) / "audits" / _audit_subdirectory(audit_regime)
    if not audit_dir.exists():
        return records
    for path in audit_dir.glob("*.json"):
        payload = load_json(path)
        validate_audit_run_payload(payload)
        if payload.get("training_run_id") == training_run_id:
            records.append(AuditRunRecord.from_dict(payload))
    return sorted(records, key=lambda record: (record.auditor_strength_rank, record.created_at_utc))


def discover_training_result_paths(results_root: str | Path) -> list[Path]:
    return sorted((Path(results_root) / "training").glob("*.json"))


def discover_audit_result_paths(results_root: str | Path) -> list[Path]:
    root = Path(results_root)
    return sorted((root / "audits" / "canary").glob("*.json")) + sorted(
        (root / "audits" / "passive").glob("*.json")
    )


def resolve_training_config_snapshot_path(
    training_result_path: str | Path,
    training_record: TrainingRunRecord,
) -> Path:
    if training_record.config_snapshot_path:
        return Path(training_record.config_snapshot_path)
    result_path = Path(training_result_path)
    return result_path.parent / "configs" / f"{training_record.training_run_id}_config.json"


def append_csv_row(path: str | Path, row: Mapping[str, Any]) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    serialized_row = {key: _csv_value(value) for key, value in row.items()}
    write_header = not target.exists()
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(serialized_row)
    return target


def _audit_subdirectory(audit_regime: str) -> str:
    if audit_regime == CANARY_AUDIT_REGIME:
        return "canary"
    if audit_regime == PASSIVE_AUDIT_REGIME:
        return "passive"
    raise ValueError(f"Unknown audit_regime: {audit_regime}")


def _to_serializable(payload: Any) -> Any:
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _to_serializable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_to_serializable(value) for value in payload]
    return payload


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
