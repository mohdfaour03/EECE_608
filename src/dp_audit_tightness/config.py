from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class DatasetConfig:
    name: str
    data_dir: str = "data/raw"
    validation_fraction: float = 0.1
    num_workers: int = 0


@dataclass(slots=True)
class ModelConfig:
    name: str
    input_dim: int
    hidden_dim: int
    num_classes: int


@dataclass(slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float = 0.0
    momentum: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    eval_batch_size: int
    epochs: int
    clipping_norm: float
    noise_multiplier: float
    optimizer: OptimizerConfig
    sampling_rate: float | None = None


@dataclass(slots=True)
class PrivacyConfig:
    delta: float
    accountant: str = "rdp"


@dataclass(slots=True)
class RunConfig:
    split_seed: int = 0
    training_seed: int = 0
    split_seeds: list[int] | None = None
    training_seeds: list[int] | None = None
    results_root: str = "results"
    save_checkpoint: bool = True
    save_debug_artifacts: bool = True
    notes: str | None = None


@dataclass(slots=True)
class TrainExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    privacy: PrivacyConfig
    run: RunConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainExperimentConfig":
        run_payload = dict(payload["run"])
        run_payload["split_seeds"] = _optional_int_list(run_payload.get("split_seeds"))
        run_payload["training_seeds"] = _optional_int_list(run_payload.get("training_seeds"))
        return cls(
            experiment_name=str(payload["experiment_name"]),
            dataset=DatasetConfig(**payload["dataset"]),
            model=ModelConfig(**payload["model"]),
            training=TrainingConfig(
                batch_size=int(payload["training"]["batch_size"]),
                eval_batch_size=int(payload["training"]["eval_batch_size"]),
                epochs=int(payload["training"]["epochs"]),
                clipping_norm=float(payload["training"]["clipping_norm"]),
                noise_multiplier=float(payload["training"]["noise_multiplier"]),
                sampling_rate=_optional_float(payload["training"].get("sampling_rate")),
                optimizer=OptimizerConfig(**payload["training"]["optimizer"]),
            ),
            privacy=PrivacyConfig(**payload["privacy"]),
            run=RunConfig(**run_payload),
        )


@dataclass(slots=True)
class CanaryConfig:
    strategy: str
    num_canaries: int
    insertion_rate: float
    optimize_steps: int = 0


@dataclass(slots=True)
class PassiveAuditorConfig:
    query_budget: int
    score_type: str
    calibration_method: str
    calibration_fraction: float = 0.0


@dataclass(slots=True)
class SaturationConfig:
    min_absolute_improvement: float = 0.05
    min_relative_improvement: float = 0.02
    tightness_ratio_tolerance: float = 0.01
    use_confidence_interval_overlap: bool = True


@dataclass(slots=True)
class AuditRunConfig:
    results_root: str = "results"
    save_debug_artifacts: bool = True
    training_result_paths: list[str] | None = None


@dataclass(slots=True)
class CanaryAuditConfig:
    training_result_path: str
    audit_regime: str
    auditor_variant: str
    auditor_strength_rank: int
    audit_mode: str
    delta: float
    repeated_seeds: list[int]
    canary: CanaryConfig
    saturation: SaturationConfig
    run: AuditRunConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanaryAuditConfig":
        run_payload = dict(payload["run"])
        run_payload["training_result_paths"] = _optional_string_list(run_payload.get("training_result_paths"))
        return cls(
            training_result_path=str(payload["training_result_path"]),
            audit_regime=str(payload["audit_regime"]),
            auditor_variant=str(payload["auditor_variant"]),
            auditor_strength_rank=int(payload["auditor_strength_rank"]),
            audit_mode=str(payload["audit_mode"]),
            delta=float(payload["delta"]),
            repeated_seeds=[int(seed) for seed in payload.get("repeated_seeds", [])],
            canary=CanaryConfig(**payload["canary"]),
            saturation=SaturationConfig(**payload["saturation"]),
            run=AuditRunConfig(**run_payload),
        )


@dataclass(slots=True)
class PassiveAuditConfig:
    training_result_path: str
    audit_regime: str
    auditor_variant: str
    auditor_strength_rank: int
    delta: float
    repeated_seeds: list[int]
    passive: PassiveAuditorConfig
    saturation: SaturationConfig
    run: AuditRunConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PassiveAuditConfig":
        run_payload = dict(payload["run"])
        run_payload["training_result_paths"] = _optional_string_list(run_payload.get("training_result_paths"))
        return cls(
            training_result_path=str(payload["training_result_path"]),
            audit_regime=str(payload["audit_regime"]),
            auditor_variant=str(payload["auditor_variant"]),
            auditor_strength_rank=int(payload["auditor_strength_rank"]),
            delta=float(payload["delta"]),
            repeated_seeds=[int(seed) for seed in payload.get("repeated_seeds", [])],
            passive=PassiveAuditorConfig(**payload["passive"]),
            saturation=SaturationConfig(**payload["saturation"]),
            run=AuditRunConfig(**run_payload),
        )


def load_train_config(path: str | Path) -> TrainExperimentConfig:
    return TrainExperimentConfig.from_dict(_load_yaml(path))


def load_train_config_snapshot(path: str | Path) -> TrainExperimentConfig:
    return TrainExperimentConfig.from_dict(_load_json(path))


def load_canary_audit_config(path: str | Path) -> CanaryAuditConfig:
    return CanaryAuditConfig.from_dict(_load_yaml(path))


def load_passive_audit_config(path: str | Path) -> PassiveAuditConfig:
    return PassiveAuditConfig.from_dict(_load_yaml(path))


def config_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return payload


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(item) for item in value]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in config snapshot: {path}")
    return payload
