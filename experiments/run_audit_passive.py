from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.auditing.passive.auditors import run_repeated_passive_audit
from dp_audit_tightness.config import (
    config_to_dict,
    load_passive_audit_config,
    load_train_config_snapshot,
)
from dp_audit_tightness.evaluation.metrics import compute_privacy_tightness_metrics
from dp_audit_tightness.evaluation.saturation import detect_saturation
from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound
from dp_audit_tightness.utils.logging_utils import configure_logging, format_kv_fields
from dp_audit_tightness.utils.paths import build_run_id
from dp_audit_tightness.utils.results import (
    AuditRunRecord,
    PASSIVE_AUDIT_REGIME,
    load_audit_results_for_training,
    load_training_result,
    resolve_training_config_snapshot_path,
    save_audit_result,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one passive final-model-only audit."
    )
    parser.add_argument("--config", required=True, help="Path to a passive-audit YAML config.")
    parser.add_argument(
        "--training-result-path",
        help="Optional override for one training result JSON path.",
    )
    parser.add_argument(
        "--training-result-paths",
        nargs="*",
        help="Optional list of training result JSON paths for a multi-seed sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    config_path = Path(args.config)
    config = load_passive_audit_config(config_path)
    if config.audit_regime != PASSIVE_AUDIT_REGIME:
        raise ValueError(
            f"run_audit_passive.py requires audit_regime={PASSIVE_AUDIT_REGIME}, got {config.audit_regime}."
        )

    training_result_paths = _resolve_training_result_paths(config, args)
    for training_result_path in training_result_paths:
        training_record = load_training_result(training_result_path)
        training_config = load_train_config_snapshot(
            resolve_training_config_snapshot_path(training_result_path, training_record)
        )
        _validate_delta(config.delta, training_record.delta)
        logger.info(
            "Loaded passive audit config %s",
            format_kv_fields(
                training_run_id=training_record.training_run_id,
                auditor_variant=config.auditor_variant,
                query_budget=config.passive.query_budget,
                delta=config.delta,
            ),
        )

        observation = run_repeated_passive_audit(
            training_record=training_record,
            training_config=training_config,
            config=config,
        )
        empirical_estimate = estimate_empirical_lower_bound(
            member_scores=observation.member_scores,
            nonmember_scores=observation.nonmember_scores,
            delta=config.delta,
            align_event_to_score_direction=True,
            require_member_favoring=True,
        )
        epsilon_upper_theory = observation.epsilon_upper_theory or training_record.epsilon_upper_theory
        utility_metrics = observation.utility_metrics or training_record.utility_metrics
        tightness_metrics = compute_privacy_tightness_metrics(
            epsilon_upper_theory=epsilon_upper_theory,
            epsilon_lower_empirical=empirical_estimate.epsilon_lower_empirical,
        )

        prior_records = load_audit_results_for_training(
            results_root=config.run.results_root,
            audit_regime=config.audit_regime,
            training_run_id=training_record.training_run_id,
        )
        prior_history = [
            {
                "epsilon_lower_empirical": record.epsilon_lower_empirical,
                "empirical_ci_lower": record.empirical_ci_lower,
                "empirical_ci_upper": record.empirical_ci_upper,
                "tightness_ratio": record.tightness_ratio,
            }
            for record in prior_records
            if record.auditor_strength_rank < config.auditor_strength_rank
        ]
        current_history_entry = {
            "epsilon_lower_empirical": empirical_estimate.epsilon_lower_empirical,
            "empirical_ci_lower": empirical_estimate.empirical_ci_lower,
            "empirical_ci_upper": empirical_estimate.empirical_ci_upper,
            "tightness_ratio": tightness_metrics.tightness_ratio,
        }
        saturation_decision = detect_saturation(
            history=prior_history + [current_history_entry],
            config=config.saturation,
        )

        audit_run_id = build_run_id(
            prefix="passive_audit",
            descriptor=f"seed{training_record.training_seed}_{config.auditor_variant}",
        )
        audit_record = AuditRunRecord(
            audit_run_id=audit_run_id,
            training_run_id=training_record.training_run_id,
            dataset=training_record.dataset,
            split_seed=training_record.split_seed,
            training_seed=training_record.training_seed,
            model_name=training_record.model_name,
            optimizer_name=training_record.optimizer_name,
            clipping_norm=training_record.clipping_norm,
            noise_multiplier=training_record.noise_multiplier,
            batch_size=training_record.batch_size,
            epochs=training_record.epochs,
            sampling_rate=training_record.sampling_rate,
            delta=config.delta,
            epsilon_upper_theory=epsilon_upper_theory,
            utility_metrics=utility_metrics,
            audit_regime=config.audit_regime,
            auditor_variant=config.auditor_variant,
            auditor_strength_rank=config.auditor_strength_rank,
            repeated_seeds=config.repeated_seeds,
            epsilon_lower_empirical=empirical_estimate.epsilon_lower_empirical,
            empirical_ci_lower=empirical_estimate.empirical_ci_lower,
            empirical_ci_upper=empirical_estimate.empirical_ci_upper,
            privacy_loss_gap=tightness_metrics.privacy_loss_gap,
            tightness_ratio=tightness_metrics.tightness_ratio,
            saturation_detected=saturation_decision.saturation_detected,
            saturation_reason=saturation_decision.reason,
            audit_mode="repeated_seeds" if len(config.repeated_seeds) > 1 else "single_seed",
            audit_metadata={
                "training_result_path": str(training_result_path),
                "raw_statistics": observation.raw_statistics,
                "artifact_payload": observation.artifact_payload,
                "estimation_method": empirical_estimate.estimation_method,
                "selected_threshold": empirical_estimate.selected_threshold,
                "selected_event": empirical_estimate.selected_event,
                "selected_event_direction": empirical_estimate.selected_event_direction,
                "selected_tpr": empirical_estimate.selected_tpr,
                "selected_fpr": empirical_estimate.selected_fpr,
                "member_event_fraction": empirical_estimate.member_event_fraction,
                "nonmember_event_fraction": empirical_estimate.nonmember_event_fraction,
                "member_favoring": empirical_estimate.member_favoring,
                "no_member_favoring_event_found": empirical_estimate.no_member_favoring_event_found,
                "warning": empirical_estimate.warning_message,
                "audited_model_run_id": observation.audited_model_run_id,
            },
        )
        audit_result_path = save_audit_result(audit_record, results_root=config.run.results_root)
        config_snapshot_path = save_json(
            Path(config.run.results_root) / "audits" / "passive" / "configs" / f"{audit_run_id}_config.json",
            config_to_dict(config),
        )

        if config.run.save_debug_artifacts:
            passive_artifact_path = save_json(
                Path(config.run.results_root)
                / "audits"
                / "passive"
                / "artifacts"
                / f"{audit_run_id}_passive_debug.json",
                {
                    "training_run_id": training_record.training_run_id,
                    "auditor_variant": config.auditor_variant,
                    "training_result_path": str(training_result_path),
                    "score_name": observation.score_name,
                    "member_scores": observation.member_scores,
                    "nonmember_scores": observation.nonmember_scores,
                    "artifact_payload": observation.artifact_payload,
                },
            )
            logger.info("Saved passive audit artifact to %s", passive_artifact_path)

        logger.info("Saved passive audit result to %s", audit_result_path)
        logger.info("Saved config snapshot to %s", config_snapshot_path)
        logger.info(
            "Computed privacy comparison %s",
            format_kv_fields(
                epsilon_upper_theory=epsilon_upper_theory,
                epsilon_lower_empirical=empirical_estimate.epsilon_lower_empirical,
                privacy_loss_gap=tightness_metrics.privacy_loss_gap,
                tightness_ratio=tightness_metrics.tightness_ratio,
                member_favoring=empirical_estimate.member_favoring,
                no_member_favoring_event_found=empirical_estimate.no_member_favoring_event_found,
                saturation_detected=saturation_decision.saturation_detected,
            ),
        )


def _resolve_training_result_paths(config, args) -> list[Path]:
    if args.training_result_paths:
        return [Path(path) for path in args.training_result_paths]
    if args.training_result_path:
        return [Path(args.training_result_path)]
    if config.run.training_result_paths:
        return [Path(path) for path in config.run.training_result_paths]
    return [Path(config.training_result_path)]


def _validate_delta(audit_delta: float, training_delta: float) -> None:
    if abs(audit_delta - training_delta) > 1e-12:
        raise ValueError(
            f"Audit delta ({audit_delta}) does not match training delta ({training_delta})."
        )


if __name__ == "__main__":
    main()
