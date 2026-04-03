from __future__ import annotations

import logging
import statistics

from dp_audit_tightness.auditing.base import AuditObservation
from dp_audit_tightness.auditing.canary.one_run import run_one_run_canary_audit
from dp_audit_tightness.auditing.canary.seeding import build_canary_seed_plan
from dp_audit_tightness.config import CanaryAuditConfig, TrainExperimentConfig
from dp_audit_tightness.utils.results import TrainingRunRecord


def run_repeated_canary_audit(
    training_record: TrainingRunRecord,
    training_config: TrainExperimentConfig,
    config: CanaryAuditConfig,
    *,
    logger: logging.Logger | None = None,
) -> AuditObservation:
    observations = [
        run_one_run_canary_audit(
            training_record=training_record,
            training_config=training_config,
            config=config,
            audit_seed=seed,
            logger=logger,
        )
        for seed in config.repeated_seeds
    ]
    member_scores = [
        score
        for observation in observations
        for score in observation.member_scores
    ]
    nonmember_scores = [
        score
        for observation in observations
        for score in observation.nonmember_scores
    ]
    epsilon_values = [
        observation.epsilon_upper_theory
        for observation in observations
        if observation.epsilon_upper_theory is not None
    ]
    utility_keys = {
        key
        for observation in observations
        for key in observation.utility_metrics.keys()
    }
    averaged_utility_metrics = {
        key: statistics.fmean(
            observation.utility_metrics[key]
            for observation in observations
            if key in observation.utility_metrics
        )
        for key in utility_keys
    }
    return AuditObservation(
        audit_regime=config.audit_regime,
        auditor_variant=config.auditor_variant,
        raw_statistics={
            "experiment_seed": float(training_record.training_seed),
            "num_runs": float(len(observations)),
            "member_score_mean": float(statistics.fmean(member_scores)),
            "nonmember_score_mean": float(statistics.fmean(nonmember_scores)),
            "score_mean_gap": float(statistics.fmean(member_scores) - statistics.fmean(nonmember_scores)),
        },
        artifact_payload={
            "audit_mode": "repeated_run",
            "experiment_seed": training_record.training_seed,
            "repeated_audit_seeds": list(config.repeated_seeds),
            "derived_seed_plans": [
                build_canary_seed_plan(
                    experiment_seed=training_record.training_seed,
                    audit_seed=seed,
                    dataset_split_seed=training_record.split_seed,
                ).to_dict()
                for seed in config.repeated_seeds
            ],
            "per_seed_runs": [
                {
                    "audit_seed": seed,
                    "seed_plan": build_canary_seed_plan(
                        experiment_seed=training_record.training_seed,
                        audit_seed=seed,
                        dataset_split_seed=training_record.split_seed,
                    ).to_dict(),
                    "audited_model_run_id": observation.audited_model_run_id,
                    "epsilon_upper_theory": observation.epsilon_upper_theory,
                    "utility_metrics": observation.utility_metrics,
                    "member_score_mean": statistics.fmean(observation.member_scores),
                    "nonmember_score_mean": statistics.fmean(observation.nonmember_scores),
                }
                for seed, observation in zip(config.repeated_seeds, observations, strict=False)
            ],
        },
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        score_name=observations[0].score_name if observations else None,
        epsilon_upper_theory=statistics.fmean(epsilon_values) if epsilon_values else None,
        utility_metrics=averaged_utility_metrics,
    )
