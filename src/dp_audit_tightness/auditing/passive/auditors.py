from __future__ import annotations

import random
import statistics

from dp_audit_tightness.auditing.base import AuditObservation
from dp_audit_tightness.auditing.passive.calibration import (
    apply_temperature,
    fit_temperature,
    percentile_rank_transform,
    zscore_with_reference,
)
from dp_audit_tightness.config import PassiveAuditConfig, TrainExperimentConfig
from dp_audit_tightness.data.datasets import load_dataset_bundle
from dp_audit_tightness.models.io import load_model_for_inference
from dp_audit_tightness.utils.results import TrainingRunRecord


def run_passive_audit_once(
    training_record: TrainingRunRecord,
    training_config: TrainExperimentConfig,
    config: PassiveAuditConfig,
    seed: int,
    *,
    dataset_bundle=None,
    model=None,
) -> AuditObservation:
    """Run one passive final-model-only audit using model outputs only."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for passive auditing.") from exc

    active_bundle = dataset_bundle or load_dataset_bundle(
        training_config.dataset,
        split_seed=training_record.split_seed,
    )
    if training_record.model_artifact_path is None:
        raise ValueError("Passive auditing requires a saved model artifact path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_model = model or load_model_for_inference(
        training_config.model,
        training_record.model_artifact_path,
        device=device,
    )
    rng = random.Random(seed)

    calibration_budget = int(config.passive.query_budget * config.passive.calibration_fraction)
    member_budget = max(1, config.passive.query_budget // 2)
    nonmember_budget = max(1, config.passive.query_budget - member_budget)

    member_indices = _sample_indices(len(active_bundle.train_dataset), member_budget, rng)
    nonmember_indices = _sample_indices(len(active_bundle.eval_dataset), nonmember_budget, rng)
    nonmember_index_set = set(nonmember_indices)
    remaining_eval_indices = [
        index for index in range(len(active_bundle.eval_dataset)) if index not in nonmember_index_set
    ]
    calibration_indices = (
        _sample_indices(len(remaining_eval_indices), calibration_budget, rng, values=remaining_eval_indices)
        if calibration_budget > 0
        else []
    )

    member_logits, member_labels = _collect_logits_and_labels(
        active_model,
        active_bundle.train_dataset,
        member_indices,
        device=device,
        batch_size=training_record.batch_size,
    )
    nonmember_logits, nonmember_labels = _collect_logits_and_labels(
        active_model,
        active_bundle.eval_dataset,
        nonmember_indices,
        device=device,
        batch_size=training_record.batch_size,
    )
    calibration_logits = calibration_labels = None
    if calibration_indices:
        calibration_logits, calibration_labels = _collect_logits_and_labels(
            active_model,
            active_bundle.eval_dataset,
            calibration_indices,
            device=device,
            batch_size=training_record.batch_size,
        )

    temperature = 1.0
    if config.passive.calibration_method == "temperature_scaling" and calibration_logits is not None:
        temperature = fit_temperature(calibration_logits, calibration_labels)
        member_logits = apply_temperature(member_logits, temperature)
        nonmember_logits = apply_temperature(nonmember_logits, temperature)
        calibration_logits = apply_temperature(calibration_logits, temperature)

    member_scores, nonmember_scores, calibration_reference = _build_membership_scores(
        member_logits=member_logits,
        member_labels=member_labels,
        nonmember_logits=nonmember_logits,
        nonmember_labels=nonmember_labels,
        calibration_logits=calibration_logits,
        calibration_labels=calibration_labels,
        score_type=config.passive.score_type,
        calibration_method=config.passive.calibration_method,
    )
    return AuditObservation(
        audit_regime=config.audit_regime,
        auditor_variant=config.auditor_variant,
        raw_statistics={
            "seed": float(seed),
            "query_budget": float(config.passive.query_budget),
            "calibration_budget": float(len(calibration_indices)),
            "member_score_mean": float(statistics.fmean(member_scores)),
            "nonmember_score_mean": float(statistics.fmean(nonmember_scores)),
            "score_mean_gap": float(statistics.fmean(member_scores) - statistics.fmean(nonmember_scores)),
            "temperature": float(temperature),
        },
        artifact_payload={
            "score_type": config.passive.score_type,
            "calibration_method": config.passive.calibration_method,
            "calibration_fraction": config.passive.calibration_fraction,
            "calibration_reference_size": len(calibration_reference),
        },
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        score_name=config.passive.score_type,
        epsilon_upper_theory=training_record.epsilon_upper_theory,
        utility_metrics=training_record.utility_metrics,
        audited_model_run_id=training_record.training_run_id,
    )


def run_repeated_passive_audit(
    training_record: TrainingRunRecord,
    training_config: TrainExperimentConfig,
    config: PassiveAuditConfig,
) -> AuditObservation:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for passive auditing.") from exc

    dataset_bundle = load_dataset_bundle(training_config.dataset, split_seed=training_record.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(
        training_config.model,
        training_record.model_artifact_path,
        device=device,
    )
    observations = [
        run_passive_audit_once(
            training_record=training_record,
            training_config=training_config,
            config=config,
            seed=seed,
            dataset_bundle=dataset_bundle,
            model=model,
        )
        for seed in config.repeated_seeds
    ]
    member_scores = [score for observation in observations for score in observation.member_scores]
    nonmember_scores = [score for observation in observations for score in observation.nonmember_scores]
    return AuditObservation(
        audit_regime=config.audit_regime,
        auditor_variant=config.auditor_variant,
        raw_statistics={
            "num_runs": float(len(observations)),
            "member_score_mean": float(statistics.fmean(member_scores)),
            "nonmember_score_mean": float(statistics.fmean(nonmember_scores)),
            "score_mean_gap": float(statistics.fmean(member_scores) - statistics.fmean(nonmember_scores)),
        },
        artifact_payload={
            "score_type": config.passive.score_type,
            "calibration_method": config.passive.calibration_method,
            "per_seed_runs": [
                {
                    "seed": seed,
                    "member_score_mean": statistics.fmean(observation.member_scores),
                    "nonmember_score_mean": statistics.fmean(observation.nonmember_scores),
                    "score_mean_gap": statistics.fmean(observation.member_scores)
                    - statistics.fmean(observation.nonmember_scores),
                }
                for seed, observation in zip(config.repeated_seeds, observations, strict=False)
            ],
        },
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        score_name=observations[0].score_name if observations else None,
        epsilon_upper_theory=training_record.epsilon_upper_theory,
        utility_metrics=training_record.utility_metrics,
        audited_model_run_id=training_record.training_run_id,
    )


def _sample_indices(total_size: int, sample_size: int, rng: random.Random, *, values=None) -> list[int]:
    population = list(range(total_size)) if values is None else list(values)
    sample_size = min(sample_size, len(population))
    return rng.sample(population, sample_size)


def _collect_logits_and_labels(model, dataset, indices, *, device, batch_size: int):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for passive auditing.") from exc

    if not indices:
        raise ValueError("At least one index is required to collect model outputs.")

    logits_batches = []
    label_batches = []
    with torch.no_grad():
        for start in range(0, len(indices), max(1, batch_size)):
            batch_indices = indices[start : start + max(1, batch_size)]
            images = []
            labels = []
            for index in batch_indices:
                image, label = dataset[index]
                images.append(image)
                labels.append(label)
            image_tensor = torch.stack(images).to(device)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            logits_batches.append(model(image_tensor))
            label_batches.append(label_tensor)

    return torch.cat(logits_batches, dim=0), torch.cat(label_batches, dim=0)


def _build_membership_scores(
    *,
    member_logits,
    member_labels,
    nonmember_logits,
    nonmember_labels,
    calibration_logits,
    calibration_labels,
    score_type: str,
    calibration_method: str,
) -> tuple[list[float], list[float], list[float]]:
    member_components = _score_components(member_logits, member_labels)
    nonmember_components = _score_components(nonmember_logits, nonmember_labels)
    if calibration_logits is not None and calibration_labels is not None:
        calibration_components = _score_components(calibration_logits, calibration_labels)
    else:
        calibration_components = nonmember_components

    if score_type == "negative_loss":
        member_scores = member_components["negative_loss"]
        nonmember_scores = nonmember_components["negative_loss"]
        calibration_reference = calibration_components["negative_loss"]
    elif score_type == "max_probability":
        member_scores = member_components["max_probability"]
        nonmember_scores = nonmember_components["max_probability"]
        calibration_reference = calibration_components["max_probability"]
    elif score_type == "logit_margin":
        member_scores = member_components["logit_margin"]
        nonmember_scores = nonmember_components["logit_margin"]
        calibration_reference = calibration_components["logit_margin"]
    elif score_type == "score_fusion":
        member_scores, nonmember_scores, calibration_reference = _score_fusion(
            member_components,
            nonmember_components,
            calibration_components,
            calibration_method=calibration_method,
        )
    else:
        raise NotImplementedError(f"Unsupported passive score_type: {score_type}")

    return (
        member_scores.detach().cpu().tolist(),
        nonmember_scores.detach().cpu().tolist(),
        calibration_reference.detach().cpu().tolist(),
    )


def _score_components(logits, labels):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for passive auditing.") from exc

    probabilities = torch.softmax(logits, dim=1)
    top_two = probabilities.topk(k=2, dim=1).values
    max_probability = top_two[:, 0]
    logit_margin = top_two[:, 0] - top_two[:, 1]
    negative_loss = -torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    return {
        "negative_loss": negative_loss,
        "max_probability": max_probability,
        "logit_margin": logit_margin,
    }


def _score_fusion(member_components, nonmember_components, calibration_components, *, calibration_method: str):
    if calibration_method == "isotonic_like":
        member_scores = _average_tensors(
            percentile_rank_transform(member_components["negative_loss"], calibration_components["negative_loss"]),
            percentile_rank_transform(member_components["max_probability"], calibration_components["max_probability"]),
            percentile_rank_transform(member_components["logit_margin"], calibration_components["logit_margin"]),
        )
        nonmember_scores = _average_tensors(
            percentile_rank_transform(nonmember_components["negative_loss"], calibration_components["negative_loss"]),
            percentile_rank_transform(nonmember_components["max_probability"], calibration_components["max_probability"]),
            percentile_rank_transform(nonmember_components["logit_margin"], calibration_components["logit_margin"]),
        )
        calibration_reference = _average_tensors(
            percentile_rank_transform(calibration_components["negative_loss"], calibration_components["negative_loss"]),
            percentile_rank_transform(calibration_components["max_probability"], calibration_components["max_probability"]),
            percentile_rank_transform(calibration_components["logit_margin"], calibration_components["logit_margin"]),
        )
        return member_scores, nonmember_scores, calibration_reference

    member_scores = _average_tensors(
        zscore_with_reference(member_components["negative_loss"], calibration_components["negative_loss"]),
        zscore_with_reference(member_components["max_probability"], calibration_components["max_probability"]),
        zscore_with_reference(member_components["logit_margin"], calibration_components["logit_margin"]),
    )
    nonmember_scores = _average_tensors(
        zscore_with_reference(nonmember_components["negative_loss"], calibration_components["negative_loss"]),
        zscore_with_reference(nonmember_components["max_probability"], calibration_components["max_probability"]),
        zscore_with_reference(nonmember_components["logit_margin"], calibration_components["logit_margin"]),
    )
    calibration_reference = _average_tensors(
        zscore_with_reference(calibration_components["negative_loss"], calibration_components["negative_loss"]),
        zscore_with_reference(calibration_components["max_probability"], calibration_components["max_probability"]),
        zscore_with_reference(calibration_components["logit_margin"], calibration_components["logit_margin"]),
    )
    return member_scores, nonmember_scores, calibration_reference


def _average_tensors(*tensors):
    return sum(tensors) / len(tensors)
