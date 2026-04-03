from __future__ import annotations

from dataclasses import replace
import logging
import statistics

from dp_audit_tightness.auditing.base import AuditObservation
from dp_audit_tightness.auditing.canary.generation import generate_canaries, insert_canaries_into_dataset
from dp_audit_tightness.auditing.canary.seeding import build_canary_seed_plan
from dp_audit_tightness.config import CanaryAuditConfig, TrainExperimentConfig
from dp_audit_tightness.data.datasets import DatasetBundle, load_dataset_bundle
from dp_audit_tightness.models.io import load_model_from_state_dict
from dp_audit_tightness.training.dp_sgd import train_dp_sgd
from dp_audit_tightness.utils.results import TrainingRunRecord
from dp_audit_tightness.utils.seeds import set_global_seed


def run_one_run_canary_audit(
    training_record: TrainingRunRecord,
    training_config: TrainExperimentConfig,
    config: CanaryAuditConfig,
    audit_seed: int,
    *,
    logger: logging.Logger | None = None,
) -> AuditObservation:
    """Run one evaluator-controlled canary stress test.

    First-pass implementation:
    - create evaluator-designed canary inputs
    - insert them into the training set
    - retrain a DP-SGD model under the same hyperparameters
    - compare inserted-canary scores to held-out decoy-canary scores
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for canary auditing.") from exc

    active_logger = logger or logging.getLogger("dp_audit_tightness.canary")
    seed_plan = build_canary_seed_plan(
        experiment_seed=training_record.training_seed,
        audit_seed=audit_seed,
        dataset_split_seed=training_record.split_seed,
    )
    base_bundle = load_dataset_bundle(training_config.dataset, split_seed=training_record.split_seed)
    image_shape = _infer_image_shape(training_config.dataset.name)
    canaries = generate_canaries(
        config.canary,
        seed=seed_plan.canary_generation_seed,
        num_classes=training_config.model.num_classes,
        image_shape=image_shape,
    )
    insertion_result = insert_canaries_into_dataset(
        base_bundle.train_dataset,
        canaries,
        insertion_rate=config.canary.insertion_rate,
        seed=seed_plan.canary_insertion_seed,
    )
    augmented_bundle = DatasetBundle(
        train_dataset=insertion_result.augmented_train_dataset,
        eval_dataset=base_bundle.eval_dataset,
        input_dim=base_bundle.input_dim,
        num_classes=base_bundle.num_classes,
        train_size=base_bundle.train_size + insertion_result.inserted_example_count,
        eval_size=base_bundle.eval_size,
    )

    audited_training_config = replace(
        training_config,
        experiment_name=f"{training_config.experiment_name}_{config.auditor_variant}_canary",
        run=replace(
            training_config.run,
            training_seed=seed_plan.retrain_seed,
            save_checkpoint=False,
            notes=(
                "Evaluator-controlled canary stress-test retraining. "
                "This model is used only for controlled leakage evaluation."
            ),
        ),
    )
    set_global_seed(seed_plan.retrain_seed)
    training_outcome = train_dp_sgd(
        audited_training_config,
        active_logger,
        dataset_bundle=augmented_bundle,
        save_checkpoint=False,
        run_descriptor=(
            f"{audited_training_config.experiment_name}"
            f"_exp{seed_plan.experiment_seed}_audit{seed_plan.audit_seed}"
        ),
        return_model_state=True,
    )
    if training_outcome.model_state_dict is None:
        raise RuntimeError("Canary stress-test training did not return model weights.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_state_dict(
        training_config.model,
        training_outcome.model_state_dict,
        device=device,
    )
    member_scores = _score_canaries(model, insertion_result.inserted_canaries, device=device)
    nonmember_scores = _score_reference_canaries(model, insertion_result.reference_canaries, device=device)
    mean_member_score = statistics.fmean(member_scores)
    mean_nonmember_score = statistics.fmean(nonmember_scores)
    return AuditObservation(
        audit_regime=config.audit_regime,
        auditor_variant=config.auditor_variant,
        raw_statistics={
            "experiment_seed": float(seed_plan.experiment_seed),
            "audit_seed": float(seed_plan.audit_seed),
            "canary_generation_seed": float(seed_plan.canary_generation_seed),
            "canary_insertion_seed": float(seed_plan.canary_insertion_seed),
            "retrain_seed": float(seed_plan.retrain_seed),
            "inserted_example_count": float(insertion_result.inserted_example_count),
            "unique_canary_count": float(insertion_result.unique_inserted_count),
            "member_score_mean": float(mean_member_score),
            "nonmember_score_mean": float(mean_nonmember_score),
            "score_mean_gap": float(mean_member_score - mean_nonmember_score),
        },
        artifact_payload={
            "audit_mode": "one_run",
            "canary_strategy": config.canary.strategy,
            "optimize_steps": config.canary.optimize_steps,
            "seed_plan": seed_plan.to_dict(),
            "canary_training_run_id": training_outcome.record.training_run_id,
            "canary_training_epsilon_upper_theory": training_outcome.record.epsilon_upper_theory,
            "canary_training_utility_metrics": training_outcome.record.utility_metrics,
            "inserted_canaries": [
                {
                    "identifier": payload.identifier,
                    "target_label": payload.target_label,
                    "descriptor": payload.descriptor,
                }
                for payload in insertion_result.inserted_canaries
            ],
        },
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        score_name="target_label_logit_margin",
        epsilon_upper_theory=training_outcome.record.epsilon_upper_theory,
        utility_metrics=training_outcome.record.utility_metrics,
        audited_model_run_id=training_outcome.record.training_run_id,
    )


def _score_canaries(model, canaries, *, device) -> list[float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for canary scoring.") from exc

    with torch.no_grad():
        images = torch.stack([payload.inserted_image for payload in canaries]).to(device)
        labels = torch.tensor([payload.target_label for payload in canaries], dtype=torch.long, device=device)
        logits = model(images)
        return _target_margin_scores(logits, labels)


def _score_reference_canaries(model, canaries, *, device) -> list[float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for canary scoring.") from exc

    with torch.no_grad():
        images = torch.stack([payload.reference_image for payload in canaries]).to(device)
        labels = torch.tensor([payload.target_label for payload in canaries], dtype=torch.long, device=device)
        logits = model(images)
        return _target_margin_scores(logits, labels)


def _target_margin_scores(logits, labels) -> list[float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for canary scoring.") from exc

    target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    other_logits = logits.masked_fill(~mask, float("-inf"))
    strongest_other = other_logits.max(dim=1).values
    return (target_logits - strongest_other).detach().cpu().tolist()


# Dataset name -> (C, H, W) for canary image generation.
_IMAGE_SHAPES: dict[str, tuple[int, int, int]] = {
    "mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
}


def _infer_image_shape(dataset_name: str) -> tuple[int, int, int]:
    name = dataset_name.lower()
    if name in _IMAGE_SHAPES:
        return _IMAGE_SHAPES[name]
    # Tabular datasets don't have a natural image shape; use a reasonable
    # 1-channel square for canary construction.
    return (1, 28, 28)
