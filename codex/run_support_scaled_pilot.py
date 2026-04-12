from __future__ import annotations

import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.auditing.canary.generation import generate_canaries, insert_canaries_into_dataset
from dp_audit_tightness.auditing.canary.one_run import (
    _infer_image_shape,
    _score_canaries,
    _score_reference_canaries,
)
from dp_audit_tightness.auditing.canary.seeding import build_canary_seed_plan
from dp_audit_tightness.auditing.passive.auditors import run_passive_audit_once
from dp_audit_tightness.config import (
    AuditRunConfig,
    CanaryAuditConfig,
    CanaryConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    PassiveAuditConfig,
    PassiveAuditorConfig,
    PrivacyConfig,
    RunConfig,
    SaturationConfig,
    TrainExperimentConfig,
    TrainingConfig,
    config_to_dict,
)
from dp_audit_tightness.data.datasets import DatasetBundle, load_dataset_bundle
from dp_audit_tightness.models.io import load_model_for_inference, load_model_from_state_dict
from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound
from dp_audit_tightness.training.dp_sgd import train_dp_sgd
from dp_audit_tightness.utils.logging_utils import configure_logging
from dp_audit_tightness.utils.results import save_json, save_training_result
from dp_audit_tightness.utils.seeds import set_global_seed


CODEX_DIR = ROOT / "codex"
DATA_DIR = CODEX_DIR / "data"
RESULTS_DIR = CODEX_DIR / "results" / "support_scaled_pilot"
SUMMARY_JSON = RESULTS_DIR / "support_scaled_pilot_summary.json"
SUMMARY_CSV = RESULTS_DIR / "support_scaled_pilot_summary.csv"

PASSIVE_SUPPORT_LEVELS = [
    {"support_label": "smoke", "query_budget": 64, "audit_seeds": [401]},
    {"support_label": "medium", "query_budget": 256, "audit_seeds": [401, 402, 403]},
    {"support_label": "large", "query_budget": 512, "audit_seeds": [401, 402, 403, 404, 405]},
]

CANARY_SUPPORT_LEVELS = [
    {"support_label": "smoke", "num_canaries": 8, "audit_seeds": [101]},
    {"support_label": "medium", "num_canaries": 16, "audit_seeds": [101, 102, 103]},
    {"support_label": "large", "num_canaries": 32, "audit_seeds": [101, 102, 103, 104, 105]},
]


def ensure_adult_npz(target_path: Path) -> tuple[int, int, int]:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_openml

    if target_path.exists():
        cached = np.load(target_path)
        features = cached["features"]
        labels = cached["labels"]
        return int(features.shape[1]), int(labels.max()) + 1, int(features.shape[0])

    target_path.parent.mkdir(parents=True, exist_ok=True)
    adult = fetch_openml("adult", version=2, as_frame=True)
    features = adult.data.copy()
    labels = adult.target.astype(str).copy()

    features = features.replace("?", pd.NA)
    mask = ~features.isna().any(axis=1) & labels.notna()
    features = features.loc[mask]
    labels = labels.loc[mask]

    encoded = pd.get_dummies(features, dummy_na=False, drop_first=False, dtype="float32")
    label_values = sorted(labels.unique().tolist())
    label_map = {label: index for index, label in enumerate(label_values)}
    y = labels.map(label_map).astype("int64").to_numpy()
    x = encoded.to_numpy(dtype="float32")

    np.savez(target_path, features=x, labels=y)
    metadata = {
        "rows": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "num_classes": int(len(label_values)),
        "label_map": label_map,
    }
    save_json(target_path.with_suffix(".metadata.json"), metadata)
    return int(x.shape[1]), int(len(label_values)), int(x.shape[0])


def subset_bundle(bundle: DatasetBundle, train_limit: int, eval_limit: int, seed: int) -> DatasetBundle:
    import torch
    from torch.utils.data import Subset

    generator = torch.Generator().manual_seed(seed)
    train_perm = torch.randperm(len(bundle.train_dataset), generator=generator).tolist()
    eval_perm = torch.randperm(len(bundle.eval_dataset), generator=generator).tolist()
    train_indices = train_perm[: min(train_limit, len(train_perm))]
    eval_indices = eval_perm[: min(eval_limit, len(eval_perm))]
    return DatasetBundle(
        train_dataset=Subset(bundle.train_dataset, train_indices),
        eval_dataset=Subset(bundle.eval_dataset, eval_indices),
        input_dim=bundle.input_dim,
        num_classes=bundle.num_classes,
        train_size=len(train_indices),
        eval_size=len(eval_indices),
    )


def make_train_config(
    *,
    dataset_name: str,
    data_dir: str,
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    learning_rate: float,
    momentum: float,
    split_seed: int,
    training_seed: int,
) -> TrainExperimentConfig:
    return TrainExperimentConfig(
        experiment_name=f"codex_support_scaled_{dataset_name}",
        dataset=DatasetConfig(
            name=dataset_name,
            data_dir=data_dir,
            validation_fraction=0.1,
            num_workers=0,
        ),
        model=ModelConfig(
            name=model_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        ),
        training=TrainingConfig(
            batch_size=128,
            eval_batch_size=256,
            epochs=1,
            clipping_norm=1.0,
            noise_multiplier=1.1,
            optimizer=OptimizerConfig(
                name="sgd",
                learning_rate=learning_rate,
                weight_decay=0.0,
                momentum=momentum,
            ),
        ),
        privacy=PrivacyConfig(delta=1e-5, accountant="rdp"),
        run=RunConfig(
            split_seed=11,
            training_seed=training_seed,
            results_root=str(RESULTS_DIR),
            save_checkpoint=True,
            notes="Codex support-scaled pilot sidecar run.",
        ),
    )


def train_dataset_case(name: str, config: TrainExperimentConfig, bundle: DatasetBundle, logger) -> dict:
    set_global_seed(config.run.training_seed)
    started = time.time()
    outcome = train_dp_sgd(config=config, logger=logger, dataset_bundle=bundle)

    config_path = RESULTS_DIR / "training" / "configs" / f"{outcome.record.training_run_id}_config.json"
    outcome.record.config_snapshot_path = str(config_path)
    save_json(config_path, config_to_dict(config))
    training_result_path = save_training_result(outcome.record, results_root=config.run.results_root)

    return {
        "dataset": name,
        "config": config,
        "bundle": bundle,
        "record": outcome.record,
        "checkpoint_path": outcome.checkpoint_path,
        "training_result_path": str(training_result_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def safe_ratio(lower: float | None, upper: float | None) -> float | None:
    if lower is None or upper is None or upper <= 0.0:
        return None
    return lower / upper


def safe_gap(upper: float | None, lower: float | None) -> float | None:
    if upper is None or lower is None:
        return None
    return upper - lower


def aggregate_estimate(member_scores: list[float], nonmember_scores: list[float], delta: float):
    return estimate_empirical_lower_bound(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=delta,
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
    )


def run_passive_support_level(
    *,
    dataset_name: str,
    attack_name: str,
    score_type: str,
    training_case: dict,
    support_label: str,
    query_budget: int,
    audit_seeds: list[int],
) -> dict:
    import torch

    config: TrainExperimentConfig = training_case["config"]
    bundle: DatasetBundle = training_case["bundle"]
    record = training_case["record"]

    passive_config = PassiveAuditConfig(
        training_result_path=training_case["training_result_path"],
        audit_regime="passive_final_model_only",
        auditor_variant=attack_name,
        auditor_strength_rank=1,
        delta=config.privacy.delta,
        repeated_seeds=list(audit_seeds),
        passive=PassiveAuditorConfig(
            query_budget=query_budget,
            score_type=score_type,
            calibration_method="none",
            calibration_fraction=0.0,
        ),
        saturation=SaturationConfig(),
        run=AuditRunConfig(
            results_root=str(RESULTS_DIR),
            save_debug_artifacts=False,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(config.model, record.model_artifact_path, device=device)

    observations = [
        run_passive_audit_once(
            training_record=record,
            training_config=config,
            config=passive_config,
            seed=audit_seed,
            dataset_bundle=bundle,
            model=model,
        )
        for audit_seed in audit_seeds
    ]
    member_scores = [score for observation in observations for score in observation.member_scores]
    nonmember_scores = [score for observation in observations for score in observation.nonmember_scores]
    estimate = aggregate_estimate(member_scores, nonmember_scores, config.privacy.delta)

    epsilon_upper_rdp = record.epsilon_upper_theory
    epsilon_upper_tighter = record.epsilon_upper_pld
    epsilon_lower = estimate.epsilon_lower_empirical
    point_lower = estimate.epsilon_lower_empirical_point_estimate

    return {
        "dataset": dataset_name,
        "attack": attack_name,
        "support_label": support_label,
        "status": "ok",
        "audit_regime": passive_config.audit_regime,
        "score_name": score_type,
        "query_budget_per_seed": query_budget,
        "num_audit_seeds": len(audit_seeds),
        "audit_seeds": json.dumps(audit_seeds),
        "epsilon_upper_rdp": epsilon_upper_rdp,
        "epsilon_upper_tighter": epsilon_upper_tighter,
        "tighter_upper_backend": record.pld_accounting_backend,
        "epsilon_lower_conservative": epsilon_lower,
        "epsilon_lower_point": point_lower,
        "tightness_ratio_rdp": safe_ratio(epsilon_lower, epsilon_upper_rdp),
        "tightness_ratio_tighter": safe_ratio(epsilon_lower, epsilon_upper_tighter),
        "privacy_loss_gap_rdp": safe_gap(epsilon_upper_rdp, epsilon_lower),
        "privacy_loss_gap_tighter": safe_gap(epsilon_upper_tighter, epsilon_lower),
        "member_favoring": estimate.member_favoring,
        "selected_tpr": estimate.selected_tpr,
        "selected_fpr": estimate.selected_fpr,
        "warning": estimate.warning_message,
        "num_member_samples": estimate.num_member_samples,
        "num_nonmember_samples": estimate.num_nonmember_samples,
    }


def run_canary_support_level(
    *,
    dataset_name: str,
    attack_name: str,
    training_case: dict,
    support_label: str,
    num_canaries: int,
    audit_seeds: list[int],
    strategy: str = "random_canaries",
) -> dict:
    import torch

    config: TrainExperimentConfig = training_case["config"]
    base_bundle: DatasetBundle = training_case["bundle"]
    record = training_case["record"]

    if dataset_name not in {"mnist", "cifar10"}:
        return {
            "dataset": dataset_name,
            "attack": attack_name,
            "support_label": support_label,
            "status": "not_supported",
            "reason": "canary sidecar pilot currently supports image datasets only",
        }

    canary_config = CanaryAuditConfig(
        training_result_path=training_case["training_result_path"],
        audit_regime="evaluator_controlled_canary_stress_test",
        auditor_variant=attack_name,
        auditor_strength_rank=1,
        audit_mode="repeated_run",
        delta=config.privacy.delta,
        repeated_seeds=list(audit_seeds),
        canary=CanaryConfig(
            strategy=strategy,
            num_canaries=num_canaries,
            insertion_rate=0.005,
            optimize_steps=0,
        ),
        saturation=SaturationConfig(),
        run=AuditRunConfig(
            results_root=str(RESULTS_DIR),
            save_debug_artifacts=False,
        ),
    )

    epsilon_upper_rdp_values: list[float] = []
    epsilon_upper_tighter_values: list[float] = []
    member_scores: list[float] = []
    nonmember_scores: list[float] = []
    inserted_example_counts: list[int] = []
    image_shape = _infer_image_shape(config.dataset.name)

    for audit_seed in audit_seeds:
        seed_plan = build_canary_seed_plan(
            experiment_seed=record.training_seed,
            audit_seed=audit_seed,
            dataset_split_seed=record.split_seed,
        )
        canaries = generate_canaries(
            canary_config.canary,
            seed=seed_plan.canary_generation_seed,
            num_classes=config.model.num_classes,
            image_shape=image_shape,
        )
        insertion_result = insert_canaries_into_dataset(
            base_bundle.train_dataset,
            canaries,
            insertion_rate=canary_config.canary.insertion_rate,
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
        canary_train_config = TrainExperimentConfig(
            experiment_name=f"{config.experiment_name}_{attack_name}_{support_label}",
            dataset=config.dataset,
            model=config.model,
            training=config.training,
            privacy=config.privacy,
            run=RunConfig(
                split_seed=config.run.split_seed,
                training_seed=seed_plan.retrain_seed,
                results_root=str(RESULTS_DIR),
                save_checkpoint=False,
                notes="Codex support-scaled canary retraining run.",
            ),
        )
        set_global_seed(seed_plan.retrain_seed)
        outcome = train_dp_sgd(
            config=canary_train_config,
            logger=configure_logging(),
            dataset_bundle=augmented_bundle,
            save_checkpoint=False,
            run_descriptor=f"{attack_name}_{dataset_name}_{support_label}",
            return_model_state=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_from_state_dict(config.model, outcome.model_state_dict, device=device)
        member_scores.extend(_score_canaries(model, insertion_result.inserted_canaries, device=device))
        nonmember_scores.extend(_score_reference_canaries(model, insertion_result.reference_canaries, device=device))
        epsilon_upper_rdp_values.append(outcome.record.epsilon_upper_theory)
        epsilon_upper_tighter_values.append(outcome.record.epsilon_upper_pld)
        inserted_example_counts.append(insertion_result.inserted_example_count)

    estimate = aggregate_estimate(member_scores, nonmember_scores, config.privacy.delta)
    epsilon_upper_rdp = statistics.fmean(epsilon_upper_rdp_values)
    epsilon_upper_tighter = statistics.fmean(epsilon_upper_tighter_values)
    epsilon_lower = estimate.epsilon_lower_empirical
    point_lower = estimate.epsilon_lower_empirical_point_estimate

    return {
        "dataset": dataset_name,
        "attack": attack_name,
        "support_label": support_label,
        "status": "ok",
        "audit_regime": canary_config.audit_regime,
        "score_name": "target_label_logit_margin",
        "num_canaries_per_seed": num_canaries,
        "num_audit_seeds": len(audit_seeds),
        "audit_seeds": json.dumps(audit_seeds),
        "mean_inserted_example_count": round(statistics.fmean(inserted_example_counts), 3),
        "epsilon_upper_rdp": epsilon_upper_rdp,
        "epsilon_upper_tighter": epsilon_upper_tighter,
        "tighter_upper_backend": "google_dp_accounting",
        "epsilon_lower_conservative": epsilon_lower,
        "epsilon_lower_point": point_lower,
        "tightness_ratio_rdp": safe_ratio(epsilon_lower, epsilon_upper_rdp),
        "tightness_ratio_tighter": safe_ratio(epsilon_lower, epsilon_upper_tighter),
        "privacy_loss_gap_rdp": safe_gap(epsilon_upper_rdp, epsilon_lower),
        "privacy_loss_gap_tighter": safe_gap(epsilon_upper_tighter, epsilon_lower),
        "member_favoring": estimate.member_favoring,
        "selected_tpr": estimate.selected_tpr,
        "selected_fpr": estimate.selected_fpr,
        "warning": estimate.warning_message,
        "num_member_samples": estimate.num_member_samples,
        "num_nonmember_samples": estimate.num_nonmember_samples,
    }


def write_summary(summary: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    fieldnames: list[str] = []
    for row in summary:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def main() -> None:
    logger = configure_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    adult_npz = DATA_DIR / "adult.npz"
    adult_input_dim, adult_num_classes, adult_rows = ensure_adult_npz(adult_npz)
    logger.info(
        "Prepared adult dataset rows=%s input_dim=%s classes=%s",
        adult_rows,
        adult_input_dim,
        adult_num_classes,
    )

    dataset_specs = [
        {
            "name": "mnist",
            "config": make_train_config(
                dataset_name="mnist",
                data_dir=str(ROOT / "data" / "raw"),
                model_name="simple_mlp",
                input_dim=784,
                hidden_dim=64,
                num_classes=10,
                learning_rate=0.15,
                momentum=0.0,
                split_seed=11,
                training_seed=123,
            ),
            "train_limit": 2048,
            "eval_limit": 512,
        },
        {
            "name": "cifar10",
            "config": make_train_config(
                dataset_name="cifar10",
                data_dir=str(DATA_DIR),
                model_name="cnn_cifar10",
                input_dim=3072,
                hidden_dim=128,
                num_classes=10,
                learning_rate=0.05,
                momentum=0.9,
                split_seed=11,
                training_seed=123,
            ),
            "train_limit": 1024,
            "eval_limit": 256,
        },
        {
            "name": "adult",
            "config": make_train_config(
                dataset_name="adult",
                data_dir=str(DATA_DIR),
                model_name="tabular_mlp",
                input_dim=adult_input_dim,
                hidden_dim=64,
                num_classes=adult_num_classes,
                learning_rate=0.1,
                momentum=0.0,
                split_seed=11,
                training_seed=123,
            ),
            "train_limit": 4096,
            "eval_limit": 1024,
        },
    ]

    summary: list[dict] = []
    for spec in dataset_specs:
        dataset_name = spec["name"]
        config = spec["config"]
        try:
            logger.info("Loading dataset bundle for %s", dataset_name)
            bundle = load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)
            bundle = subset_bundle(bundle, spec["train_limit"], spec["eval_limit"], seed=777)
            logger.info(
                "Dataset %s subset sizes train=%s eval=%s",
                dataset_name,
                bundle.train_size,
                bundle.eval_size,
            )
            training_case = train_dataset_case(dataset_name, config, bundle, logger)
            summary.append(
                {
                    "dataset": dataset_name,
                    "attack": "__training__",
                    "support_label": "__training__",
                    "status": "ok",
                    "train_size": bundle.train_size,
                    "eval_size": bundle.eval_size,
                    "epsilon_upper_rdp": training_case["record"].epsilon_upper_theory,
                    "epsilon_upper_tighter": training_case["record"].epsilon_upper_pld,
                    "tighter_upper_backend": training_case["record"].pld_accounting_backend,
                    "accuracy": training_case["record"].utility_metrics.get("accuracy"),
                    "loss": training_case["record"].utility_metrics.get("loss"),
                    "training_elapsed_seconds": training_case["elapsed_seconds"],
                    "training_run_id": training_case["record"].training_run_id,
                    "training_result_path": training_case["training_result_path"],
                }
            )

            for attack_name, score_type in [
                ("passive_negative_loss", "negative_loss"),
                ("passive_score_fusion", "score_fusion"),
            ]:
                for support in PASSIVE_SUPPORT_LEVELS:
                    try:
                        started = time.time()
                        row = run_passive_support_level(
                            dataset_name=dataset_name,
                            attack_name=attack_name,
                            score_type=score_type,
                            training_case=training_case,
                            support_label=support["support_label"],
                            query_budget=support["query_budget"],
                            audit_seeds=support["audit_seeds"],
                        )
                        row["elapsed_seconds"] = round(time.time() - started, 3)
                        summary.append(row)
                        logger.info(
                            "Completed %s on %s support=%s eps_lower=%s",
                            attack_name,
                            dataset_name,
                            support["support_label"],
                            row.get("epsilon_lower_conservative"),
                        )
                    except Exception as exc:
                        summary.append(
                            {
                                "dataset": dataset_name,
                                "attack": attack_name,
                                "support_label": support["support_label"],
                                "status": "error",
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        logger.exception(
                            "Attack %s failed on %s support=%s",
                            attack_name,
                            dataset_name,
                            support["support_label"],
                        )

            for support in CANARY_SUPPORT_LEVELS:
                try:
                    started = time.time()
                    row = run_canary_support_level(
                        dataset_name=dataset_name,
                        attack_name="canary_random",
                        training_case=training_case,
                        support_label=support["support_label"],
                        num_canaries=support["num_canaries"],
                        audit_seeds=support["audit_seeds"],
                    )
                    row["elapsed_seconds"] = round(time.time() - started, 3)
                    summary.append(row)
                    logger.info(
                        "Completed canary_random on %s support=%s status=%s eps_lower=%s",
                        dataset_name,
                        support["support_label"],
                        row.get("status"),
                        row.get("epsilon_lower_conservative"),
                    )
                except Exception as exc:
                    summary.append(
                        {
                            "dataset": dataset_name,
                            "attack": "canary_random",
                            "support_label": support["support_label"],
                            "status": "error",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    logger.exception(
                        "Canary attack failed on %s support=%s",
                        dataset_name,
                        support["support_label"],
                    )
        except Exception as exc:
            summary.append(
                {
                    "dataset": dataset_name,
                    "attack": "__dataset__",
                    "support_label": "__dataset__",
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            logger.exception("Dataset case failed for %s", dataset_name)

    write_summary(summary)
    print(f"Wrote summary JSON to {SUMMARY_JSON}")
    print(f"Wrote summary CSV to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
