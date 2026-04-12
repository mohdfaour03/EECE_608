from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.config import (
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    PrivacyConfig,
    RunConfig,
    TrainExperimentConfig,
    TrainingConfig,
    config_to_dict,
)
from dp_audit_tightness.data.datasets import DatasetBundle, load_dataset_bundle
from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound
from dp_audit_tightness.training.dp_sgd import train_dp_sgd
from dp_audit_tightness.utils.logging_utils import configure_logging
from dp_audit_tightness.utils.results import save_json, save_training_result
from dp_audit_tightness.utils.seeds import set_global_seed


CODEX_DIR = ROOT / "codex"
DATA_DIR = CODEX_DIR / "data"
RESULTS_DIR = CODEX_DIR / "results" / "raw_lira_pilot"
SUMMARY_JSON = RESULTS_DIR / "raw_lira_pilot_summary.json"
SUMMARY_CSV = RESULTS_DIR / "raw_lira_pilot_summary.csv"

QUERY_BUDGET = 512
AUDIT_SEEDS = [401, 402, 403, 404, 405]
K_VALUES = [8, 16]
SHADOW_SUBSET_FRACTION = 0.5


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
    training_seed: int,
) -> TrainExperimentConfig:
    return TrainExperimentConfig(
        experiment_name=f"codex_raw_lira_{dataset_name}",
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
            notes="Codex raw LiRA sidecar pilot.",
        ),
    )


def train_target_case(dataset_name: str, config: TrainExperimentConfig, bundle: DatasetBundle, logger) -> dict:
    set_global_seed(config.run.training_seed)
    started = time.time()
    outcome = train_dp_sgd(config=config, logger=logger, dataset_bundle=bundle)

    config_path = RESULTS_DIR / "training" / "configs" / f"{outcome.record.training_run_id}_config.json"
    outcome.record.config_snapshot_path = str(config_path)
    save_json(config_path, config_to_dict(config))
    training_result_path = save_training_result(outcome.record, results_root=config.run.results_root)

    return {
        "dataset": dataset_name,
        "config": config,
        "bundle": bundle,
        "record": outcome.record,
        "training_result_path": str(training_result_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def sample_query_indices(train_size: int, eval_size: int) -> tuple[list[int], list[int], dict[int, list[int]], dict[int, list[int]]]:
    all_train_indices: set[int] = set()
    all_eval_indices: set[int] = set()
    per_seed_train: dict[int, list[int]] = {}
    per_seed_eval: dict[int, list[int]] = {}
    for seed in AUDIT_SEEDS:
        rng = random.Random(seed)
        train_indices = rng.sample(range(train_size), QUERY_BUDGET // 2)
        eval_indices = rng.sample(range(eval_size), QUERY_BUDGET // 2)
        per_seed_train[seed] = train_indices
        per_seed_eval[seed] = eval_indices
        all_train_indices.update(train_indices)
        all_eval_indices.update(eval_indices)
    return sorted(all_train_indices), sorted(all_eval_indices), per_seed_train, per_seed_eval


def compute_loss_for_indices(model, dataset, indices, device, batch_size=256):
    import torch
    import torch.nn.functional as F

    if not indices:
        return []

    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images = []
            labels = []
            for index in batch_indices:
                image, label = dataset[index]
                images.append(image)
                labels.append(label)
            x = torch.stack(images).to(device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            batch_losses = F.cross_entropy(model(x), y, reduction="none")
            losses.extend(float(value) for value in batch_losses.detach().cpu().tolist())
    return losses


def estimate_conservative(member_scores: list[float], nonmember_scores: list[float], delta: float):
    return estimate_empirical_lower_bound(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=delta,
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
    )


def estimate_conservative_with_direction(
    member_scores: list[float],
    nonmember_scores: list[float],
    delta: float,
    score_direction: str,
):
    return estimate_empirical_lower_bound(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=delta,
        score_direction=score_direction,
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
    )


def train_shadow_losses(
    *,
    training_case: dict,
    dataset_name: str,
    query_train_indices: list[int],
    query_eval_indices: list[int],
    k_max: int,
    logger,
) -> tuple[list[list[float]], list[list[float]], list[set[int]]]:
    import torch
    from torch.utils.data import Subset
    from dp_audit_tightness.models.io import load_model_from_state_dict

    config: TrainExperimentConfig = training_case["config"]
    bundle: DatasetBundle = training_case["bundle"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shadow_train_losses: list[list[float]] = []
    shadow_eval_losses: list[list[float]] = []
    shadow_member_sets: list[set[int]] = []

    train_size = len(bundle.train_dataset)
    shadow_train_size = max(2, int(train_size * SHADOW_SUBSET_FRACTION))

    for shadow_index in range(k_max):
        shadow_seed = 5000 + shadow_index
        rng = random.Random(shadow_seed)
        all_indices = list(range(train_size))
        rng.shuffle(all_indices)
        in_indices = sorted(all_indices[:shadow_train_size])
        shadow_member_sets.append(set(in_indices))

        shadow_bundle = DatasetBundle(
            train_dataset=Subset(bundle.train_dataset, in_indices),
            eval_dataset=bundle.eval_dataset,
            input_dim=bundle.input_dim,
            num_classes=bundle.num_classes,
            train_size=len(in_indices),
            eval_size=bundle.eval_size,
        )
        shadow_config = make_train_config(
            dataset_name=config.dataset.name,
            data_dir=config.dataset.data_dir,
            model_name=config.model.name,
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            num_classes=config.model.num_classes,
            learning_rate=config.training.optimizer.learning_rate,
            momentum=config.training.optimizer.momentum,
            training_seed=shadow_seed,
        )
        shadow_config.experiment_name = f"{config.experiment_name}_shadow_{shadow_index}"
        shadow_config.run.results_root = str(RESULTS_DIR)
        shadow_config.run.save_checkpoint = False
        shadow_config.run.notes = "Codex raw LiRA shadow model."

        set_global_seed(shadow_seed)
        outcome = train_dp_sgd(
            config=shadow_config,
            logger=logger,
            dataset_bundle=shadow_bundle,
            save_checkpoint=False,
            run_descriptor=f"{dataset_name}_shadow_{shadow_index}",
            return_model_state=True,
        )
        shadow_model = load_model_from_state_dict(config.model, outcome.model_state_dict, device=device)
        shadow_train_losses.append(
            compute_loss_for_indices(shadow_model, bundle.train_dataset, query_train_indices, device=device)
        )
        shadow_eval_losses.append(
            compute_loss_for_indices(shadow_model, bundle.eval_dataset, query_eval_indices, device=device)
        )
        logger.info(
            "Shadow %s/%s complete for %s",
            shadow_index + 1,
            k_max,
            dataset_name,
        )

    return shadow_train_losses, shadow_eval_losses, shadow_member_sets


def raw_lira_scores(
    *,
    query_train_indices: list[int],
    query_eval_indices: list[int],
    per_seed_train: dict[int, list[int]],
    per_seed_eval: dict[int, list[int]],
    target_train_losses: list[float],
    target_eval_losses: list[float],
    shadow_train_losses: list[list[float]],
    shadow_eval_losses: list[list[float]],
    shadow_member_sets: list[set[int]],
    k: int,
) -> tuple[list[float], list[float]]:
    train_pos = {index: pos for pos, index in enumerate(query_train_indices)}
    eval_pos = {index: pos for pos, index in enumerate(query_eval_indices)}

    member_scores: list[float] = []
    nonmember_scores: list[float] = []

    for seed in AUDIT_SEEDS:
        for index in per_seed_train[seed]:
            pos = train_pos[index]
            in_losses = []
            out_losses = []
            for shadow_index in range(k):
                loss_value = float(shadow_train_losses[shadow_index][pos])
                if index in shadow_member_sets[shadow_index]:
                    in_losses.append(loss_value)
                else:
                    out_losses.append(loss_value)
            if in_losses and out_losses:
                score = statistics.fmean(out_losses) - statistics.fmean(in_losses)
            elif out_losses:
                score = statistics.fmean(out_losses)
            else:
                score = -statistics.fmean(in_losses) if in_losses else 0.0
            member_scores.append(float(score))

        for index in per_seed_eval[seed]:
            pos = eval_pos[index]
            shadow_losses = [float(shadow_eval_losses[shadow_index][pos]) for shadow_index in range(k)]
            shadow_mean = statistics.fmean(shadow_losses) if shadow_losses else 0.0
            score = shadow_mean - float(target_eval_losses[pos])
            nonmember_scores.append(float(score))

    return member_scores, nonmember_scores


def negative_loss_scores(
    *,
    query_train_indices: list[int],
    query_eval_indices: list[int],
    per_seed_train: dict[int, list[int]],
    per_seed_eval: dict[int, list[int]],
    target_train_losses: list[float],
    target_eval_losses: list[float],
) -> tuple[list[float], list[float]]:
    train_pos = {index: pos for pos, index in enumerate(query_train_indices)}
    eval_pos = {index: pos for pos, index in enumerate(query_eval_indices)}
    member_scores: list[float] = []
    nonmember_scores: list[float] = []

    for seed in AUDIT_SEEDS:
        member_scores.extend(-float(target_train_losses[train_pos[index]]) for index in per_seed_train[seed])
        nonmember_scores.extend(-float(target_eval_losses[eval_pos[index]]) for index in per_seed_eval[seed])

    return member_scores, nonmember_scores


def build_result_row(
    *,
    dataset_name: str,
    attack_name: str,
    k: int,
    training_case: dict,
    member_scores: list[float],
    nonmember_scores: list[float],
    score_direction: str = "higher",
) -> dict:
    estimate = estimate_conservative_with_direction(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=training_case["config"].privacy.delta,
        score_direction=score_direction,
    )
    record = training_case["record"]
    epsilon_upper_tighter = record.epsilon_upper_pld
    epsilon_lower = estimate.epsilon_lower_empirical
    return {
        "dataset": dataset_name,
        "attack": attack_name,
        "status": "ok",
        "k_shadows": k,
        "score_direction": score_direction,
        "query_budget_per_seed": QUERY_BUDGET,
        "num_audit_seeds": len(AUDIT_SEEDS),
        "audit_seeds": json.dumps(AUDIT_SEEDS),
        "epsilon_upper_rdp": record.epsilon_upper_theory,
        "epsilon_upper_tighter": epsilon_upper_tighter,
        "tighter_upper_backend": record.pld_accounting_backend,
        "epsilon_lower_conservative": epsilon_lower,
        "epsilon_lower_point": estimate.epsilon_lower_empirical_point_estimate,
        "tightness_ratio_tighter": (epsilon_lower / epsilon_upper_tighter) if epsilon_upper_tighter else None,
        "privacy_loss_gap_tighter": (epsilon_upper_tighter - epsilon_lower) if epsilon_upper_tighter is not None else None,
        "selected_tpr": estimate.selected_tpr,
        "selected_fpr": estimate.selected_fpr,
        "warning": estimate.warning_message,
        "num_member_samples": estimate.num_member_samples,
        "num_nonmember_samples": estimate.num_nonmember_samples,
        "member_score_mean": statistics.fmean(member_scores),
        "nonmember_score_mean": statistics.fmean(nonmember_scores),
        "score_gap": statistics.fmean(member_scores) - statistics.fmean(nonmember_scores),
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
    import torch
    from dp_audit_tightness.models.io import load_model_for_inference

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
                training_seed=123,
            ),
            "train_limit": 4096,
            "eval_limit": 1024,
        },
    ]

    summary: list[dict] = []
    k_max = max(K_VALUES)

    for spec in dataset_specs:
        dataset_name = spec["name"]
        config = spec["config"]
        try:
            logger.info("Loading dataset bundle for %s", dataset_name)
            bundle = load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)
            bundle = subset_bundle(bundle, spec["train_limit"], spec["eval_limit"], seed=777)
            training_case = train_target_case(dataset_name, config, bundle, logger)

            summary.append(
                {
                    "dataset": dataset_name,
                    "attack": "__training__",
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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            target_model = load_model_for_inference(config.model, training_case["record"].model_artifact_path, device=device)

            query_train_indices, query_eval_indices, per_seed_train, per_seed_eval = sample_query_indices(
                train_size=len(bundle.train_dataset),
                eval_size=len(bundle.eval_dataset),
            )
            target_train_losses = compute_loss_for_indices(
                target_model,
                bundle.train_dataset,
                query_train_indices,
                device=device,
            )
            target_eval_losses = compute_loss_for_indices(
                target_model,
                bundle.eval_dataset,
                query_eval_indices,
                device=device,
            )

            started = time.time()
            shadow_train_losses, shadow_eval_losses, shadow_member_sets = train_shadow_losses(
                training_case=training_case,
                dataset_name=dataset_name,
                query_train_indices=query_train_indices,
                query_eval_indices=query_eval_indices,
                k_max=k_max,
                logger=logger,
            )
            logger.info(
                "Prepared raw LiRA shadow losses for %s in %.1fs",
                dataset_name,
                time.time() - started,
            )

            for k in K_VALUES:
                try:
                    nl_member_scores, nl_nonmember_scores = negative_loss_scores(
                        query_train_indices=query_train_indices,
                        query_eval_indices=query_eval_indices,
                        per_seed_train=per_seed_train,
                        per_seed_eval=per_seed_eval,
                        target_train_losses=target_train_losses,
                        target_eval_losses=target_eval_losses,
                    )
                    row = build_result_row(
                        dataset_name=dataset_name,
                        attack_name="passive_negative_loss_matched",
                        k=k,
                        training_case=training_case,
                        member_scores=nl_member_scores,
                        nonmember_scores=nl_nonmember_scores,
                        score_direction="higher",
                    )
                    summary.append(row)
                    logger.info(
                        "Completed matched negative-loss baseline on %s K=%s eps_lower=%s",
                        dataset_name,
                        k,
                        row["epsilon_lower_conservative"],
                    )
                except Exception as exc:
                    summary.append(
                        {
                            "dataset": dataset_name,
                            "attack": "passive_negative_loss_matched",
                            "k_shadows": k,
                            "status": "error",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    logger.exception("Matched negative-loss failed on %s K=%s", dataset_name, k)

                try:
                    raw_member_scores, raw_nonmember_scores = raw_lira_scores(
                        query_train_indices=query_train_indices,
                        query_eval_indices=query_eval_indices,
                        per_seed_train=per_seed_train,
                        per_seed_eval=per_seed_eval,
                        target_train_losses=target_train_losses,
                        target_eval_losses=target_eval_losses,
                        shadow_train_losses=shadow_train_losses,
                        shadow_eval_losses=shadow_eval_losses,
                        shadow_member_sets=shadow_member_sets,
                        k=k,
                    )
                    row = build_result_row(
                        dataset_name=dataset_name,
                        attack_name="passive_raw_lira",
                        k=k,
                        training_case=training_case,
                        member_scores=raw_member_scores,
                        nonmember_scores=raw_nonmember_scores,
                        score_direction="lower",
                    )
                    summary.append(row)
                    logger.info(
                        "Completed raw LiRA on %s K=%s eps_lower=%s",
                        dataset_name,
                        k,
                        row["epsilon_lower_conservative"],
                    )
                except Exception as exc:
                    summary.append(
                        {
                            "dataset": dataset_name,
                            "attack": "passive_raw_lira",
                            "k_shadows": k,
                            "status": "error",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    logger.exception("Raw LiRA failed on %s K=%s", dataset_name, k)
        except Exception as exc:
            summary.append(
                {
                    "dataset": dataset_name,
                    "attack": "__dataset__",
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
