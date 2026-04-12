from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODEX_DIR = ROOT / "codex"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

import run_raw_lira_pilot as raw


RESULTS_DIR = CODEX_DIR / "results" / "raw_lira_pathology"
SUMMARY_JSON = RESULTS_DIR / "raw_lira_pathology_summary.json"
SUMMARY_CSV = RESULTS_DIR / "raw_lira_pathology_summary.csv"


def quantiles(values: list[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def auc_pair(member_scores: list[float], nonmember_scores: list[float]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
    scores = member_scores + nonmember_scores
    auc_forward = float(roc_auc_score(labels, scores))
    auc_negated = float(roc_auc_score(labels, [-score for score in scores]))
    return {
        "auc_forward": auc_forward,
        "auc_negated": auc_negated,
        "preferred_direction": "higher_is_member" if auc_forward >= auc_negated else "lower_is_member",
    }


def build_diagnostic_rows(
    *,
    dataset_name: str,
    training_case: dict,
    member_scores: list[float],
    nonmember_scores: list[float],
    attack_name: str,
    k: int,
) -> list[dict]:
    estimate_higher = raw.estimate_conservative(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=training_case["config"].privacy.delta,
    )
    estimate_lower = raw.estimate_empirical_lower_bound(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=training_case["config"].privacy.delta,
        score_direction="lower",
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
    )
    auc = auc_pair(member_scores, nonmember_scores)
    member_q = quantiles(member_scores)
    nonmember_q = quantiles(nonmember_scores)
    record = training_case["record"]

    return [
        {
            "dataset": dataset_name,
            "attack": attack_name,
            "k_shadows": k,
            "direction_tested": "higher_is_member",
            "epsilon_upper_tighter": record.epsilon_upper_pld,
            "epsilon_lower_conservative": estimate_higher.epsilon_lower_empirical,
            "epsilon_lower_point": estimate_higher.epsilon_lower_empirical_point_estimate,
            "selected_tpr": estimate_higher.selected_tpr,
            "selected_fpr": estimate_higher.selected_fpr,
            "warning": estimate_higher.warning_message,
            "auc_forward": auc["auc_forward"],
            "auc_negated": auc["auc_negated"],
            "preferred_direction": auc["preferred_direction"],
            "member_mean": sum(member_scores) / len(member_scores),
            "nonmember_mean": sum(nonmember_scores) / len(nonmember_scores),
            "member_median": member_q["median"],
            "nonmember_median": nonmember_q["median"],
            "member_q95": member_q["q95"],
            "nonmember_q95": nonmember_q["q95"],
            "member_q99": member_q["q99"],
            "nonmember_q99": nonmember_q["q99"],
        },
        {
            "dataset": dataset_name,
            "attack": attack_name,
            "k_shadows": k,
            "direction_tested": "lower_is_member",
            "epsilon_upper_tighter": record.epsilon_upper_pld,
            "epsilon_lower_conservative": estimate_lower.epsilon_lower_empirical,
            "epsilon_lower_point": estimate_lower.epsilon_lower_empirical_point_estimate,
            "selected_tpr": estimate_lower.selected_tpr,
            "selected_fpr": estimate_lower.selected_fpr,
            "warning": estimate_lower.warning_message,
            "auc_forward": auc["auc_forward"],
            "auc_negated": auc["auc_negated"],
            "preferred_direction": auc["preferred_direction"],
            "member_mean": sum(member_scores) / len(member_scores),
            "nonmember_mean": sum(nonmember_scores) / len(nonmember_scores),
            "member_median": member_q["median"],
            "nonmember_median": nonmember_q["median"],
            "member_q95": member_q["q95"],
            "nonmember_q95": nonmember_q["q95"],
            "member_q99": member_q["q99"],
            "nonmember_q99": nonmember_q["q99"],
        },
    ]


def write_summary(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    import torch
    from dp_audit_tightness.models.io import load_model_for_inference

    logger = raw.configure_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.DATA_DIR.mkdir(parents=True, exist_ok=True)

    adult_npz = raw.DATA_DIR / "adult.npz"
    adult_input_dim, adult_num_classes, _adult_rows = raw.ensure_adult_npz(adult_npz)

    dataset_specs = [
        {
            "name": "cifar10",
            "config": raw.make_train_config(
                dataset_name="cifar10",
                data_dir=str(raw.DATA_DIR),
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
            "config": raw.make_train_config(
                dataset_name="adult",
                data_dir=str(raw.DATA_DIR),
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

    rows: list[dict] = []
    k_values = [16]

    for spec in dataset_specs:
        dataset_name = spec["name"]
        logger.info("Preparing pathology diagnostics for %s", dataset_name)
        bundle = raw.load_dataset_bundle(spec["config"].dataset, split_seed=spec["config"].run.split_seed)
        bundle = raw.subset_bundle(bundle, spec["train_limit"], spec["eval_limit"], seed=777)
        training_case = raw.train_target_case(dataset_name, spec["config"], bundle, logger)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_model = load_model_for_inference(spec["config"].model, training_case["record"].model_artifact_path, device=device)
        query_train_indices, query_eval_indices, per_seed_train, per_seed_eval = raw.sample_query_indices(
            train_size=len(bundle.train_dataset),
            eval_size=len(bundle.eval_dataset),
        )
        target_train_losses = raw.compute_loss_for_indices(target_model, bundle.train_dataset, query_train_indices, device=device)
        target_eval_losses = raw.compute_loss_for_indices(target_model, bundle.eval_dataset, query_eval_indices, device=device)

        started = time.time()
        shadow_train_losses, shadow_eval_losses, shadow_member_sets = raw.train_shadow_losses(
            training_case=training_case,
            dataset_name=dataset_name,
            query_train_indices=query_train_indices,
            query_eval_indices=query_eval_indices,
            k_max=max(k_values),
            logger=logger,
        )
        logger.info(
            "Prepared pathology shadows for %s in %.1fs",
            dataset_name,
            time.time() - started,
        )

        for k in k_values:
            raw_member_scores, raw_nonmember_scores = raw.raw_lira_scores(
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
            rows.extend(
                build_diagnostic_rows(
                    dataset_name=dataset_name,
                    training_case=training_case,
                    member_scores=raw_member_scores,
                    nonmember_scores=raw_nonmember_scores,
                    attack_name="passive_raw_lira",
                    k=k,
                )
            )

    write_summary(rows)
    print(f"Wrote summary JSON to {SUMMARY_JSON}")
    print(f"Wrote summary CSV to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
