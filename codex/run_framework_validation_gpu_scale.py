from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODEX_DIR = ROOT / "codex"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

import torch

import run_framework_validation as framework_validation
import run_raw_lira_pilot as raw_lira
import run_support_scaled_pilot as support_scaled
from dp_audit_tightness.data.datasets import load_dataset_bundle
from dp_audit_tightness.models.io import load_model_for_inference
from dp_audit_tightness.utils.logging_utils import configure_logging


RESULTS_DIR = CODEX_DIR / "results" / "framework_validation_gpu_scale"
SUMMARY_JSON = RESULTS_DIR / "framework_validation_gpu_scale_summary.json"
SUMMARY_CSV = RESULTS_DIR / "framework_validation_gpu_scale_rows.csv"
CHECKS_JSON = RESULTS_DIR / "framework_validation_gpu_scale_checks.json"
REPORT_MD = RESULTS_DIR / "framework_validation_gpu_scale_report.md"

PASSIVE_AUDIT_SEEDS = list(range(401, 411))
CANARY_AUDIT_SEEDS = list(range(101, 111))
QUERY_BUDGET = 2048
RAW_LIRA_K = 32
NUM_CANARIES = 128


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for codex/run_framework_validation_gpu_scale.py. "
            "This machine does not currently expose a GPU."
        )


def patch_sidecar_modules() -> None:
    support_scaled.RESULTS_DIR = RESULTS_DIR
    support_scaled.SUMMARY_JSON = RESULTS_DIR / "unused_support_summary.json"
    support_scaled.SUMMARY_CSV = RESULTS_DIR / "unused_support_summary.csv"
    support_scaled.DATA_DIR = CODEX_DIR / "data"

    raw_lira.RESULTS_DIR = RESULTS_DIR
    raw_lira.SUMMARY_JSON = RESULTS_DIR / "unused_raw_lira_summary.json"
    raw_lira.SUMMARY_CSV = RESULTS_DIR / "unused_raw_lira_summary.csv"
    raw_lira.DATA_DIR = CODEX_DIR / "data"
    raw_lira.QUERY_BUDGET = QUERY_BUDGET
    raw_lira.AUDIT_SEEDS = list(PASSIVE_AUDIT_SEEDS)
    raw_lira.K_VALUES = [RAW_LIRA_K]
    raw_lira.SHADOW_SUBSET_FRACTION = 0.75


def make_scaled_config(
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
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
) -> Any:
    config = support_scaled.make_train_config(
        dataset_name=dataset_name,
        data_dir=data_dir,
        model_name=model_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        learning_rate=learning_rate,
        momentum=momentum,
        split_seed=split_seed,
        training_seed=training_seed,
    )
    config.training.batch_size = batch_size
    config.training.eval_batch_size = eval_batch_size
    config.training.epochs = epochs
    config.run.results_root = str(RESULTS_DIR)
    config.run.save_checkpoint = True
    config.run.notes = "Codex GPU-scale framework validation run."
    return config


def dataset_specs() -> list[dict[str, Any]]:
    adult_npz = CODEX_DIR / "data" / "adult.npz"
    adult_input_dim, adult_num_classes, _ = support_scaled.ensure_adult_npz(adult_npz)
    return [
        {
            "name": "mnist",
            "config": make_scaled_config(
                dataset_name="mnist",
                data_dir=str(ROOT / "data" / "raw"),
                model_name="simple_mlp",
                input_dim=784,
                hidden_dim=256,
                num_classes=10,
                learning_rate=0.15,
                momentum=0.0,
                split_seed=11,
                training_seed=123,
                batch_size=256,
                eval_batch_size=512,
                epochs=5,
            ),
            "train_limit": 60000,
            "eval_limit": 10000,
        },
        {
            "name": "cifar10",
            "config": make_scaled_config(
                dataset_name="cifar10",
                data_dir=str(CODEX_DIR / "data"),
                model_name="cnn_cifar10",
                input_dim=3072,
                hidden_dim=256,
                num_classes=10,
                learning_rate=0.05,
                momentum=0.9,
                split_seed=11,
                training_seed=123,
                batch_size=256,
                eval_batch_size=512,
                epochs=8,
            ),
            "train_limit": 50000,
            "eval_limit": 10000,
        },
        {
            "name": "adult",
            "config": make_scaled_config(
                dataset_name="adult",
                data_dir=str(CODEX_DIR / "data"),
                model_name="tabular_mlp",
                input_dim=adult_input_dim,
                hidden_dim=128,
                num_classes=adult_num_classes,
                learning_rate=0.05,
                momentum=0.0,
                split_seed=11,
                training_seed=123,
                batch_size=512,
                eval_batch_size=1024,
                epochs=5,
            ),
            "train_limit": 30000,
            "eval_limit": 10000,
        },
    ]


def run_gpu_scale_validation() -> dict[str, Any]:
    logger = configure_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    patch_sidecar_modules()

    training_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for spec in dataset_specs():
        dataset_name = spec["name"]
        config = spec["config"]
        try:
            logger.info("GPU-scale validation loading dataset bundle for %s", dataset_name)
            bundle = load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)
            bundle = support_scaled.subset_bundle(
                bundle,
                spec["train_limit"],
                spec["eval_limit"],
                seed=777,
            )

            training_case = support_scaled.train_dataset_case(dataset_name, config, bundle, logger)
            record = training_case["record"]
            training_rows.append(
                {
                    "dataset": dataset_name,
                    "status": "ok",
                    "train_size": bundle.train_size,
                    "eval_size": bundle.eval_size,
                    "epsilon_upper_rdp": record.epsilon_upper_theory,
                    "epsilon_upper_tighter": record.epsilon_upper_pld,
                    "upper_bound_backend": record.pld_accounting_backend,
                    "accuracy": record.utility_metrics.get("accuracy"),
                    "loss": record.utility_metrics.get("loss"),
                    "training_elapsed_seconds": training_case["elapsed_seconds"],
                    "training_result_path": training_case["training_result_path"],
                    "training_run_id": record.training_run_id,
                }
            )

            started = time.time()
            passive_row = support_scaled.run_passive_support_level(
                dataset_name=dataset_name,
                attack_name="passive_negative_loss",
                score_type="negative_loss",
                training_case=training_case,
                support_label="gpu_scale",
                query_budget=QUERY_BUDGET,
                audit_seeds=PASSIVE_AUDIT_SEEDS,
            )
            passive_row["elapsed_seconds"] = round(time.time() - started, 3)
            raw_rows.append(passive_row)

            started = time.time()
            canary_row = support_scaled.run_canary_support_level(
                dataset_name=dataset_name,
                attack_name="canary_random",
                training_case=training_case,
                support_label="gpu_scale",
                num_canaries=NUM_CANARIES,
                audit_seeds=CANARY_AUDIT_SEEDS,
            )
            canary_row["elapsed_seconds"] = round(time.time() - started, 3)
            raw_rows.append(canary_row)

            device = torch.device("cuda")
            target_model = load_model_for_inference(
                config.model,
                training_case["record"].model_artifact_path,
                device=device,
            )

            query_train_indices, query_eval_indices, per_seed_train, per_seed_eval = raw_lira.sample_query_indices(
                train_size=len(bundle.train_dataset),
                eval_size=len(bundle.eval_dataset),
            )
            target_train_losses = raw_lira.compute_loss_for_indices(
                target_model,
                bundle.train_dataset,
                query_train_indices,
                device=device,
                batch_size=config.training.eval_batch_size,
            )
            target_eval_losses = raw_lira.compute_loss_for_indices(
                target_model,
                bundle.eval_dataset,
                query_eval_indices,
                device=device,
                batch_size=config.training.eval_batch_size,
            )

            shadow_started = time.time()
            shadow_train_losses, shadow_eval_losses, shadow_member_sets = raw_lira.train_shadow_losses(
                training_case=training_case,
                dataset_name=dataset_name,
                query_train_indices=query_train_indices,
                query_eval_indices=query_eval_indices,
                k_max=RAW_LIRA_K,
                logger=logger,
            )
            logger.info(
                "GPU-scale shadows ready for %s in %.1fs",
                dataset_name,
                time.time() - shadow_started,
            )

            nl_member_scores, nl_nonmember_scores = raw_lira.negative_loss_scores(
                query_train_indices=query_train_indices,
                query_eval_indices=query_eval_indices,
                per_seed_train=per_seed_train,
                per_seed_eval=per_seed_eval,
                target_train_losses=target_train_losses,
                target_eval_losses=target_eval_losses,
            )
            raw_rows.append(
                raw_lira.build_result_row(
                    dataset_name=dataset_name,
                    attack_name="passive_negative_loss_matched",
                    k=RAW_LIRA_K,
                    training_case=training_case,
                    member_scores=nl_member_scores,
                    nonmember_scores=nl_nonmember_scores,
                    score_direction="higher",
                )
            )

            raw_member_scores, raw_nonmember_scores = raw_lira.raw_lira_scores(
                query_train_indices=query_train_indices,
                query_eval_indices=query_eval_indices,
                per_seed_train=per_seed_train,
                per_seed_eval=per_seed_eval,
                target_train_losses=target_train_losses,
                target_eval_losses=target_eval_losses,
                shadow_train_losses=shadow_train_losses,
                shadow_eval_losses=shadow_eval_losses,
                shadow_member_sets=shadow_member_sets,
                k=RAW_LIRA_K,
            )
            raw_rows.append(
                raw_lira.build_result_row(
                    dataset_name=dataset_name,
                    attack_name="passive_raw_lira",
                    k=RAW_LIRA_K,
                    training_case=training_case,
                    member_scores=raw_member_scores,
                    nonmember_scores=raw_nonmember_scores,
                    score_direction="lower",
                )
            )
        except Exception as exc:
            logger.exception("GPU-scale framework validation failed on dataset %s", dataset_name)
            training_rows.append(
                {
                    "dataset": dataset_name,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    normalized_rows = [framework_validation.normalize_attack_row(row) for row in raw_rows]
    checks = framework_validation.build_checks(training_rows, normalized_rows)

    payload = {
        "scope": "gpu-scale full-framework validation pass",
        "cuda_device": torch.cuda.get_device_name(0),
        "training_rows": training_rows,
        "audit_rows": normalized_rows,
        "checks": checks,
        "gpu_scale_parameters": {
            "query_budget_per_seed": QUERY_BUDGET,
            "passive_audit_seeds": PASSIVE_AUDIT_SEEDS,
            "canary_audit_seeds": CANARY_AUDIT_SEEDS,
            "raw_lira_k": RAW_LIRA_K,
            "num_canaries": NUM_CANARIES,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    framework_validation.write_csv(SUMMARY_CSV, normalized_rows)
    CHECKS_JSON.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        framework_validation.build_report(training_rows, normalized_rows, checks),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    require_cuda()
    started = time.time()
    payload = run_gpu_scale_validation()
    elapsed = round(time.time() - started, 3)
    print(f"Wrote validation summary to {SUMMARY_JSON}")
    print(f"Wrote validation rows CSV to {SUMMARY_CSV}")
    print(f"Wrote validation checks to {CHECKS_JSON}")
    print(f"Wrote validation report to {REPORT_MD}")
    print(f"Validation rows: {len(payload['audit_rows'])}, elapsed_seconds={elapsed}")


if __name__ == "__main__":
    main()
