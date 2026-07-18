from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODEX_DIR = ROOT / "codex"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

import run_raw_lira_pilot as raw_lira
import run_support_scaled_pilot as support_scaled
from dp_audit_tightness.data.datasets import load_dataset_bundle
from dp_audit_tightness.models.io import load_model_for_inference
from dp_audit_tightness.utils.logging_utils import configure_logging


RESULTS_DIR = CODEX_DIR / "results" / "final_paper_checks"
SUMMARY_JSON = RESULTS_DIR / "final_paper_checks.json"
SUMMARY_CSV = RESULTS_DIR / "final_paper_checks.csv"
REPORT_MD = RESULTS_DIR / "final_paper_checks.md"
TABLE_TEX = RESULTS_DIR / "final_paper_checks_table.tex"

TRAIN_LIMIT = 2048
EVAL_LIMIT = 512
QUERY_BUDGET = 256
PASSIVE_AUDIT_SEEDS = [401, 402]
RAW_LIRA_K_VALUES = [8, 16]
CANARY_AUDIT_SEEDS = [101, 102]
NUM_CANARIES = 16


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
    raw_lira.K_VALUES = list(RAW_LIRA_K_VALUES)


def safe_ratio(lower: float | None, upper: float | None) -> float | None:
    if lower is None or upper is None or upper <= 0:
        return None
    return lower / upper


def paper_row(
    *,
    check_name: str,
    status: str,
    row: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    epsilon_upper = row.get("epsilon_upper_tighter")
    epsilon_lower = row.get("epsilon_lower_conservative")
    return {
        "check_name": check_name,
        "status": status,
        "dataset": row.get("dataset", "mnist"),
        "attack": row.get("attack"),
        "support": row.get("support_label") or (
            f"K={row.get('k_shadows')}" if row.get("k_shadows") else None
        ),
        "epsilon_upper_pld": epsilon_upper,
        "epsilon_upper_rdp": row.get("epsilon_upper_rdp"),
        "epsilon_lower_wilson": epsilon_lower,
        "epsilon_lower_point": row.get("epsilon_lower_point"),
        "tightness_pld": safe_ratio(epsilon_lower, epsilon_upper),
        "selected_tpr": row.get("selected_tpr"),
        "selected_fpr": row.get("selected_fpr"),
        "num_member_samples": row.get("num_member_samples"),
        "num_nonmember_samples": row.get("num_nonmember_samples"),
        "gdp_valid_lower_bound": row.get("gdp_valid_lower_bound"),
        "gdp_auc": row.get("gdp_auc"),
        "warning": row.get("warning") or row.get("gdp_warning"),
        "notes": notes,
    }


def write_outputs(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Final Paper Quick Checks",
        "",
        "These are intentionally small MNIST checks for final-report support, not a full sweep.",
        "",
        "| Check | Status | Attack | Support | eps upper PLD | eps lower Wilson | Tightness | Note |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        upper = row.get("epsilon_upper_pld")
        lower = row.get("epsilon_lower_wilson")
        ratio = row.get("tightness_pld")
        lines.append(
            "| "
            f"{row['check_name']} | {row['status']} | {row.get('attack') or ''} | "
            f"{row.get('support') or ''} | "
            f"{upper:.4f}" if isinstance(upper, float) else "| "
        )
        lines[-1] = (
            f"| {row['check_name']} | {row['status']} | {row.get('attack') or ''} | "
            f"{row.get('support') or ''} | "
            f"{upper:.4f}" if isinstance(upper, float) else f"| {row['check_name']} | {row['status']} | {row.get('attack') or ''} | {row.get('support') or ''} | "
        )
        lines[-1] += f" | {lower:.4f}" if isinstance(lower, float) else " | "
        lines[-1] += f" | {ratio:.1%}" if isinstance(ratio, float) else " | "
        lines[-1] += f" | {row.get('notes') or ''} |"

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_rows = [
        row
        for row in rows
        if row["check_name"]
        in {
            "train_accounting_smoke",
            "passive_budget_smoke",
            "canary_smoke",
            "raw_lira_k_ladder",
        }
    ]
    tex_lines = [
        "\\begin{table}[h]",
        "\\caption{Final quick validation checks on MNIST. These checks are intentionally small and are used to validate framework behavior rather than to claim final scientific tightness.}",
        "\\label{tab:final_quick_checks}",
        "\\centering",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "\\textbf{Check} & \\textbf{Attack} & \\textbf{Support} & $\\boldsymbol{\\varepsilon_{\\mathrm{upper}}}$ & $\\boldsymbol{\\varepsilon_{\\mathrm{lower}}}$ & \\textbf{Tightness} \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        upper = _tex_num(row.get("epsilon_upper_pld"))
        lower = _tex_num(row.get("epsilon_lower_wilson"))
        ratio = _tex_pct(row.get("tightness_pld"))
        tex_lines.append(
            f"{_tex_escape(str(row['check_name']))} & "
            f"{_tex_escape(str(row.get('attack') or ''))} & "
            f"{_tex_escape(str(row.get('support') or ''))} & "
            f"{upper} & {lower} & {ratio} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    TABLE_TEX.write_text("\n".join(tex_lines), encoding="utf-8")


def _tex_num(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else "--"


def _tex_pct(value: Any) -> str:
    return f"{100.0 * value:.1f}\\%" if isinstance(value, float) else "--"


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def main() -> None:
    import torch

    started_all = time.time()
    logger = configure_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    patch_sidecar_modules()

    config = support_scaled.make_train_config(
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
    )
    bundle = load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)
    bundle = support_scaled.subset_bundle(bundle, TRAIN_LIMIT, EVAL_LIMIT, seed=777)
    training_case = support_scaled.train_dataset_case("mnist", config, bundle, logger)
    record = training_case["record"]

    rows: list[dict[str, Any]] = [
        {
            "check_name": "train_accounting_smoke",
            "status": "pass"
            if record.epsilon_upper_pld is not None
            and record.epsilon_upper_pld <= record.epsilon_upper_theory
            else "fail",
            "dataset": "mnist",
            "attack": "__training__",
            "support": f"train={TRAIN_LIMIT}, eval={EVAL_LIMIT}",
            "epsilon_upper_pld": record.epsilon_upper_pld,
            "epsilon_upper_rdp": record.epsilon_upper_theory,
            "epsilon_lower_wilson": None,
            "epsilon_lower_point": None,
            "tightness_pld": None,
            "selected_tpr": None,
            "selected_fpr": None,
            "num_member_samples": None,
            "num_nonmember_samples": None,
            "gdp_valid_lower_bound": None,
            "gdp_auc": None,
            "warning": None,
            "notes": "PLD upper bound should be positive and no larger than RDP.",
        }
    ]

    passive_small = support_scaled.run_passive_support_level(
        dataset_name="mnist",
        attack_name="passive_negative_loss",
        score_type="negative_loss",
        training_case=training_case,
        support_label="budget_128x2",
        query_budget=128,
        audit_seeds=PASSIVE_AUDIT_SEEDS,
    )
    rows.append(
        paper_row(
            check_name="passive_budget_smoke",
            status="pass",
            row=passive_small,
            notes="Passive baseline should remain conservative; zero is acceptable.",
        )
    )

    canary_row = support_scaled.run_canary_support_level(
        dataset_name="mnist",
        attack_name="canary_random",
        training_case=training_case,
        support_label="16_canaries_x2",
        num_canaries=NUM_CANARIES,
        audit_seeds=CANARY_AUDIT_SEEDS,
    )
    rows.append(
        paper_row(
            check_name="canary_smoke",
            status="pass" if canary_row.get("status") == "ok" else "fail",
            row=canary_row,
            notes="Evaluator-controlled track executes and records conservative Wilson support.",
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = load_model_for_inference(
        config.model,
        record.model_artifact_path,
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
    )
    target_eval_losses = raw_lira.compute_loss_for_indices(
        target_model,
        bundle.eval_dataset,
        query_eval_indices,
        device=device,
    )
    shadow_train_losses, shadow_eval_losses, shadow_member_sets = raw_lira.train_shadow_losses(
        training_case=training_case,
        dataset_name="mnist",
        query_train_indices=query_train_indices,
        query_eval_indices=query_eval_indices,
        k_max=max(RAW_LIRA_K_VALUES),
        logger=logger,
    )

    raw_rows: list[dict[str, Any]] = []
    for k in RAW_LIRA_K_VALUES:
        member_scores, nonmember_scores = raw_lira.raw_lira_scores(
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
        raw_row = raw_lira.build_result_row(
            dataset_name="mnist",
            attack_name="passive_raw_lira",
            k=k,
            training_case=training_case,
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            score_direction="lower",
        )
        raw_rows.append(raw_row)
        rows.append(
            paper_row(
                check_name="raw_lira_k_ladder",
                status="pass",
                row=raw_row,
                notes="Raw LiRA should produce explicit K-dependent Wilson and GDP diagnostic fields.",
            )
        )

    if raw_rows:
        k8, k16 = raw_rows[0], raw_rows[-1]
        rows.append(
            {
                "check_name": "raw_lira_k_sanity",
                "status": "pass"
                if (k16.get("epsilon_lower_conservative") or 0.0)
                >= 0.5 * (k8.get("epsilon_lower_conservative") or 0.0)
                else "warn",
                "dataset": "mnist",
                "attack": "passive_raw_lira",
                "support": "K=8->16",
                "epsilon_upper_pld": record.epsilon_upper_pld,
                "epsilon_upper_rdp": record.epsilon_upper_theory,
                "epsilon_lower_wilson": k16.get("epsilon_lower_conservative"),
                "epsilon_lower_point": k16.get("epsilon_lower_point"),
                "tightness_pld": safe_ratio(
                    k16.get("epsilon_lower_conservative"),
                    record.epsilon_upper_pld,
                ),
                "selected_tpr": k16.get("selected_tpr"),
                "selected_fpr": k16.get("selected_fpr"),
                "num_member_samples": k16.get("num_member_samples"),
                "num_nonmember_samples": k16.get("num_nonmember_samples"),
                "gdp_valid_lower_bound": k16.get("gdp_valid_lower_bound"),
                "gdp_auc": k16.get("gdp_auc"),
                "warning": k16.get("warning") or k16.get("gdp_warning"),
                "notes": "Checks K scaling is not catastrophically unstable on the quick run.",
            }
        )

    rows.append(
        {
            "check_name": "runtime",
            "status": "info",
            "dataset": "mnist",
            "attack": "__runtime__",
            "support": f"device={device}",
            "epsilon_upper_pld": None,
            "epsilon_upper_rdp": None,
            "epsilon_lower_wilson": None,
            "epsilon_lower_point": None,
            "tightness_pld": None,
            "selected_tpr": None,
            "selected_fpr": None,
            "num_member_samples": None,
            "num_nonmember_samples": None,
            "gdp_valid_lower_bound": None,
            "gdp_auc": None,
            "warning": None,
            "notes": f"elapsed_seconds={time.time() - started_all:.1f}",
        }
    )

    write_outputs(rows)
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {TABLE_TEX}")


if __name__ == "__main__":
    main()
