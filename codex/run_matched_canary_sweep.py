"""Matched vs legacy canary sweep.

PURPOSE
-------
Replace the pathological high-sigma canary results (243% / 657% tightness at
sigma=3.0 / 5.0) with honest numbers, and quantify how much of the legacy
canary signal was the appearance artifact.

The legacy design builds inserted and reference canaries from DIFFERENT
distributions (patch shifted by (+7, +11), intensity x0.75), so the auditor can
separate them by appearance without any memorization. The matched design
(strategy prefix ``matched_``) draws all canaries from ONE distribution and
decides membership by a random split (Steinke et al. 2023-style randomized
inclusion), making inserted/held-out scores exchangeable under the null.

DESIGN
------
For each sigma in SIGMAS:
  1. Train one base DP-SGD model (for epsilon_upper RDP + PLD reference).
  2. For each design (legacy, matched) and each audit seed:
     run ``run_one_run_canary_audit`` -- retrains with canaries inserted and
     scores inserted vs reference/held-out canaries.
  3. Pool member/non-member scores across audit seeds per design; estimate
     the conservative Wilson lower bound (2026-07-08: selection-valid holdout
     path is PRIMARY; legacy in-sample retained as diagnostic) and the GDP
     lower bound.

Expected outcome: matched eps_lower <= eps_upper everywhere (no pathology);
the legacy-minus-matched difference measures the artifact's contribution.

OUTPUT
------
codex/results/matched_canary_sweep/
    matched_canary_summary.{json,csv}    per (sigma, design) rows
    matched_canary_scores_<sigma>_<design>.json   pooled raw scores

HOW TO RUN
----------
    python codex/run_matched_canary_sweep.py            # full sweep
    SMOKE=1 python codex/run_matched_canary_sweep.py    # 1 sigma, 2 seeds, 1 epoch
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.auditing.canary.one_run import run_one_run_canary_audit  # noqa: E402
from dp_audit_tightness.config import (  # noqa: E402
    AuditRunConfig,
    CanaryAuditConfig,
    CanaryConfig,
    SaturationConfig,
    TrainExperimentConfig,
)
from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound  # noqa: E402
from dp_audit_tightness.privacy.gdp_estimation import estimate_epsilon_gdp  # noqa: E402
from dp_audit_tightness.training.dp_sgd import train_dp_sgd  # noqa: E402
from dp_audit_tightness.utils.logging_utils import configure_logging  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"

# ---- Sweep knobs ------------------------------------------------------------
SIGMAS = [1.1] if SMOKE else [1.1, 3.0, 5.0]
AUDIT_SEEDS = [401, 402] if SMOKE else [401, 402, 403, 404, 405]
EPOCHS = 1 if SMOKE else 15
NUM_CANARIES = 100 if SMOKE else 1000  # per group (matched) / total unique (legacy)
DELTA = 1e-5
HIDDEN_DIM = 64          # matches the colab_mnist_15ep campaign
# Legacy inserted-count target ~= NUM_CANARIES (54k train examples after split).
LEGACY_INSERTION_RATE = NUM_CANARIES / 54000.0

DESIGNS = {
    "legacy": "random_canaries",
    "matched": "matched_random_canaries",
}

RESULTS_DIR = ROOT / "codex" / "results" / "matched_canary_sweep"


def make_train_config(sigma: float) -> TrainExperimentConfig:
    return TrainExperimentConfig.from_dict(
        {
            "experiment_name": f"mnist_mlp_matched_sweep_s{str(sigma).replace('.', 'p')}",
            "dataset": {
                "name": "mnist",
                "data_dir": str(ROOT / "data" / "raw"),
                "validation_fraction": 0.1,
                "num_workers": 0,
            },
            "model": {
                "name": "simple_mlp",
                "input_dim": 784,
                "hidden_dim": HIDDEN_DIM,
                "num_classes": 10,
            },
            "training": {
                "batch_size": 256,
                "eval_batch_size": 512,
                "epochs": EPOCHS,
                "clipping_norm": 1.0,
                "noise_multiplier": sigma,
                "sampling_rate": None,
                "optimizer": {
                    "name": "sgd",
                    "learning_rate": 0.15,
                    "weight_decay": 0.0,
                    "momentum": 0.0,
                },
            },
            "privacy": {"delta": DELTA, "accountant": "rdp"},
            "run": {
                "split_seed": 11,
                "training_seed": 123,
                "results_root": str(RESULTS_DIR / "training"),
                "save_checkpoint": False,
                "notes": "Base model for matched-vs-legacy canary sweep.",
            },
        }
    )


def make_canary_config(strategy: str) -> CanaryAuditConfig:
    return CanaryAuditConfig(
        training_result_path="__in_memory__",
        audit_regime="evaluator_controlled_canary_stress_test",
        auditor_variant=strategy,
        auditor_strength_rank=2,
        audit_mode="repeated_run",
        delta=DELTA,
        repeated_seeds=list(AUDIT_SEEDS),
        canary=CanaryConfig(
            strategy=strategy,
            num_canaries=NUM_CANARIES,
            insertion_rate=LEGACY_INSERTION_RATE,
            optimize_steps=0,
        ),
        saturation=SaturationConfig(),
        run=AuditRunConfig(results_root=str(RESULTS_DIR), save_debug_artifacts=False),
    )


def estimate_row(member_scores, nonmember_scores, eps_upper_rdp, eps_upper_pld):
    common = dict(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=DELTA,
        score_direction="higher",
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
    )
    # Primary: sample-split holdout (2026-07-06 fix #1) -> valid 95% conservative bound.
    conservative = estimate_empirical_lower_bound(
        threshold_selection="holdout",
        **common,
    )
    # Diagnostic only: legacy in-sample path (anti-conservative under selection;
    # retained for diffing against pre-fix runs, never quotable).
    insample = estimate_empirical_lower_bound(
        threshold_selection="in_sample",
        **common,
    )
    gdp = estimate_epsilon_gdp(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=DELTA,
        score_direction="higher",
        n_bootstrap=2000,
    )
    eps_lower = conservative.epsilon_lower_empirical
    return {
        "epsilon_lower_conservative": eps_lower,
        "epsilon_lower_point": conservative.epsilon_lower_empirical_point_estimate,
        "threshold_selection": conservative.threshold_selection,
        "holdout_selection_half_sizes": json.dumps(conservative.selection_half_sizes),
        "holdout_estimation_half_sizes": json.dumps(conservative.estimation_half_sizes),
        "epsilon_lower_conservative_insample": insample.epsilon_lower_empirical,
        "epsilon_lower_point_insample": insample.epsilon_lower_empirical_point_estimate,
        "epsilon_lower_gdp": gdp.epsilon_lower,
        "epsilon_lower_gdp_point": gdp.epsilon_lower_point,
        "gdp_auc": gdp.auc,
        "selected_tpr": conservative.selected_tpr,
        "selected_fpr": conservative.selected_fpr,
        "warning": conservative.warning_message,
        "tightness_rdp": (eps_lower / eps_upper_rdp) if eps_upper_rdp else None,
        "tightness_pld": (eps_lower / eps_upper_pld) if eps_upper_pld else None,
        "pathological": bool(eps_upper_pld and eps_lower and eps_lower > eps_upper_pld),
        "num_member_samples": len(member_scores),
        "num_nonmember_samples": len(nonmember_scores),
        "member_score_mean": statistics.fmean(member_scores) if member_scores else None,
        "nonmember_score_mean": statistics.fmean(nonmember_scores) if nonmember_scores else None,
    }


def main() -> None:
    logger = configure_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for sigma in SIGMAS:
        train_config = make_train_config(sigma)
        logger.info("Training base model sigma=%s epochs=%s", sigma, EPOCHS)
        started = time.time()
        outcome = train_dp_sgd(
            train_config,
            logger,
            save_checkpoint=False,
            run_descriptor=f"matched_sweep_base_s{sigma}",
            return_model_state=True,
        )
        record = outcome.record
        eps_rdp = record.epsilon_upper_theory
        eps_pld = record.epsilon_upper_pld
        logger.info(
            "Base model sigma=%s eps_rdp=%.4f eps_pld=%s acc=%s (%.0fs)",
            sigma, eps_rdp, eps_pld, record.utility_metrics.get("accuracy"),
            time.time() - started,
        )

        for design, strategy in DESIGNS.items():
            canary_config = make_canary_config(strategy)
            all_member: list[float] = []
            all_nonmember: list[float] = []
            for audit_seed in AUDIT_SEEDS:
                obs = run_one_run_canary_audit(
                    record, train_config, canary_config, audit_seed, logger=logger
                )
                all_member.extend(obs.member_scores)
                all_nonmember.extend(obs.nonmember_scores)
                logger.info(
                    "sigma=%s design=%s seed=%s gap=%.4f",
                    sigma, design, audit_seed,
                    obs.raw_statistics["score_mean_gap"],
                )

            scores_path = RESULTS_DIR / f"matched_canary_scores_{sigma}_{design}.json"
            with scores_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"member_scores": all_member, "nonmember_scores": all_nonmember},
                    handle,
                )

            row = {
                "sigma": sigma,
                "design": design,
                "strategy": strategy,
                "epochs": EPOCHS,
                "num_canaries_per_group": NUM_CANARIES,
                "audit_seeds": json.dumps(AUDIT_SEEDS),
                "epsilon_upper_rdp": eps_rdp,
                "epsilon_upper_pld": eps_pld,
                "base_accuracy": record.utility_metrics.get("accuracy"),
            }
            row.update(estimate_row(all_member, all_nonmember, eps_rdp, eps_pld))
            summary.append(row)
            logger.info(
                "RESULT sigma=%s design=%s eps_lower=%s tightness_pld=%s pathological=%s",
                sigma, design,
                row["epsilon_lower_conservative"],
                row["tightness_pld"],
                row["pathological"],
            )

    summary_json = RESULTS_DIR / "matched_canary_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    fieldnames: list[str] = []
    for row in summary:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (RESULTS_DIR / "matched_canary_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nWrote {summary_json}")
    print("\nsigma  design   eps_lower(cons)  eps_upper(PLD)  tightness  pathological")
    for row in summary:
        print(
            f"{row['sigma']:<6} {row['design']:<8} "
            f"{row['epsilon_lower_conservative']}  {row['epsilon_upper_pld']}  "
            f"{row['tightness_pld']}  {row['pathological']}"
        )


if __name__ == "__main__":
    main()
