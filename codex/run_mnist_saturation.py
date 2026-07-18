"""MNIST Raw-LiRA saturation sweep.

PURPOSE
-------
Answer one question cleanly: as we give the passive Raw-LiRA auditor more
shadow models (K), does the empirical lower bound eps_lower keep climbing, or
does it plateau?

This matters because our smoke results showed eps_lower jumping from ~4% (K=8)
to ~39% (K=16). If eps_lower is still climbing at large K, then a large part of
the audit gap is "weak auditor / not enough compute" -- NOT loose accounting or
a weak threat model. We can only attribute the residual gap to the bound once
the auditor has saturated (eps_lower stops moving with more K).

DESIGN
------
- MNIST only -- the single trustworthy canonical line.
- Everything held fixed (model, noise, dataset subset, seeds, query budget);
  ONLY K is varied.
- We REUSE the already-debugged scoring/estimation functions from
  ``run_raw_lira_pilot.py`` (corrected symmetric LiRA: score = mean(OUT-shadow
  loss) - target loss, applied identically to members and non-members).
- Shadow models are trained once at the largest K and reused for every smaller K.

OUTPUT
------
codex/results/mnist_saturation/
    mnist_saturation_summary.json   full per-K rows
    mnist_saturation_summary.csv    same, flat
    mnist_saturation_verdict.json   consecutive eps_lower deltas + plateau verdict

HOW TO RUN
----------
    python codex/run_mnist_saturation.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODEX = ROOT / "codex"
for path in (str(SRC), str(CODEX)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_raw_lira_pilot as pilot  # noqa: E402
from dp_audit_tightness.data.datasets import load_dataset_bundle  # noqa: E402
from dp_audit_tightness.models.io import load_model_for_inference  # noqa: E402
from dp_audit_tightness.utils.logging_utils import configure_logging  # noqa: E402

# ---- Saturation knobs (the only thing that varies is K) --------------------
# v3 (2026-07-05): K extended to 512 because the 2026-07-02 run was STILL
# CLIMBING at K=256 (+0.082 nats on the last doubling, threshold 0.05).
# Add 1024 only if 512 still climbs: shadow training time scales linearly in
# K, so 1024 roughly doubles the total shadow-training stage vs 512.
K_LADDER = [2, 4, 8, 16, 32, 64, 128, 256, 512]  # extend with 1024 if needed
QUERY_BUDGET = 512
AUDIT_SEEDS = [401, 402, 403, 404, 405]
TRAIN_LIMIT = 2048
# v3: was 512, which forced the 5 audit seeds to resample from only 512
# distinct eval points (256 draws/seed x 5 seeds) -> pooled non-member counts
# were not independent and the Wilson CI was anti-conservative. 2560 gives
# 2x headroom over the 1280 distinct points needed for disjoint draws.
EVAL_LIMIT = 2560
PLATEAU_DELTA_NATS = 0.05

RESULTS_DIR = CODEX / "results" / "mnist_saturation"
SUMMARY_JSON = RESULTS_DIR / "mnist_saturation_summary.json"
SUMMARY_CSV = RESULTS_DIR / "mnist_saturation_summary.csv"
VERDICT_JSON = RESULTS_DIR / "mnist_saturation_verdict.json"


def _apply_pilot_overrides() -> None:
    """Point the reused pilot helpers at our saturation config + output dir."""
    pilot.K_VALUES = list(K_LADDER)
    pilot.QUERY_BUDGET = QUERY_BUDGET
    pilot.AUDIT_SEEDS = list(AUDIT_SEEDS)
    pilot.RESULTS_DIR = RESULTS_DIR


def _write_summary(summary: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    fieldnames: list[str] = []
    for row in summary:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def _saturation_verdict(rows: list[dict]) -> dict:
    """Compute consecutive eps_lower deltas and decide if the curve plateaued.

    Only cells that pass the validity gate (eps_lower <= eps_upper) are used, so
    a pathological cell can never drive the plateau verdict.
    """
    lira_rows = [
        r for r in rows
        if r.get("attack") == "passive_raw_lira"
        and r.get("status") == "ok"
        and r.get("validity", "ok") == "ok"
        # v3: censored cells (conservative 0 with positive point/GDP estimate)
        # carry no information about auditor strength -- excluding them stops a
        # detection-floor artifact from faking a plateau at eps_lower = 0.
        and r.get("censored", "not_censored") == "not_censored"
    ]
    lira_rows.sort(key=lambda r: r["k_shadows"])
    ladder = []
    prev = None
    for r in lira_rows:
        eps = r.get("epsilon_lower_conservative")
        delta = None if prev is None or eps is None else round(eps - prev, 6)
        ladder.append({
            "k_shadows": r["k_shadows"],
            "epsilon_lower_conservative": eps,
            "epsilon_lower_gdp": r.get("epsilon_lower_gdp"),
            "tightness_ratio_tighter": r.get("tightness_ratio_tighter"),
            "delta_vs_prev_k": delta,
        })
        prev = eps if eps is not None else prev
    last_delta = ladder[-1]["delta_vs_prev_k"] if len(ladder) >= 2 else None
    # Fix #3 (2026-07-06): with <2 non-censored valid cells there is no ladder to
    # assess, so the verdict must say so -- NOT fall through to "STILL CLIMBING",
    # which previously masqueraded a default as a measured finding.
    if len(ladder) < 2:
        plateaued = None
        verdict = "NO VERDICT -- fewer than 2 non-censored valid cells; cannot assess plateau"
    else:
        plateaued = last_delta is not None and abs(last_delta) < PLATEAU_DELTA_NATS
        verdict = (
            "PLATEAUED -- residual gap can now be attributed to bound/threat-model"
            if plateaued
            else "STILL CLIMBING -- gap is still partly weak-auditor; scale K further before attributing"
        )
    return {
        "plateau_delta_threshold_nats": PLATEAU_DELTA_NATS,
        "last_delta_vs_prev_k": last_delta,
        "num_ladder_cells": len(ladder),
        "plateaued": plateaued,
        "verdict": verdict,
        "ladder": ladder,
    }


def main() -> None:
    import torch

    logger = configure_logging()
    _apply_pilot_overrides()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = pilot.make_train_config(
        dataset_name="mnist",
        data_dir=str(ROOT / "data" / "raw"),
        model_name="simple_mlp",
        input_dim=784,
        hidden_dim=64,
        num_classes=10,
        learning_rate=0.15,
        momentum=0.0,
        training_seed=123,
    )

    logger.info("Loading MNIST bundle")
    bundle = load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)
    bundle = pilot.subset_bundle(bundle, TRAIN_LIMIT, EVAL_LIMIT, seed=777)
    logger.info("MNIST subset train=%s eval=%s", bundle.train_size, bundle.eval_size)

    summary: list[dict] = []

    # 1) Train the single target model.
    training_case = pilot.train_target_case("mnist", config, bundle, logger)
    summary.append({
        "dataset": "mnist",
        "attack": "__training__",
        "status": "ok",
        "train_size": bundle.train_size,
        "eval_size": bundle.eval_size,
        "epsilon_upper_rdp": training_case["record"].epsilon_upper_theory,
        "epsilon_upper_tighter": training_case["record"].epsilon_upper_pld,
        "accuracy": training_case["record"].utility_metrics.get("accuracy"),
        "training_elapsed_seconds": training_case["elapsed_seconds"],
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = load_model_for_inference(
        config.model, training_case["record"].model_artifact_path, device=device
    )

    # 2) Sample the fixed query set and target losses.
    # v3: disjoint per-seed draws (see pilot.sample_query_indices_disjoint) so
    # pooled Wilson counts are over distinct examples.
    (query_train_indices, query_eval_indices,
     per_seed_train, per_seed_eval) = pilot.sample_query_indices_disjoint(
        train_size=len(bundle.train_dataset),
        eval_size=len(bundle.eval_dataset),
    )
    target_train_losses = pilot.compute_loss_for_indices(
        target_model, bundle.train_dataset, query_train_indices, device=device
    )
    target_eval_losses = pilot.compute_loss_for_indices(
        target_model, bundle.eval_dataset, query_eval_indices, device=device
    )

    # 3) Train the shadow models once at the largest K; reuse for every K.
    started = time.time()
    shadow_train_losses, shadow_eval_losses, shadow_member_sets = pilot.train_shadow_losses(
        training_case=training_case,
        dataset_name="mnist",
        query_train_indices=query_train_indices,
        query_eval_indices=query_eval_indices,
        k_max=max(K_LADDER),
        logger=logger,
    )
    logger.info("Trained %s shadow models in %.1fs", max(K_LADDER), time.time() - started)

    # 4) Sweep K, scoring Raw LiRA against the same shadows each time.
    for k in K_LADDER:
        member_scores, nonmember_scores = pilot.raw_lira_scores(
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
        row = pilot.build_result_row(
            dataset_name="mnist",
            attack_name="passive_raw_lira",
            k=k,
            training_case=training_case,
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            # Corrected symmetric LiRA (mean(OUT) - target_loss) makes MEMBERS
            # score HIGHER, so the estimator event is score>=threshold.
            score_direction="higher",
        )
        # v3: validity gate + detection-floor censoring, shared with the sigma
        # sweep via pilot.apply_quality_flags.
        row = pilot.apply_quality_flags(row)
        summary.append(row)
        logger.info(
            "K=%s eps_lower=%.4f tightness=%.3f validity=%s censored=%s",
            k,
            row.get("epsilon_lower_conservative") or 0.0,
            row["tightness_ratio_tighter"] or 0.0,
            row["validity"],
            row["censored"],
        )

    _write_summary(summary)
    verdict = _saturation_verdict(summary)
    with VERDICT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)

    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {VERDICT_JSON}")
    print("\nSaturation verdict:", verdict["verdict"])
    for step in verdict["ladder"]:
        print(
            f"  K={step['k_shadows']:>3}  eps_lower={step['epsilon_lower_conservative']}"
            f"  delta={step['delta_vs_prev_k']}"
        )


if __name__ == "__main__":
    main()
