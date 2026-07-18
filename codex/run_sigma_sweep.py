"""Sigma sweep v2 -- gap decomposition with the CORRECTED symmetric LiRA scorer.

WHY A v2
--------
The original sweep (autoresearch/notebooks/colab_sigma_sweep.ipynb, git blob
4ca8e88, since deleted) scored members as mean(OUT-shadow loss) - mean(IN-shadow
loss) WITHOUT ever querying the target model, while non-members were scored as
mean(shadow loss) - target loss. Two different formulas on two different scales
separate by construction, so every eps_lower in the old findings.md table is
untrustworthy (and the sigma=5.0 eps_lower > eps_PLD anomaly is fully explained).

This runner reuses the already-debugged scoring/estimation path from
``run_raw_lira_pilot.py`` (symmetric scorer: score = mean(OUT-shadow loss) -
target loss, identical for members and non-members, score_direction="higher").
LiRA scoring is NOT reimplemented here.

DESIGN (matches the original sweep unless flagged v2)
-----------------------------------------------------
- Full MNIST (no subsetting), 1 epoch, simple_mlp 784-64-10, batch 256, clip 1.0,
  delta 1e-5, sigma in {0.5, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0}.
- K = 32 shadow models per sigma, trained once and reused; shadows inherit the
  target's noise_multiplier/batch_size (v2 fix -- the old helper hardcoded 1.1/128).
- 5 audit seeds. v2: query budget raised 256 -> 2048 per seed (pooled ~5120/5120)
  because at the old budget the Wilson detection floor sits above every
  eps_PLD <= 1.0 in the grid, making sub-1-eps cells structurally uncertifiable.
- v2: per-seed member/non-member draws are DISJOINT slices, so pooled Wilson
  counts are over distinct examples (the old draws resampled the same pools).
- Three estimators per cell: Wilson-conservative (headline), point estimate,
  GDP (diagnostic ONLY, never the headline bound -- per the 2026-07-02 code
  audit its Gaussianity assumption fails for DP-SGD score distributions).
- Validity gate + censored flag via pilot.apply_quality_flags.

NOTE ON COMPARABILITY WITH THE OLD TABLE
----------------------------------------
The old sweep's training split came from autoresearch/prepare.py (not in this
repo; sampling_rate 0.00449 implies ~57k train). This runner uses the repo's own
loader (validation_fraction=0.1 -> 54000 train / 6000 eval), so eps_upper values
will shift slightly. Both eps_RDP and eps_PLD are recomputed in-run, so the new
table is self-consistent. Expected: the sigma=0.5 accounting share (~93%) should
survive (verified insensitive to eps_lower on 2026-07-05); the sigma~2.0 shares
are sensitive and are the reason for this rerun.

OUTPUT
------
codex/results/sigma_sweep_v2/
    sigma_sweep_summary.json      full per-sigma rows
    sigma_sweep_summary.csv       same, flat
    sigma_sweep_decomposition.md  findings.md-style decomposition table

HOW TO RUN
----------
    python codex/run_sigma_sweep.py
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

# ---- Sweep knobs ------------------------------------------------------------
SIGMAS = [0.5, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0]
K_SHADOWS = 32
QUERY_BUDGET = 2048  # per audit seed: 1024 members + 1024 non-members
AUDIT_SEEDS = [401, 402, 403, 404, 405]
BATCH_SIZE = 256     # matches the original sweep's training setup
TRAINING_SEED = 123

RESULTS_DIR = CODEX / "results" / "sigma_sweep_v2"
SUMMARY_JSON = RESULTS_DIR / "sigma_sweep_summary.json"
SUMMARY_CSV = RESULTS_DIR / "sigma_sweep_summary.csv"
DECOMPOSITION_MD = RESULTS_DIR / "sigma_sweep_decomposition.md"


def _apply_pilot_overrides() -> None:
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


def _decomposition_table(rows: list[dict]) -> str:
    """findings.md-style decomposition table so old and new diff cleanly."""
    lines = [
        "# Sigma Sweep v2 -- Gap Decomposition (corrected symmetric scorer)",
        "",
        "| sigma | eps_rdp | eps_pld | eps_lower | tr(RDP) | tr(PLD) | Acct gap | Audit gap | Acct% | Aud% | Accuracy | validity | censored |",
        "|-------|---------|---------|-----------|---------|---------|----------|-----------|-------|------|----------|----------|----------|",
    ]
    for r in sorted(rows, key=lambda r: r["sigma"]):
        rdp = r.get("epsilon_upper_rdp")
        pld = r.get("epsilon_upper_tighter")
        low = r.get("epsilon_lower_conservative") or 0.0
        if rdp is None or pld is None:
            continue
        acct_gap = rdp - pld
        audit_gap = pld - low
        total_gap = rdp - low
        acct_pct = 100.0 * acct_gap / total_gap if total_gap > 0 else float("nan")
        aud_pct = 100.0 * audit_gap / total_gap if total_gap > 0 else float("nan")
        lines.append(
            f"| {r['sigma']} | {rdp:.2f} | {pld:.2f} | {low:.3f} "
            f"| {100.0 * low / rdp:.1f}% | {100.0 * low / pld:.1f}% "
            f"| {acct_gap:.2f} | {audit_gap:.2f} | {acct_pct:.0f}% | {aud_pct:.0f}% "
            f"| {100.0 * (r.get('accuracy') or 0.0):.1f}% "
            f"| {r.get('validity', '?')} | {r.get('censored', '?')} |"
        )
    lines += [
        "",
        "Censored cells: Wilson-conservative returned 0 while the point/GDP",
        "estimate is positive -- the true leakage is below the estimator's",
        "detection floor at this budget. Report as 'below detection floor',",
        "never as 'attack recovers 0%'. GDP columns in the CSV are diagnostic",
        "only and never a headline bound.",
    ]
    return "\n".join(lines)


def main() -> None:
    import torch

    logger = configure_logging()
    _apply_pilot_overrides()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading FULL MNIST bundle (no subsetting)")
    probe_config = pilot.make_train_config(
        dataset_name="mnist",
        data_dir=str(ROOT / "data" / "raw"),
        model_name="simple_mlp",
        input_dim=784,
        hidden_dim=64,
        num_classes=10,
        learning_rate=0.15,
        momentum=0.0,
        training_seed=TRAINING_SEED,
    )
    bundle = load_dataset_bundle(probe_config.dataset, split_seed=probe_config.run.split_seed)
    logger.info("MNIST full train=%s eval=%s", bundle.train_size, bundle.eval_size)

    # Disjoint per-seed query draws, fixed ONCE and reused for every sigma so
    # the only thing that varies across cells is sigma.
    (query_train_indices, query_eval_indices,
     per_seed_train, per_seed_eval) = pilot.sample_query_indices_disjoint(
        train_size=len(bundle.train_dataset),
        eval_size=len(bundle.eval_dataset),
    )

    summary: list[dict] = []

    for sigma in SIGMAS:
        sweep_started = time.time()
        logger.info("=== sigma = %s ===", sigma)

        config = pilot.make_train_config(
            dataset_name="mnist",
            data_dir=str(ROOT / "data" / "raw"),
            model_name="simple_mlp",
            input_dim=784,
            hidden_dim=64,
            num_classes=10,
            learning_rate=0.15,
            momentum=0.0,
            training_seed=TRAINING_SEED,
        )
        # Sweep overrides (configs are mutable dataclasses).
        config.training.noise_multiplier = float(sigma)
        config.training.batch_size = BATCH_SIZE
        config.experiment_name = f"codex_sigma_sweep_{str(sigma).replace('.', 'p')}"

        # 1) Target model at this sigma.
        training_case = pilot.train_target_case("mnist", config, bundle, logger)
        target_model = load_model_for_inference(
            config.model, training_case["record"].model_artifact_path, device=device
        )
        target_train_losses = pilot.compute_loss_for_indices(
            target_model, bundle.train_dataset, query_train_indices, device=device
        )
        target_eval_losses = pilot.compute_loss_for_indices(
            target_model, bundle.eval_dataset, query_eval_indices, device=device
        )

        # 2) K=32 shadows at this sigma (train_shadow_losses inherits the
        #    target's noise_multiplier/batch_size -- v2 fix).
        shadows_started = time.time()
        shadow_train_losses, shadow_eval_losses, shadow_member_sets = pilot.train_shadow_losses(
            training_case=training_case,
            dataset_name="mnist",
            query_train_indices=query_train_indices,
            query_eval_indices=query_eval_indices,
            k_max=K_SHADOWS,
            logger=logger,
        )
        logger.info(
            "sigma=%s: %s shadows in %.1fs",
            sigma, K_SHADOWS, time.time() - shadows_started,
        )

        # 3) Corrected symmetric Raw-LiRA scores (imported, not reimplemented).
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
            k=K_SHADOWS,
        )

        # 4) Three estimators + gates, all inside build_result_row/apply_quality_flags.
        row = pilot.build_result_row(
            dataset_name="mnist",
            attack_name="passive_raw_lira",
            k=K_SHADOWS,
            training_case=training_case,
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            score_direction="higher",
        )
        row["sigma"] = sigma
        row["accuracy"] = training_case["record"].utility_metrics.get("accuracy")
        row["sweep_elapsed_seconds"] = round(time.time() - sweep_started, 1)
        row = pilot.apply_quality_flags(row)
        summary.append(row)

        logger.info(
            "sigma=%s eps_rdp=%.4f eps_pld=%.4f eps_lower=%.4f validity=%s censored=%s",
            sigma,
            row.get("epsilon_upper_rdp") or float("nan"),
            row.get("epsilon_upper_tighter") or float("nan"),
            row.get("epsilon_lower_conservative") or 0.0,
            row["validity"],
            row["censored"],
        )

        # Persist after EVERY sigma so a crash late in the sweep loses nothing.
        _write_summary(summary)
        DECOMPOSITION_MD.write_text(_decomposition_table(summary), encoding="utf-8")

    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DECOMPOSITION_MD}")
    print()
    print(_decomposition_table(summary))


if __name__ == "__main__":
    main()
