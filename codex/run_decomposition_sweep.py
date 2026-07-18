"""E4 decomposition sweep — the paper's Table 1 (EXPERIMENT_PLAN_2026-07-08 §E4).

PURPOSE
-------
For every tuned (dataset, target_epsilon) cell produced by E1
(``results/hpo/best_configs/*.yaml``), audit the SAME final models with both
threat models and both estimators, then attribute the total RDP-vs-empirical
gap to its sources:

    D = eps_upper_RDP - eps_lower[passive, wilson_holdout]        (total gap)

    accounting_share = (eps_upper_RDP - eps_upper_PLD)                  / D
    threat_share     = (eps_lower[canary,w]   - eps_lower[passive,w])   / D
    estimator_share  = (eps_lower[canary,gdp] - eps_lower[canary,w])    / D
    residual_share   = (eps_upper_PLD         - eps_lower[canary,gdp])  / D

The four terms TELESCOPE exactly to 1 (plan §E4 acceptance: 1 +/- 0.02).
When the GDP bound violates the validity gate (I7) it is diagnostic-only:
estimator_share is excluded, residual falls back to
``eps_upper_PLD - eps_lower[canary,w]`` and the row carries
``estimator_share_excluded=true``. When the canary track is unavailable
(tabular datasets have no pixel-canary design yet) the passive bounds take the
canary slots, ``threat_share_excluded=true`` is set, and the sum still
telescopes.

INVARIANTS HONOURED (plan §0)
-----------------------------
I1  conservative bounds come from ``estimate_empirical_lower_bound(...,
    threshold_selection="holdout")`` via ``pilot.build_result_row`` /
    the canary estimate path below; in-sample values are stored only under
    ``*_insample`` keys.
I2  eps_upper is recorded from BOTH accountants; the runner CRASHES if the
    target record's PLD backend is not google ``dp_accounting``.
I3  canary and passive scores never mix: separate audit functions, separate
    rows, one ``threat_model`` per row.
I4  passive scoring = symmetric Raw-LiRA with ``match_out_counts=True`` and
    ``sample_query_indices_disjoint`` (both reused from
    ``codex/run_raw_lira_pilot.py``; nothing reimplemented).
I5  canary design = ``matched_random_canaries`` (one generation pool).
I6  canonical seed triple for audited models; nothing here touches HPO seeds.
I7  every row carries ``quality_flag`` and the validity gate
    ``eps_lower_conservative <= eps_upper_pld``; violations are FLAGS.
I8  pooled budget below 1000+1000 observations => ``exploratory``.
I9  a censored zero is reported as censored, never as "0%".

COMPUTE SHAPE
-------------
Per (dataset, epsilon) cell:
    3 target models   (trained here if E2 checkpoints are absent -> running
                       this script also produces the E2 artifacts)
    K=32 shadows      trained ONCE per cell, reused for all 3 targets
    3 x 5 canary retrains (MNIST cells only)
Results are flushed to disk after every row: a Colab disconnect loses at most
one row. Re-running skips rows already present (resume-friendly).

OUTPUT
------
<results_root>/decomposition/
    rows.json / rows.csv       one row per (dataset, eps, seed, threat_model)
                               carrying BOTH estimators (wilson_holdout + gdp)
    shares.json / shares.csv   per (dataset, eps, seed) share decomposition +
                               per-cell mean/min/max aggregation rows
    diagnostics/               full wide result rows (build_result_row output)

HOW TO RUN
----------
    python codex/run_decomposition_sweep.py                       # full grid
    python codex/run_decomposition_sweep.py --dataset mnist       # one dataset
    python codex/run_decomposition_sweep.py --dataset mnist --epsilon 1.0
    SMOKE=1 python codex/run_decomposition_sweep.py --dataset mnist --epsilon 1.0
        # G3 gate: 64-query budget, K=4, 1 seed -> rows MUST come out
        # quality_flag="exploratory"; writes to results/decomposition_smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODEX = ROOT / "codex"
for path in (str(SRC), str(CODEX)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_raw_lira_pilot as pilot  # noqa: E402  (I4: reuse, don't reimplement)
from dp_audit_tightness.auditing.canary.one_run import (  # noqa: E402
    _infer_image_shape,
    run_one_run_canary_audit,
)
from dp_audit_tightness.config import (  # noqa: E402
    AuditRunConfig,
    CanaryAuditConfig,
    CanaryConfig,
    SaturationConfig,
    TrainExperimentConfig,
    config_to_dict,
    load_train_config,
)
from dp_audit_tightness.data.datasets import load_dataset_bundle  # noqa: E402
from dp_audit_tightness.models.io import load_model_for_inference  # noqa: E402
from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound  # noqa: E402
from dp_audit_tightness.privacy.gdp_estimation import estimate_epsilon_gdp  # noqa: E402
from dp_audit_tightness.training.dp_sgd import train_dp_sgd  # noqa: E402
from dp_audit_tightness.utils.logging_utils import configure_logging  # noqa: E402
from dp_audit_tightness.utils.results import (  # noqa: E402
    load_training_result,
    save_json,
    save_training_result,
)
from dp_audit_tightness.utils.seeds import set_global_seed  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"

# ---- E4 knobs (plan §E4) ----------------------------------------------------
K_SHADOWS = 4 if SMOKE else 32
# Per-seed budget; 5 seeds x 1024 = 5120 total disjoint queries (2560 members +
# 2560 non-members pooled), satisfying the plan's ">=5120 disjoint queries" and
# the I8 Wilson floor of 1000+1000.
QUERY_BUDGET = 64 if SMOKE else 1024
AUDIT_SEEDS = [401, 402] if SMOKE else [401, 402, 403, 404, 405]
NUM_CANARIES = 20 if SMOKE else 1000
CANONICAL_TRAINING_SEEDS = [123] if SMOKE else [123, 124, 125]  # I6
WILSON_FLOOR_PER_CLASS = 1000  # I8
SHARES_SUM_TOLERANCE = 0.02    # plan §E4 acceptance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4 decomposition sweep (see module docstring).")
    parser.add_argument("--configs-dir", default=str(ROOT / "results" / "hpo" / "best_configs"),
                        help="Directory of E1 winning configs (hpo_<dataset>_eps<eps>.yaml).")
    parser.add_argument("--dataset", default=None, help="Only run this dataset (e.g. mnist, diabetes).")
    parser.add_argument("--epsilon", type=float, default=None, help="Only run this target epsilon.")
    parser.add_argument("--results-root", default=str(ROOT / "results"),
                        help="Canonical results root (training artifacts + decomposition outputs).")
    parser.add_argument("--skip-canary", action="store_true",
                        help="Passive-only pass (threat_share_excluded on every cell).")
    return parser.parse_args()


# ---- E1 output discovery -----------------------------------------------------

def discover_cells(configs_dir: Path, dataset_filter: str | None, epsilon_filter: float | None):
    """Yield (dataset, target_epsilon, config_path) for every tuned config."""
    cells = []
    for path in sorted(configs_dir.glob("*.yaml")):
        config = load_train_config(path)
        dataset = config.dataset.name
        # Study naming: hpo_<dataset>_eps<target> (tune_hyperparams.py).
        stem = path.stem
        try:
            target_eps = float(stem.rsplit("eps", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse target epsilon from best-config name '{path.name}' "
                "(expected hpo_<dataset>_eps<value>.yaml)"
            ) from exc
        if dataset_filter and dataset != dataset_filter:
            continue
        if epsilon_filter is not None and abs(target_eps - epsilon_filter) > 1e-9:
            continue
        cells.append((dataset, target_eps, path))
    if not cells:
        raise FileNotFoundError(
            f"No matching best configs in {configs_dir}. Run E1 (tune_hyperparams.py) first; "
            "E4 audits the tuned models only."
        )
    return cells


# ---- E2: get-or-train the final audited models -------------------------------

def get_or_train_target(
    config: TrainExperimentConfig,
    training_seed: int,
    results_root: Path,
    logger,
):
    """Return the TrainingRunRecord for (config, seed), training it if absent.

    Reuses an existing E2 checkpoint when the training result JSON matches
    (experiment_name, split_seed, training_seed) and its artifact exists.
    """
    training_dir = results_root / "training"
    active_config = replace(
        config,
        run=replace(
            config.run,
            training_seed=training_seed,
            split_seeds=None,
            training_seeds=None,
            results_root=str(results_root),
            save_checkpoint=True,
        ),
    )
    if training_dir.exists():
        for result_path in sorted(training_dir.glob("*.json")):
            try:
                record = load_training_result(result_path)
            except Exception:
                continue
            if (
                record.experiment_name == active_config.experiment_name
                and record.training_seed == training_seed
                and record.split_seed == active_config.run.split_seed
                and record.model_artifact_path
            ):
                artifact = Path(record.model_artifact_path)
                if not artifact.exists():
                    # Records store absolute paths; when E2 results were made in a
                    # different session (/content vs /kaggle/working) re-anchor on
                    # this run's checkpoints directory before giving up.
                    relocated = training_dir / "checkpoints" / artifact.name
                    if relocated.exists():
                        record.model_artifact_path = str(relocated)
                        artifact = relocated
                if artifact.exists():
                    logger.info("Reusing E2 model %s (seed=%s)", record.training_run_id, training_seed)
                    return record
    logger.info("Training target %s seed=%s (E2 artifact)", active_config.experiment_name, training_seed)
    set_global_seed(training_seed)
    outcome = train_dp_sgd(config=active_config, logger=logger)
    config_snapshot_path = training_dir / "configs" / f"{outcome.record.training_run_id}_config.json"
    outcome.record.config_snapshot_path = str(config_snapshot_path)
    save_json(config_snapshot_path, config_to_dict(active_config))
    save_training_result(outcome.record, results_root=str(results_root))
    return outcome.record


def enforce_accounting_invariant(record, cell_name: str) -> None:
    """I2: PLD via google dp_accounting or CRASH — no silent fallback."""
    if record.epsilon_upper_pld is None or record.pld_accounting_backend != "google_dp_accounting":
        raise RuntimeError(
            f"I2 violation in {cell_name}: epsilon_upper_pld={record.epsilon_upper_pld}, "
            f"backend={record.pld_accounting_backend!r}. Install dp_accounting and retrain; "
            "the analytical fallback is forbidden for decomposition rows."
        )


# ---- Quality flags (I7 / I8) --------------------------------------------------

def quality_flag_for(row: dict) -> str:
    if row.get("validity") == "invalid_exceeds_upper_bound":
        return "invalid_exceeds_upper_bound"
    if (
        row.get("num_member_samples", 0) < WILSON_FLOOR_PER_CLASS
        or row.get("num_nonmember_samples", 0) < WILSON_FLOOR_PER_CLASS
    ):
        return "exploratory"
    if row.get("censored") == "conservative_zero_below_floor_or_no_signal":
        return "conservative_zero_below_floor_or_no_signal"
    return "ok"


# ---- Passive track (I4) --------------------------------------------------------

def passive_audit_cell(cell, config, targets, bundle, results_root, logger) -> list[dict]:
    """Raw-LiRA passive audit of every target in the cell; shadows trained once."""
    import torch

    dataset, target_eps = cell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Point the reused pilot helpers at our knobs/output dir.
    pilot.QUERY_BUDGET = QUERY_BUDGET
    pilot.AUDIT_SEEDS = list(AUDIT_SEEDS)
    pilot.RESULTS_DIR = results_root / "decomposition" / "shadow_training"

    (query_train_indices, query_eval_indices,
     per_seed_train, per_seed_eval) = pilot.sample_query_indices_disjoint(
        train_size=len(bundle.train_dataset),
        eval_size=len(bundle.eval_dataset),
    )

    # Shadows depend only on (pipeline, split), not the target seed -> once per cell.
    shadow_case = {"config": targets[0]["config"], "bundle": bundle}
    started = time.time()
    shadow_train_losses, shadow_eval_losses, shadow_member_sets = pilot.train_shadow_losses(
        training_case=shadow_case,
        dataset_name=dataset,
        query_train_indices=query_train_indices,
        query_eval_indices=query_eval_indices,
        k_max=K_SHADOWS,
        logger=logger,
    )
    logger.info("Cell %s eps=%s: %s shadows in %.1fs", dataset, target_eps, K_SHADOWS, time.time() - started)

    rows = []
    for target in targets:
        record = target["record"]
        target_model = load_model_for_inference(
            target["config"].model, record.model_artifact_path, device=device
        )
        target_train_losses = pilot.compute_loss_for_indices(
            target_model, bundle.train_dataset, query_train_indices, device=device
        )
        target_eval_losses = pilot.compute_loss_for_indices(
            target_model, bundle.eval_dataset, query_eval_indices, device=device
        )
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
        row = pilot.build_result_row(
            dataset_name=dataset,
            attack_name="passive_raw_lira",
            k=K_SHADOWS,
            training_case={"config": target["config"], "record": record},
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            score_direction="higher",
        )
        row = pilot.apply_quality_flags(row)
        row.update(
            threat_model="passive",
            target_epsilon=target_eps,
            training_seed=record.training_seed,
            split_seed=record.split_seed,
            training_run_id=record.training_run_id,
            quality_flag=quality_flag_for(row),
        )
        rows.append(row)
        logger.info(
            "passive %s eps=%s seed=%s eps_lower=%s flag=%s",
            dataset, target_eps, record.training_seed,
            row["epsilon_lower_conservative"], row["quality_flag"],
        )
    return rows


# ---- Canary track (I5) ---------------------------------------------------------

def canary_supported(config: TrainExperimentConfig) -> bool:
    """Pixel-patch canaries only fit models whose input matches the image shape."""
    shape = _infer_image_shape(config.dataset.name)
    return int(math.prod(shape)) == int(config.model.input_dim)


def canary_audit_target(cell, config, target, results_root, logger) -> dict:
    dataset, target_eps = cell
    record = target["record"]
    delta = config.privacy.delta
    train_size = 0
    canary_config = CanaryAuditConfig(
        training_result_path="__in_memory__",
        audit_regime="evaluator_controlled_canary_stress_test",
        auditor_variant="matched_random_canaries",
        auditor_strength_rank=2,
        audit_mode="repeated_run",
        delta=delta,
        repeated_seeds=list(AUDIT_SEEDS),
        canary=CanaryConfig(
            strategy="matched_random_canaries",  # I5: one generation pool
            num_canaries=NUM_CANARIES,
            insertion_rate=NUM_CANARIES / 54000.0,  # unused by the matched path
            optimize_steps=0,
        ),
        saturation=SaturationConfig(),
        run=AuditRunConfig(
            results_root=str(results_root / "decomposition" / "canary_training"),
            save_debug_artifacts=False,
        ),
    )
    member_scores: list[float] = []
    nonmember_scores: list[float] = []
    for audit_seed in AUDIT_SEEDS:
        obs = run_one_run_canary_audit(record, config, canary_config, audit_seed, logger=logger)
        member_scores.extend(obs.member_scores)
        nonmember_scores.extend(obs.nonmember_scores)
        logger.info(
            "canary %s eps=%s seed=%s audit_seed=%s gap=%.4f",
            dataset, target_eps, record.training_seed, audit_seed,
            obs.raw_statistics["score_mean_gap"],
        )

    common = dict(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=delta,
        score_direction="higher",
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
    )
    conservative = estimate_empirical_lower_bound(threshold_selection="holdout", **common)  # I1
    insample = estimate_empirical_lower_bound(threshold_selection="in_sample", **common)
    gdp = estimate_epsilon_gdp(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        delta=delta,
        score_direction="higher",
        n_bootstrap=2000,
    )
    eps_upper_pld = record.epsilon_upper_pld
    row = {
        "dataset": dataset,
        "attack": "canary_matched_logit_margin",
        "status": "ok",
        "k_shadows": None,
        "score_direction": "higher",
        "query_budget_per_seed": 2 * NUM_CANARIES,
        "num_audit_seeds": len(AUDIT_SEEDS),
        "audit_seeds": json.dumps(AUDIT_SEEDS),
        "epsilon_upper_rdp": record.epsilon_upper_theory,
        "epsilon_upper_tighter": eps_upper_pld,
        "tighter_upper_backend": record.pld_accounting_backend,
        "epsilon_lower_conservative": conservative.epsilon_lower_empirical,
        "epsilon_lower_point": conservative.epsilon_lower_empirical_point_estimate,
        "threshold_selection": conservative.threshold_selection,
        "holdout_selection_half_sizes": json.dumps(conservative.selection_half_sizes),
        "holdout_estimation_half_sizes": json.dumps(conservative.estimation_half_sizes),
        "epsilon_lower_conservative_insample": insample.epsilon_lower_empirical,
        "epsilon_lower_point_insample": insample.epsilon_lower_empirical_point_estimate,
        "tightness_ratio_tighter": (
            (conservative.epsilon_lower_empirical / eps_upper_pld) if eps_upper_pld else None
        ),
        "epsilon_lower_gdp": gdp.epsilon_lower,
        "epsilon_lower_gdp_point": gdp.epsilon_lower_point,
        "gdp_valid_lower_bound": (
            eps_upper_pld is not None and gdp.epsilon_lower <= eps_upper_pld
        ),
        "gdp_auc": gdp.auc,
        "selected_tpr": conservative.selected_tpr,
        "selected_fpr": conservative.selected_fpr,
        "warning": conservative.warning_message,
        "num_member_samples": conservative.num_member_samples,
        "num_nonmember_samples": conservative.num_nonmember_samples,
        "member_score_mean": statistics.fmean(member_scores) if member_scores else None,
        "nonmember_score_mean": statistics.fmean(nonmember_scores) if nonmember_scores else None,
    }
    row = pilot.apply_quality_flags(row)
    row.update(
        threat_model="canary",
        target_epsilon=target_eps,
        training_seed=record.training_seed,
        split_seed=record.split_seed,
        training_run_id=record.training_run_id,
        quality_flag=quality_flag_for(row),
    )
    return row


# ---- Share decomposition (§0 formulas, telescoping variant) --------------------

def compute_shares(passive_row: dict, canary_row: dict | None) -> dict:
    eps_rdp = passive_row["epsilon_upper_rdp"]
    eps_pld = passive_row["epsilon_upper_tighter"]
    pass_w = passive_row["epsilon_lower_conservative"] or 0.0
    pass_gdp = passive_row["epsilon_lower_gdp"] or 0.0
    pass_gdp_valid = bool(passive_row.get("gdp_valid_lower_bound"))

    threat_share_excluded = canary_row is None
    if canary_row is not None:
        can_w = canary_row["epsilon_lower_conservative"] or 0.0
        can_gdp = canary_row["epsilon_lower_gdp"] or 0.0
        can_gdp_valid = bool(canary_row.get("gdp_valid_lower_bound"))
    else:
        # Tabular cell: passive bounds take the canary slots so the sum still
        # telescopes; threat_share is excluded (see module docstring).
        can_w, can_gdp, can_gdp_valid = pass_w, pass_gdp, pass_gdp_valid

    denominator = eps_rdp - pass_w  # Delta_eps(RDP, passive, wilson_holdout)
    estimator_share_excluded = not can_gdp_valid
    best_canary = can_w if estimator_share_excluded else can_gdp

    if denominator <= 0:
        shares = dict.fromkeys(
            ("accounting_share", "threat_share", "estimator_share", "residual_share"), None
        )
        shares_sum = None
    else:
        accounting = (eps_rdp - eps_pld) / denominator
        threat = 0.0 if threat_share_excluded else (can_w - pass_w) / denominator
        estimator = 0.0 if estimator_share_excluded else (can_gdp - can_w) / denominator
        residual = (eps_pld - best_canary) / denominator
        shares = {
            "accounting_share": round(accounting, 6),
            "threat_share": round(threat, 6),
            "estimator_share": round(estimator, 6),
            "residual_share": round(residual, 6),
        }
        shares_sum = round(accounting + threat + estimator + residual, 6)

    return {
        "dataset": passive_row["dataset"],
        "target_epsilon": passive_row["target_epsilon"],
        "training_seed": passive_row["training_seed"],
        "row_kind": "per_seed",
        "epsilon_upper_rdp": eps_rdp,
        "epsilon_upper_pld": eps_pld,
        "epsilon_lower_passive_wilson": pass_w,
        "epsilon_lower_passive_gdp": pass_gdp,
        "epsilon_lower_canary_wilson": None if canary_row is None else can_w,
        "epsilon_lower_canary_gdp": None if canary_row is None else can_gdp,
        "total_gap_rdp_passive_wilson": round(denominator, 6),
        "residual_nats": round(eps_pld - best_canary, 6),
        **shares,
        "shares_sum": shares_sum,
        "shares_sum_ok": (
            None if shares_sum is None else abs(shares_sum - 1.0) <= SHARES_SUM_TOLERANCE
        ),
        "estimator_share_excluded": estimator_share_excluded,
        "threat_share_excluded": threat_share_excluded,
        "passive_quality_flag": passive_row["quality_flag"],
        "canary_quality_flag": None if canary_row is None else canary_row["quality_flag"],
    }


def aggregate_shares(per_seed: list[dict]) -> list[dict]:
    """Mean over seeds with min/max as error bars (plan §E4), per cell."""
    cells: dict[tuple, list[dict]] = {}
    for row in per_seed:
        cells.setdefault((row["dataset"], row["target_epsilon"]), []).append(row)
    aggregated = []
    share_keys = ("accounting_share", "threat_share", "estimator_share", "residual_share")
    for (dataset, target_eps), rows in sorted(cells.items()):
        out = {
            "dataset": dataset,
            "target_epsilon": target_eps,
            "training_seed": None,
            "row_kind": "cell_aggregate",
            "num_seeds": len(rows),
        }
        for key in share_keys:
            values = [r[key] for r in rows if r[key] is not None]
            out[f"{key}_mean"] = round(statistics.fmean(values), 6) if values else None
            out[f"{key}_min"] = round(min(values), 6) if values else None
            out[f"{key}_max"] = round(max(values), 6) if values else None
        out["estimator_share_excluded_any"] = any(r["estimator_share_excluded"] for r in rows)
        out["threat_share_excluded_any"] = any(r["threat_share_excluded"] for r in rows)
        out["shares_sum_ok_all"] = all(bool(r["shares_sum_ok"]) for r in rows)
        aggregated.append(out)
    return aggregated


# ---- Long-format rows + persistence --------------------------------------------

def tidy_rows(wide_row: dict) -> list[dict]:
    """Explode one wide audit row into per-estimator rows (dataset, eps, seed, theta, E)."""
    base = {
        "dataset": wide_row["dataset"],
        "target_epsilon": wide_row["target_epsilon"],
        "training_seed": wide_row["training_seed"],
        "split_seed": wide_row["split_seed"],
        "threat_model": wide_row["threat_model"],
        "attack": wide_row["attack"],
        "training_run_id": wide_row["training_run_id"],
        "epsilon_upper_rdp": wide_row["epsilon_upper_rdp"],
        "epsilon_upper_pld": wide_row["epsilon_upper_tighter"],
        "pld_accounting_backend": wide_row["tighter_upper_backend"],
        "num_member_samples": wide_row["num_member_samples"],
        "num_nonmember_samples": wide_row["num_nonmember_samples"],
        "quality_flag": wide_row["quality_flag"],
        "validity": wide_row["validity"],
        "censored": wide_row["censored"],
    }
    wilson = dict(
        base,
        estimator="wilson_holdout",
        epsilon_lower=wide_row["epsilon_lower_conservative"],
        epsilon_lower_point=wide_row["epsilon_lower_point"],
        tightness_ratio_rdp=(
            (wide_row["epsilon_lower_conservative"] or 0.0) / wide_row["epsilon_upper_rdp"]
            if wide_row["epsilon_upper_rdp"] else None
        ),
        tightness_ratio_pld=(
            (wide_row["epsilon_lower_conservative"] or 0.0) / wide_row["epsilon_upper_tighter"]
            if wide_row["epsilon_upper_tighter"] else None
        ),
        estimator_valid=True,
    )
    gdp_valid = bool(wide_row.get("gdp_valid_lower_bound"))
    gdp = dict(
        base,
        estimator="gdp",
        epsilon_lower=wide_row["epsilon_lower_gdp"],
        epsilon_lower_point=wide_row["epsilon_lower_gdp_point"],
        tightness_ratio_rdp=(
            (wide_row["epsilon_lower_gdp"] or 0.0) / wide_row["epsilon_upper_rdp"]
            if gdp_valid and wide_row["epsilon_upper_rdp"] else None
        ),
        tightness_ratio_pld=(
            (wide_row["epsilon_lower_gdp"] or 0.0) / wide_row["epsilon_upper_tighter"]
            if gdp_valid and wide_row["epsilon_upper_tighter"] else None
        ),
        estimator_valid=gdp_valid,
    )
    return [wilson, gdp]


def write_table(path_base: Path, rows: list[dict]) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    with path_base.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path_base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict) -> str:
    return f"{row['dataset']}|{row['target_epsilon']}|{row['training_seed']}|{row['threat_model']}"


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    results_root = Path(args.results_root)
    if SMOKE:
        results_root = results_root.parent / (results_root.name + "_decomposition_smoke")
        logger.warning("SMOKE mode: results_root=%s, budgets are sub-floor by design (G3).", results_root)
    out_dir = results_root / "decomposition"
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    cells = discover_cells(Path(args.configs_dir), args.dataset, args.epsilon)
    logger.info("E4 grid: %s cells", len(cells))

    # Resume support: reload previously finished wide rows.
    wide_path = out_dir / "wide_rows.json"
    wide_rows: list[dict] = []
    if wide_path.exists():
        wide_rows = json.loads(wide_path.read_text(encoding="utf-8"))
        logger.info("Resuming: %s wide rows already present", len(wide_rows))
    done = {row_key(r) for r in wide_rows}

    def flush() -> None:
        save_json(wide_path, wide_rows)  # type: ignore[arg-type]
        tidy: list[dict] = []
        for wide in wide_rows:
            tidy.extend(tidy_rows(wide))
        write_table(out_dir / "rows", tidy)
        per_seed = []
        by_target: dict[tuple, dict[str, dict]] = {}
        for wide in wide_rows:
            key = (wide["dataset"], wide["target_epsilon"], wide["training_seed"])
            by_target.setdefault(key, {})[wide["threat_model"]] = wide
        for key in sorted(by_target):
            tracks = by_target[key]
            if "passive" not in tracks:
                continue  # shares are anchored on the passive gap
            per_seed.append(compute_shares(tracks["passive"], tracks.get("canary")))
        write_table(out_dir / "shares", per_seed + aggregate_shares(per_seed))

    for dataset, target_eps, config_path in cells:
        cell = (dataset, target_eps)
        cell_name = f"{dataset}_eps{target_eps}"
        config = load_train_config(config_path)

        # E2 gate: recorded eps_PLD must hit the target (plan §E2 acceptance).
        targets = []
        for seed in (config.run.training_seeds or CANONICAL_TRAINING_SEEDS):
            record = get_or_train_target(config, seed, results_root, logger)
            enforce_accounting_invariant(record, cell_name)  # I2
            if abs(record.epsilon_upper_pld - target_eps) > 1e-3:
                logger.warning(
                    "%s seed=%s: eps_PLD=%.6f misses target %s by >1e-3 — flagging, not hiding.",
                    cell_name, seed, record.epsilon_upper_pld, target_eps,
                )
            targets.append({"config": config, "record": record})

        bundle = load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)

        pending_passive = [t for t in targets
                           if f"{dataset}|{target_eps}|{t['record'].training_seed}|passive" not in done]
        if pending_passive:
            for row in passive_audit_cell(cell, config, pending_passive, bundle, results_root, logger):
                wide_rows.append(row)
                save_json(diagnostics_dir / f"{cell_name}_seed{row['training_seed']}_passive.json", row)
                flush()
        else:
            logger.info("%s: passive rows already complete", cell_name)

        if args.skip_canary:
            logger.info("%s: canary skipped by flag", cell_name)
        elif not canary_supported(config):
            logger.warning(
                "%s: matched pixel canaries do not fit input_dim=%s — no tabular canary "
                "design exists yet (open research direction); cell will carry "
                "threat_share_excluded=true.",
                cell_name, config.model.input_dim,
            )
        else:
            for target in targets:
                seed = target["record"].training_seed
                if f"{dataset}|{target_eps}|{seed}|canary" in done:
                    logger.info("%s seed=%s: canary row already complete", cell_name, seed)
                    continue
                row = canary_audit_target(cell, config, target, results_root, logger)
                wide_rows.append(row)
                save_json(diagnostics_dir / f"{cell_name}_seed{seed}_canary.json", row)
                flush()

        # G2 gate report after every cell.
        flush()
        shares_rows = json.loads((out_dir / "shares.json").read_text(encoding="utf-8"))
        bad = [
            r for r in shares_rows
            if r.get("row_kind") == "per_seed"
            and r["dataset"] == dataset and r["target_epsilon"] == target_eps
            and r["shares_sum_ok"] is False
        ]
        if bad:
            logger.error(
                "G2 FAILURE in %s: shares do not telescope to 1 +/- %s in %s row(s) — "
                "inspect before launching more cells.",
                cell_name, SHARES_SUM_TOLERANCE, len(bad),
            )
        else:
            logger.info("G2 ok for %s (shares telescope).", cell_name)

    print(f"\nWrote {out_dir / 'rows.csv'}")
    print(f"Wrote {out_dir / 'shares.csv'}")
    print(f"Wrote {wide_path} (+ per-row diagnostics in {diagnostics_dir})")


if __name__ == "__main__":
    main()
