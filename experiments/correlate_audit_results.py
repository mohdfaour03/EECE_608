"""Correlate canary and passive audit results into unified comparison JSONs.

For each training run, finds matching canary and passive audit results and
produces a single JSON with both audit outcomes side-by-side plus a gap
decomposition summary.

Usage
-----
    python experiments/correlate_audit_results.py \
        --results-root results \
        --output-dir results/comparisons
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dp_audit_tightness.utils.results import (
    CANARY_AUDIT_REGIME,
    PASSIVE_AUDIT_REGIME,
)


def correlate_results(results_root: str | Path, output_dir: str | Path) -> list[Path]:
    """Find all training runs and correlate audit results for each."""
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all training run records.
    training_dir = results_root / "training"
    training_records = _load_json_files(training_dir, pattern="train_*.json")

    # Load all audit records from both regimes.
    canary_records = _load_audit_records(results_root, CANARY_AUDIT_REGIME)
    passive_records = _load_audit_records(results_root, PASSIVE_AUDIT_REGIME)

    # Index audit records by training_run_id.
    canary_by_train = _index_by_training_run(canary_records)
    passive_by_train = _index_by_training_run(passive_records)

    written: list[Path] = []
    for train_record in training_records:
        training_run_id = train_record.get("training_run_id", "")
        if not training_run_id:
            continue

        canary_audits = canary_by_train.get(training_run_id, [])
        passive_audits = passive_by_train.get(training_run_id, [])

        if not canary_audits and not passive_audits:
            continue

        comparison = _build_comparison(train_record, canary_audits, passive_audits)
        out_path = output_dir / f"comparison_{training_run_id}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2)
        written.append(out_path)

    return written


def _build_comparison(
    train_record: dict[str, Any],
    canary_audits: list[dict[str, Any]],
    passive_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build unified comparison JSON for one training run."""
    eps_upper = train_record.get("epsilon_upper_theory", 0.0)

    eps_upper_pld = train_record.get("epsilon_upper_pld")

    comparison: dict[str, Any] = {
        "training_run_id": train_record.get("training_run_id"),
        "dataset": train_record.get("dataset"),
        "model": train_record.get("model_name"),
        "dp_params": {
            "noise_multiplier": train_record.get("noise_multiplier"),
            "max_grad_norm": train_record.get("clipping_norm"),
            "batch_size": train_record.get("batch_size"),
            "epochs": train_record.get("epochs"),
            "delta": train_record.get("delta"),
            "sampling_rate": train_record.get("sampling_rate"),
        },
        "epsilon_upper_theory": eps_upper,
        "epsilon_upper_pld": eps_upper_pld,
        "pld_accounting_backend": train_record.get("pld_accounting_backend"),
        "utility": train_record.get("utility_metrics", {}),
    }

    # Best canary result (highest epsilon_lower_empirical).
    if canary_audits:
        best_canary = max(canary_audits, key=lambda r: r.get("epsilon_lower_empirical", 0.0))
        eps_canary = best_canary.get("epsilon_lower_empirical", 0.0)
        comparison["canary_audit"] = {
            "auditor_variant": best_canary.get("auditor_variant"),
            "epsilon_lower_empirical": eps_canary,
            "empirical_ci_lower": best_canary.get("empirical_ci_lower"),
            "empirical_ci_upper": best_canary.get("empirical_ci_upper"),
            "privacy_loss_gap": best_canary.get("privacy_loss_gap"),
            "tightness_ratio": best_canary.get("tightness_ratio"),
            "raw_stats": best_canary.get("audit_metadata", {}).get("raw_statistics", {}),
        }
    else:
        eps_canary = None
        comparison["canary_audit"] = None

    # Best passive result.
    if passive_audits:
        best_passive = max(passive_audits, key=lambda r: r.get("epsilon_lower_empirical", 0.0))
        eps_passive = best_passive.get("epsilon_lower_empirical", 0.0)
        comparison["passive_audit"] = {
            "auditor_variant": best_passive.get("auditor_variant"),
            "epsilon_lower_empirical": eps_passive,
            "empirical_ci_lower": best_passive.get("empirical_ci_lower"),
            "empirical_ci_upper": best_passive.get("empirical_ci_upper"),
            "privacy_loss_gap": best_passive.get("privacy_loss_gap"),
            "tightness_ratio": best_passive.get("tightness_ratio"),
            "raw_stats": best_passive.get("audit_metadata", {}).get("raw_statistics", {}),
        }
    else:
        eps_passive = None
        comparison["passive_audit"] = None

    # Gap decomposition (full, using both RDP and PLD upper bounds).
    comparison["comparison"] = _compute_gaps(eps_upper, eps_upper_pld, eps_canary, eps_passive)

    return comparison


def _compute_gaps(
    eps_upper_rdp: float,
    eps_upper_pld: float | None,
    eps_canary: float | None,
    eps_passive: float | None,
) -> dict[str, Any]:
    """Compute the full gap decomposition.

    total_gap = accounting_gap + threat_model_gap + residual_gap

    Where:
        accounting_gap   = eps_upper_rdp - eps_upper_pld
        threat_model_gap = eps_lower_canary - eps_lower_passive
        residual_gap     = tightest_upper - eps_lower_canary
    """
    result: dict[str, Any] = {}

    # Accounting gap: how much looser is RDP vs PLD?
    if eps_upper_pld is not None:
        result["accounting_gap"] = round(eps_upper_rdp - eps_upper_pld, 6)
    else:
        result["accounting_gap"] = None

    # Tightest available upper bound
    tightest_upper = eps_upper_pld if eps_upper_pld is not None else eps_upper_rdp

    # Canary gap and ratio (against tightest upper)
    if eps_canary is not None:
        result["gap_canary"] = round(tightest_upper - eps_canary, 6)
        result["ratio_canary"] = round(eps_canary / tightest_upper, 6) if tightest_upper > 0 else None
    else:
        result["gap_canary"] = None
        result["ratio_canary"] = None

    # Passive gap and ratio (against tightest upper)
    if eps_passive is not None:
        result["gap_passive"] = round(tightest_upper - eps_passive, 6)
        result["ratio_passive"] = round(eps_passive / tightest_upper, 6) if tightest_upper > 0 else None
    else:
        result["gap_passive"] = None
        result["ratio_passive"] = None

    # Threat model gap: how much more does canary recover vs passive?
    if eps_canary is not None and eps_passive is not None:
        result["threat_model_gap"] = round(eps_canary - eps_passive, 6)
    else:
        result["threat_model_gap"] = None

    # Residual gap: tightest_upper - best canary lower bound
    if eps_canary is not None:
        result["residual_gap"] = round(tightest_upper - eps_canary, 6)
    else:
        result["residual_gap"] = None

    # Total gap (RDP upper - best lower)
    best_lower = max(v for v in (eps_canary, eps_passive) if v is not None) if any(v is not None for v in (eps_canary, eps_passive)) else 0.0
    result["total_gap"] = round(eps_upper_rdp - best_lower, 6)

    return result


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_json_files(directory: Path, pattern: str = "*.json") -> list[dict[str, Any]]:
    records = []
    if not directory.exists():
        return records
    for path in sorted(directory.glob(pattern)):
        try:
            with path.open("r", encoding="utf-8") as fh:
                records.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def _load_audit_records(results_root: Path, audit_regime: str) -> list[dict[str, Any]]:
    """Load audit JSON files from the regime-specific subdirectory."""
    audit_dir = results_root / "audits" / audit_regime
    return _load_json_files(audit_dir)


def _index_by_training_run(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        tid = record.get("training_run_id", "")
        if tid:
            index.setdefault(tid, []).append(record)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate audit results into comparison JSONs.")
    parser.add_argument("--results-root", default="results", help="Root results directory.")
    parser.add_argument("--output-dir", default="results/comparisons", help="Output directory.")
    args = parser.parse_args()

    written = correlate_results(args.results_root, args.output_dir)
    for path in written:
        print(f"  wrote {path}")
    print(f"\nCorrelated {len(written)} comparison(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
