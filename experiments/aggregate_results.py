from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.reporting.summary import aggregate_audit_records, build_audit_run_summary_rows
from dp_audit_tightness.utils.results import (
    discover_audit_result_paths,
    discover_training_result_paths,
    load_audit_result,
    load_training_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate training and audit result artifacts.")
    parser.add_argument("--results-root", default="results", help="Root results directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    training_records = {
        record.training_run_id: record
        for record in (
            load_training_result(path)
            for path in discover_training_result_paths(results_root)
        )
    }
    audit_records = [
        load_audit_result(path)
        for path in discover_audit_result_paths(results_root)
    ]

    run_summary_rows = build_audit_run_summary_rows(training_records, audit_records)
    group_summaries = aggregate_audit_records(audit_records)

    summaries_dir = results_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    run_summary_csv = summaries_dir / "audit_run_summary.csv"
    run_summary_json = summaries_dir / "audit_run_summary.json"
    group_summary_csv = summaries_dir / "audit_group_summary.csv"
    group_summary_json = summaries_dir / "audit_group_summary.json"

    _write_csv(run_summary_csv, run_summary_rows)
    _write_json(run_summary_json, run_summary_rows)
    _write_csv(group_summary_csv, [summary.to_flat_dict() for summary in group_summaries])
    _write_json(group_summary_json, [summary.to_dict() for summary in group_summaries])

    print(f"Wrote per-run summary to {run_summary_csv}")
    print(f"Wrote grouped summary to {group_summary_csv}")

    aggregate_decomposition(results_root, summaries_dir)


def aggregate_decomposition(results_root: Path, summaries_dir: Path) -> None:
    """E7: fold E4 decomposition outputs into the paper's Table 1 (plan §E7).

    Reads ONLY the schema'd JSON written by ``codex/run_decomposition_sweep.py``
    (results/decomposition/rows.json + shares.json). Skips silently when E4 has
    not been run yet so the legacy aggregation keeps working unchanged.
    """
    decomposition_dir = results_root / "decomposition"
    rows_path = decomposition_dir / "rows.json"
    shares_path = decomposition_dir / "shares.json"
    if not rows_path.exists() or not shares_path.exists():
        print("No decomposition results found (run codex/run_decomposition_sweep.py first) -- skipping Table 1.")
        return

    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    shares = json.loads(shares_path.read_text(encoding="utf-8"))
    aggregates = {
        (row["dataset"], row["target_epsilon"]): row
        for row in shares
        if row.get("row_kind") == "cell_aggregate"
    }

    cells: dict[tuple, dict] = {}
    for row in rows:
        key = (row["dataset"], row["target_epsilon"])
        cell = cells.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "target_epsilon": row["target_epsilon"],
                "epsilon_upper_rdp": [],
                "epsilon_upper_pld": [],
                "bounds": {},
                "quality_flags": [],
            },
        )
        cell["epsilon_upper_rdp"].append(row["epsilon_upper_rdp"])
        cell["epsilon_upper_pld"].append(row["epsilon_upper_pld"])
        cell["quality_flags"].append(row["quality_flag"])
        # Censored != zero (I9): only non-censored, valid cells contribute the
        # mean bound; censored cells are counted separately.
        if row.get("quality_flag") == "ok" and row.get("epsilon_lower") is not None:
            cell["bounds"].setdefault((row["threat_model"], row["estimator"]), []).append(
                row["epsilon_lower"]
            )

    table_rows = []
    for key in sorted(cells):
        cell = cells[key]
        eps_rdp = statistics.fmean(cell["epsilon_upper_rdp"])
        eps_pld = statistics.fmean(cell["epsilon_upper_pld"])
        out = {
            "dataset": cell["dataset"],
            "target_epsilon": cell["target_epsilon"],
            "epsilon_upper_rdp": round(eps_rdp, 6),
            "epsilon_upper_pld": round(eps_pld, 6),
        }
        for theta in ("passive", "canary"):
            for estimator in ("wilson_holdout", "gdp"):
                values = cell["bounds"].get((theta, estimator))
                column = f"eps_lower_{theta}_{estimator}"
                out[column] = round(statistics.fmean(values), 6) if values else None
                out[f"{column}_n"] = len(values) if values else 0
                out[f"tightness_{theta}_{estimator}_pld"] = (
                    round(statistics.fmean(values) / eps_pld, 6) if values and eps_pld else None
                )
        aggregate = aggregates.get(key, {})
        for share in ("accounting_share", "threat_share", "estimator_share", "residual_share"):
            out[f"{share}_mean"] = aggregate.get(f"{share}_mean")
            out[f"{share}_min"] = aggregate.get(f"{share}_min")
            out[f"{share}_max"] = aggregate.get(f"{share}_max")
        out["estimator_share_excluded_any"] = aggregate.get("estimator_share_excluded_any")
        out["threat_share_excluded_any"] = aggregate.get("threat_share_excluded_any")
        out["shares_sum_ok_all"] = aggregate.get("shares_sum_ok_all")
        flags = cell["quality_flags"]
        out["num_rows"] = len(flags)
        out["num_censored"] = flags.count("conservative_zero_below_floor_or_no_signal")
        out["num_invalid"] = flags.count("invalid_exceeds_upper_bound")
        out["num_exploratory"] = flags.count("exploratory")
        table_rows.append(out)

    table_csv = summaries_dir / "decomposition_table1.csv"
    table_json = summaries_dir / "decomposition_table1.json"
    _write_csv(table_csv, table_rows)
    _write_json(table_json, table_rows)
    print(f"Wrote decomposition Table 1 to {table_csv}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
