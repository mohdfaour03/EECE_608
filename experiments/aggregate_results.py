from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
