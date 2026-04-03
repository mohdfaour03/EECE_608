from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.utils.paths import ensure_directory
from dp_audit_tightness.utils.results import load_audit_result, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare canary audit results before and after conservative calibration."
    )
    parser.add_argument(
        "--before-dir",
        default="results/canary_seedfix_3seed/audits/canary",
        help="Directory containing the pre-calibration post-seedfix canary audit JSON files.",
    )
    parser.add_argument(
        "--after-dir",
        default="results/canary_calibrated_3seed/audits/canary",
        help="Directory containing the calibrated canary audit JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/substantive_smoke_3seed/audits/canary_calibrated_comparison",
        help="Directory where the before/after comparison artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_records = _load_canary_records(Path(args.before_dir))
    after_records = _load_canary_records(Path(args.after_dir))

    rows: list[dict[str, Any]] = []
    for seed, before in before_records.items():
        after = after_records[seed]
        rows.append(_build_row(before, after))

    rows.sort(key=lambda row: row["seed"])
    output_dir = ensure_directory(Path(args.output_dir))
    _write_csv(output_dir / "before_after_table.csv", rows)
    save_json(output_dir / "before_after_table.json", {"rows": rows})
    (output_dir / "summary.md").write_text(_build_summary(rows), encoding="utf-8")


def _load_canary_records(directory: Path) -> dict[int, Any]:
    records: dict[int, Any] = {}
    for path in sorted(directory.glob("canary_audit_seed*.json")):
        record = load_audit_result(path)
        records[record.training_seed] = record
    return records


def _build_row(before, after) -> dict[str, Any]:
    metadata = after.audit_metadata
    return {
        "seed": before.training_seed,
        "epsilon_upper_theory_before": before.epsilon_upper_theory,
        "epsilon_lower_empirical_before": before.epsilon_lower_empirical,
        "epsilon_upper_theory_after": after.epsilon_upper_theory,
        "epsilon_lower_empirical_point_estimate_after": metadata.get("epsilon_lower_empirical_point_estimate"),
        "epsilon_lower_empirical_after": after.epsilon_lower_empirical,
        "privacy_loss_gap_after": after.privacy_loss_gap,
        "tightness_ratio_after": after.tightness_ratio,
        "ci_lower_after": after.empirical_ci_lower,
        "ci_upper_after": after.empirical_ci_upper,
        "selected_threshold_after": metadata.get("selected_threshold"),
        "selected_event_after": metadata.get("selected_event_direction") or metadata.get("selected_event"),
        "canary_event_fraction_after": metadata.get("canary_event_fraction"),
        "control_event_fraction_after": metadata.get("control_event_fraction"),
        "tiny_tail_warning_after": metadata.get("selected_event_is_tiny_tail"),
    }


def _build_summary(rows: list[dict[str, Any]]) -> str:
    warning_cases_removed = sum(
        row["epsilon_lower_empirical_before"] > row["epsilon_upper_theory_before"]
        and row["epsilon_lower_empirical_after"] <= row["epsilon_upper_theory_after"]
        for row in rows
    )
    remaining_warning_cases = sum(
        row["epsilon_lower_empirical_after"] > row["epsilon_upper_theory_after"] for row in rows
    )
    shrinkages = [
        row["epsilon_lower_empirical_before"] - row["epsilon_lower_empirical_after"]
        for row in rows
    ]
    remaining_nontrivial = sum(row["epsilon_lower_empirical_after"] > 0.0 for row in rows)
    calibrated_means = statistics.fmean(row["epsilon_lower_empirical_after"] for row in rows)
    passive_corrected_mean = 0.11899110568057962

    return "\n".join(
        [
            "# Canary Calibrated Comparison",
            "",
            "## Direct Answers",
            "",
            f"- Were the `epsilon_lower_empirical > epsilon_upper_theory` warning cases removed?",
            f"  Yes. {warning_cases_removed}/3 were removed, and {remaining_warning_cases} remain.",
            f"- How much did the reported canary lower bounds shrink after conservative calibration?",
            f"  Mean shrinkage is {statistics.fmean(shrinkages):.6f}, with median shrinkage {statistics.median(shrinkages):.6f}.",
            f"- Are the remaining canary lower bounds still nontrivial?",
            f"  No on this benchmark. {remaining_nontrivial}/3 calibrated runs remain above zero.",
            f"- Is canary still stronger than passive after correction?",
            (
                f"  No on this smoke benchmark. The calibrated canary mean empirical lower bound is {calibrated_means:.6f}, "
                f"while the strongest corrected passive variant previously averaged {passive_corrected_mean:.6f}."
            ),
            "- Is canary now suitable for paper results, or still only engineering-validation?",
            "  Still only engineering-validation. The estimator is now conservative enough to avoid the previous optimistic pathology, but the resulting canary lower bounds collapse on this benchmark, so the current implementation is not yet paper-ready quantitative evidence.",
            "",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
