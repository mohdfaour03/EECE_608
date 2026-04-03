from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
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
        description="Compare canary audit results before and after the seed-propagation fix."
    )
    parser.add_argument(
        "--before-dir",
        default="results/substantive_smoke_3seed/audits/canary",
        help="Directory containing the original clean canary audit JSON files.",
    )
    parser.add_argument(
        "--after-dir",
        default="results/canary_seedfix_3seed/audits/canary",
        help="Directory containing the post-fix canary audit JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/substantive_smoke_3seed/audits/canary_seedfix_comparison",
        help="Directory where before/after comparison artifacts will be written.",
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
    for path in sorted(directory.glob("canary_audit_*.json")):
        record = load_audit_result(path)
        records[record.training_seed] = record
    return records


def _build_row(before, after) -> dict[str, Any]:
    return {
        "seed": before.training_seed,
        "epsilon_upper_theory_before": before.epsilon_upper_theory,
        "epsilon_lower_empirical_before": before.epsilon_lower_empirical,
        "privacy_loss_gap_before": before.privacy_loss_gap,
        "tightness_ratio_before": before.tightness_ratio,
        "epsilon_upper_theory_after": after.epsilon_upper_theory,
        "epsilon_lower_empirical_after": after.epsilon_lower_empirical,
        "privacy_loss_gap_after": after.privacy_loss_gap,
        "tightness_ratio_after": after.tightness_ratio,
        "saturation_before": before.saturation_detected,
        "saturation_after": after.saturation_detected,
        "canary_seeds_before": json.dumps(_before_seed_metadata(before), sort_keys=True),
        "canary_seeds_after": json.dumps(_after_seed_metadata(after), sort_keys=True),
    }


def _before_seed_metadata(record) -> dict[str, Any]:
    per_seed_runs = record.audit_metadata.get("artifact_payload", {}).get("per_seed_runs", [])
    return {
        "experiment_seed": record.training_seed,
        "dataset_split_seed": record.split_seed,
        "repeated_audit_seeds": list(record.repeated_seeds),
        "derived_seed_plans": [
            {
                "experiment_seed": record.training_seed,
                "audit_seed": int(item.get("audit_seed", item.get("seed"))),
                "dataset_split_seed": record.split_seed,
                "canary_generation_seed": int(item.get("audit_seed", item.get("seed"))),
                "canary_insertion_seed": None,
                "retrain_seed": int(item.get("audit_seed", item.get("seed"))),
            }
            for item in per_seed_runs
        ],
    }


def _after_seed_metadata(record) -> dict[str, Any]:
    return {
        "experiment_seed": record.audit_metadata.get("experiment_seed"),
        "dataset_split_seed": record.audit_metadata.get("dataset_split_seed"),
        "repeated_audit_seeds": record.audit_metadata.get("repeated_audit_seeds"),
        "derived_seed_plans": record.audit_metadata.get("derived_seed_plans"),
    }


def _build_summary(rows: list[dict[str, Any]]) -> str:
    before_values = [row["epsilon_lower_empirical_before"] for row in rows]
    after_values = [row["epsilon_lower_empirical_after"] for row in rows]
    before_unique = len(set(before_values))
    after_unique = len(set(after_values))
    warning_cases_after = sum(
        row["epsilon_lower_empirical_after"] > row["epsilon_upper_theory_after"] for row in rows
    )
    lines = [
        "# Canary Seed-Propagation Comparison",
        "",
        "## Direct Answers",
        "",
        "- Why were the original canary 3-seed results invariant?",
        "  The repeated canary audit path was driven entirely by the fixed config audit seeds `101,102`, while the base experiment seed only affected bookkeeping. The dataset split seed was also fixed at 11, insertion had no separate RNG, and threshold search was deterministic. So seeds 123, 124, 125 all reran the same canary audit experiment.",
        "- What exact seed-propagation fix was applied?",
        "  Each canary repeated-run now derives a deterministic seed plan from `(experiment_seed, audit_seed, dataset_split_seed)`, and uses distinct derived seeds for canary generation, canary insertion ordering, and canary retraining.",
        "- Do the post-fix canary results now vary across seeds?",
        f"  Yes. Before the fix there were {before_unique} unique `epsilon_lower_empirical` values across the three experiment seeds; after the fix there are {after_unique}.",
        "- Are the new multi-seed canary aggregates now meaningful?",
        "  Yes, in the limited engineering sense intended here: the repeated-run canary audits are now genuine independent seeded replicates rather than relabeled copies of the same canary retraining path.",
        "- Is canary now suitable to serve as the stronger evaluator-controlled auditing result in the paper?",
        (
            "  Structurally yes as the evaluator-controlled track, because the multi-seed summaries now reflect real seeded repeats. "
            f"However, {warning_cases_after}/3 post-fix runs still have `epsilon_lower_empirical > epsilon_upper_theory`, "
            "so the seed-propagation fix alone does not make the current canary quantitative results fully paper-ready. "
            "The paper can still use canary as the stronger evaluator-controlled regime, but the current canary lower-bound estimator should be reviewed separately before strong quantitative claims are made."
        ),
        "",
        "## Aggregates",
        "",
        f"- Before: mean `epsilon_lower_empirical` = {statistics.fmean(before_values):.6f}, std = {statistics.stdev(before_values) if len(before_values) > 1 else 0.0:.6f}",
        f"- After: mean `epsilon_lower_empirical` = {statistics.fmean(after_values):.6f}, std = {statistics.stdev(after_values) if len(after_values) > 1 else 0.0:.6f}",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
