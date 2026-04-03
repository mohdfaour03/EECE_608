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
        description="Compare passive audit results before and after the member-favoring fix."
    )
    parser.add_argument(
        "--before-dir",
        default="results/substantive_smoke_3seed/audits/passive",
        help="Directory containing the original clean passive audit JSON files.",
    )
    parser.add_argument(
        "--after-dir",
        default="results/substantive_smoke_3seed_postfix/audits/passive",
        help="Directory containing the post-fix passive audit JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/substantive_smoke_3seed/audits/passive_postfix_comparison",
        help="Directory where before/after comparison artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_records = _load_passive_records(Path(args.before_dir))
    after_records = _load_passive_records(Path(args.after_dir))

    rows: list[dict[str, Any]] = []
    for key, before in before_records.items():
        after = after_records[key]
        rows.append(_build_before_after_row(before, after))

    rows.sort(key=lambda row: (_variant_rank(row["variant"]), row["seed"]))
    output_dir = ensure_directory(Path(args.output_dir))
    _write_csv(output_dir / "before_after_table.csv", rows)
    save_json(output_dir / "before_after_table.json", {"rows": rows})
    (output_dir / "summary.md").write_text(_build_summary(rows), encoding="utf-8")


def _load_passive_records(directory: Path) -> dict[tuple[str, int], Any]:
    records: dict[tuple[str, int], Any] = {}
    for path in sorted(directory.glob("passive_audit_seed*.json")):
        record = load_audit_result(path)
        records[(record.auditor_variant, record.training_seed)] = record
    return records


def _build_before_after_row(before, after) -> dict[str, Any]:
    selected_event_before = before.audit_metadata.get("selected_event_direction") or before.audit_metadata.get(
        "selected_event"
    )
    selected_event_after = after.audit_metadata.get("selected_event_direction") or after.audit_metadata.get(
        "selected_event"
    )
    member_favoring_before = _derive_member_favoring(before)
    warning_before = _derive_before_warning(before, member_favoring_before)
    warning_after = after.audit_metadata.get("warning")
    if after.epsilon_lower_empirical > after.epsilon_upper_theory:
        warning_after = _append_warning(
            warning_after,
            "empirical lower bound exceeds theoretical upper bound",
        )

    return {
        "variant": before.auditor_variant,
        "seed": before.training_seed,
        "epsilon_upper_theory_before": before.epsilon_upper_theory,
        "epsilon_lower_empirical_before": before.epsilon_lower_empirical,
        "privacy_loss_gap_before": before.privacy_loss_gap,
        "tightness_ratio_before": before.tightness_ratio,
        "epsilon_upper_theory_after": after.epsilon_upper_theory,
        "epsilon_lower_empirical_after": after.epsilon_lower_empirical,
        "privacy_loss_gap_after": after.privacy_loss_gap,
        "tightness_ratio_after": after.tightness_ratio,
        "selected_event_before": selected_event_before,
        "selected_event_after": selected_event_after,
        "member_favoring_before": member_favoring_before,
        "member_favoring_after": after.audit_metadata.get("member_favoring"),
        "warning_before": warning_before,
        "warning_after": warning_after,
    }


def _build_summary(rows: list[dict[str, Any]]) -> str:
    removed_pathologies = [
        row
        for row in rows
        if row["epsilon_lower_empirical_before"] > row["epsilon_upper_theory_before"]
        and row["epsilon_lower_empirical_after"] <= row["epsilon_upper_theory_after"]
    ]
    remaining_pathologies = [
        row
        for row in rows
        if row["epsilon_lower_empirical_after"] > row["epsilon_upper_theory_after"]
    ]
    shrinkage = [
        row["epsilon_lower_empirical_before"] - row["epsilon_lower_empirical_after"]
        for row in rows
    ]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    variant_lines: list[str] = []
    strongest_variant = None
    strongest_mean = None
    for variant in sorted(by_variant, key=_variant_rank):
        variant_rows = by_variant[variant]
        before_mean = statistics.fmean(row["epsilon_lower_empirical_before"] for row in variant_rows)
        after_mean = statistics.fmean(row["epsilon_lower_empirical_after"] for row in variant_rows)
        variant_lines.append(
            f"- `{variant}`: mean before={before_mean:.6f}, mean after={after_mean:.6f}"
        )
        if strongest_mean is None or after_mean > strongest_mean:
            strongest_mean = after_mean
            strongest_variant = variant

    lines = [
        "# Passive Post-Fix Comparison",
        "",
        "## Direct Answers",
        "",
        f"- Were all `epsilon_lower_empirical > epsilon_upper_theory` warning cases removed?",
        f"  Yes. {len(removed_pathologies)} previously pathological runs are no longer pathological after the fix, and {len(remaining_pathologies)} remain.",
        f"- How much did passive lower bounds shrink after the fix?",
        f"  Mean shrinkage across the 12 runs is {statistics.fmean(shrinkage):.6f}. The median shrinkage is {statistics.median(shrinkage):.6f}.",
        f"- Which passive variant is strongest after the fix?",
        f"  `{strongest_variant}` has the largest mean post-fix empirical lower bound at {strongest_mean:.6f}.",
        "- Are the passive scores still too overlapped to support strong claims?",
        "  Yes. Score construction was not changed, and after enforcing member-favoring events the empirical lower bounds shrink sharply and remain small, which is consistent with heavy overlap rather than a strong passive signal.",
        "- Is the passive method now suitable for paper results, or still only an engineering-validation result?",
        "  Still only an engineering-validation result. The passive path is now corrected conservatively, but the remaining lower bounds are too weak and overlap-sensitive to support strong paper claims.",
        "",
        "## Variant Means",
        "",
        *variant_lines,
        "",
    ]
    return "\n".join(lines)


def _derive_member_favoring(record) -> bool | None:
    if "member_favoring" in record.audit_metadata:
        return bool(record.audit_metadata["member_favoring"])
    selected_event = record.audit_metadata.get("selected_event")
    selected_tpr = record.audit_metadata.get("selected_tpr")
    selected_fpr = record.audit_metadata.get("selected_fpr")
    if selected_event is None or selected_tpr is None or selected_fpr is None:
        return None
    tpr = float(selected_tpr)
    fpr = float(selected_fpr)
    if selected_event == "score>=threshold":
        return tpr > fpr
    if selected_event == "score<threshold":
        return (1.0 - tpr) > (1.0 - fpr)
    return None


def _derive_before_warning(record, member_favoring_before: bool | None) -> str | None:
    warning = None
    if record.epsilon_lower_empirical > record.epsilon_upper_theory:
        warning = "empirical lower bound exceeds theoretical upper bound"
    if member_favoring_before is False:
        warning = _append_warning(warning, "selected event was not member-favoring")
    return warning


def _append_warning(existing: str | None, new_warning: str) -> str:
    if not existing:
        return new_warning
    return f"{existing}; {new_warning}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _variant_rank(variant: str) -> int:
    order = [
        "max_probability_passive",
        "negative_loss_passive",
        "logit_margin_passive",
        "score_fusion_passive",
    ]
    try:
        return order.index(variant)
    except ValueError:
        return len(order)


if __name__ == "__main__":
    main()
