from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CODEX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODEX_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "framework_ledger"

SUMMARY_SOURCES = {
    "smoke_matrix": RESULTS_DIR / "smoke_matrix" / "smoke_matrix_summary.json",
    "support_scaled_pilot": RESULTS_DIR / "support_scaled_pilot" / "support_scaled_pilot_summary.json",
    "raw_lira_pilot": RESULTS_DIR / "raw_lira_pilot" / "raw_lira_pilot_summary.json",
}
PATHOLOGY_SOURCE = (
    RESULTS_DIR / "raw_lira_pathology" / "raw_lira_pathology_summary.json"
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_attack_family(attack_name: str) -> str:
    if attack_name == "passive_raw_lira":
        return "passive_lira"
    if attack_name.startswith("passive_"):
        return "passive_threshold"
    if attack_name.startswith("canary_"):
        return "active_canary"
    return "other"


def infer_score_direction(row: dict[str, Any]) -> tuple[str | None, bool]:
    direction = row.get("score_direction")
    if direction == "higher":
        return "higher_is_member", False
    if direction == "lower":
        return "lower_is_member", False
    if direction in {"higher_is_member", "lower_is_member"}:
        return direction, False

    attack_name = str(row.get("attack", ""))
    score_name = str(row.get("score_name", ""))
    if attack_name == "passive_raw_lira":
        return "lower_is_member", True
    if score_name:
        return "higher_is_member", True
    return None, False


def infer_calibration_method(attack_name: str) -> str:
    if attack_name == "passive_raw_lira":
        return "shadow_model_likelihood_ratio"
    return "threshold_sweep"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_tags(
    row: dict[str, Any],
    normalized: dict[str, Any],
    pathology: dict[str, Any] | None,
) -> list[str]:
    tags: list[str] = []
    status = str(normalized.get("status", ""))
    if status != "ok":
        tags.append(status or "non_ok_status")

    backend = normalized.get("upper_bound_backend")
    if backend and backend != "google_dp_accounting":
        tags.append("non_exact_upper_backend")

    warning = str(normalized.get("warning") or "").lower()
    if "sparse tail" in warning:
        tags.append("sparse_tail")

    num_member = normalized.get("num_member_samples") or 0
    num_nonmember = normalized.get("num_nonmember_samples") or 0
    min_support = min(num_member, num_nonmember)
    if 0 < min_support < 128:
        tags.append("low_support")
    elif 128 <= min_support < 1000:
        tags.append("medium_support")
    elif min_support >= 1000:
        tags.append("high_support")

    score_direction = normalized.get("score_direction")
    if normalized.get("score_direction_inferred"):
        tags.append("score_direction_inferred")
    if score_direction == "lower_is_member":
        tags.append("score_direction_sensitive")

    epsilon_lower = normalized.get("epsilon_lower_conservative")
    epsilon_upper = normalized.get("epsilon_upper_tighter")
    if (
        epsilon_lower is not None
        and epsilon_upper is not None
        and epsilon_lower > epsilon_upper + 0.05
    ):
        tags.extend(["pathological_distribution", "exceeds_theoretical_upper"])

    epsilon_point = normalized.get("epsilon_lower_point")
    if (
        epsilon_lower is not None
        and epsilon_point is not None
        and epsilon_lower == 0.0
        and epsilon_point > 0.0
    ):
        tags.append("finite_sample_limited_signal")

    selected_tpr = normalized.get("selected_tpr")
    selected_fpr = normalized.get("selected_fpr")
    if selected_tpr is not None and selected_tpr <= 0.02:
        tags.append("extreme_tail_threshold")
    if selected_fpr is not None and selected_fpr == 0.0:
        tags.append("zero_fpr_threshold")

    if normalized.get("attack_family") == "active_canary":
        tags.append("active_stress_test")

    if pathology:
        tags.append("pathology_checked")
        preferred = pathology.get("preferred_direction")
        if preferred:
            tags.append("score_direction_sensitive")
            if preferred != score_direction:
                tags.append("score_direction_mismatch")
        auc_forward = safe_float(pathology.get("auc_forward"))
        auc_negated = safe_float(pathology.get("auc_negated"))
        if auc_forward is not None and auc_negated is not None:
            if abs(auc_negated - auc_forward) >= 0.1:
                tags.append("direction_flip_material")

    if (
        normalized.get("attack_family") == "passive_threshold"
        and min_support >= 1000
        and (epsilon_lower or 0.0) == 0.0
    ):
        tags.append("attack_limited_or_underpowered")

    if (
        normalized.get("attack_family") == "passive_lira"
        and (normalized.get("k_shadows") or 0) < 16
    ):
        tags.append("shadow_count_limited")

    return sorted(set(tags))


def assign_trust_tier(normalized: dict[str, Any], tags: list[str]) -> str:
    if "pathological_distribution" in tags or normalized.get("status") != "ok":
        return "invalidated"

    if normalized.get("upper_bound_backend") != "google_dp_accounting":
        return "exploratory"

    if "low_support" in tags:
        return "exploratory"

    if "sparse_tail" in tags and (
        "extreme_tail_threshold" in tags or "zero_fpr_threshold" in tags
    ):
        return "exploratory"

    epsilon_lower = normalized.get("epsilon_lower_conservative") or 0.0
    if epsilon_lower > 0.0:
        return "provisional"

    return "exploratory"


def support_profile(normalized: dict[str, Any]) -> str:
    pieces: list[str] = []
    if normalized.get("support_label"):
        pieces.append(str(normalized["support_label"]))
    if normalized.get("num_audit_seeds"):
        pieces.append(f"seeds={normalized['num_audit_seeds']}")
    if normalized.get("query_budget_per_seed"):
        pieces.append(f"budget={normalized['query_budget_per_seed']}")
    if normalized.get("k_shadows"):
        pieces.append(f"K={normalized['k_shadows']}")
    if normalized.get("canary_inserted_example_count"):
        pieces.append(
            f"canaries={normalized['canary_inserted_example_count']}"
        )
    return ", ".join(pieces) or "unspecified"


def normalize_row(
    source_name: str,
    row: dict[str, Any],
    pathology_index: dict[tuple[str, str, int | None], dict[str, Any]],
) -> dict[str, Any] | None:
    attack_name = str(row.get("attack", ""))
    if attack_name == "__training__":
        return None

    score_direction, inferred = infer_score_direction(row)
    k_shadows = row.get("k_shadows")
    pathology = pathology_index.get(
        (str(row.get("dataset")), attack_name, int(k_shadows) if k_shadows else None)
    )

    epsilon_upper_tighter = safe_float(row.get("epsilon_upper_tighter"))
    epsilon_lower_conservative = safe_float(row.get("epsilon_lower_conservative"))

    normalized: dict[str, Any] = {
        "row_id": (
            f"{source_name}::{row.get('dataset')}::{attack_name}::"
            f"{row.get('support_label', 'na')}::{k_shadows or 'na'}"
        ),
        "source_run": source_name,
        "dataset": row.get("dataset"),
        "attack_name": attack_name,
        "attack_family": infer_attack_family(attack_name),
        "audit_regime": row.get("audit_regime"),
        "status": row.get("status"),
        "score_name": row.get("score_name"),
        "score_direction": score_direction,
        "score_direction_inferred": inferred,
        "calibration_method": infer_calibration_method(attack_name),
        "upper_bound_backend": row.get("tighter_upper_backend"),
        "epsilon_upper_rdp": safe_float(row.get("epsilon_upper_rdp")),
        "epsilon_upper_tighter": epsilon_upper_tighter,
        "epsilon_lower_conservative": epsilon_lower_conservative,
        "epsilon_lower_point": safe_float(row.get("epsilon_lower_point")),
        "privacy_loss_gap_tighter": safe_float(
            row.get("privacy_loss_gap_tighter")
        ),
        "tightness_ratio_tighter": safe_float(
            row.get("tightness_ratio_tighter")
        ),
        "selected_tpr": safe_float(row.get("selected_tpr")),
        "selected_fpr": safe_float(row.get("selected_fpr")),
        "num_member_samples": int(row.get("num_member_samples") or 0),
        "num_nonmember_samples": int(row.get("num_nonmember_samples") or 0),
        "support_label": row.get("support_label"),
        "query_budget_per_seed": row.get("query_budget_per_seed"),
        "num_audit_seeds": row.get("num_audit_seeds"),
        "audit_seeds": row.get("audit_seeds"),
        "k_shadows": k_shadows,
        "canary_inserted_example_count": row.get(
            "canary_inserted_example_count"
        ),
        "member_favoring": row.get("member_favoring"),
        "warning": row.get("warning"),
        "member_score_mean": safe_float(row.get("member_score_mean")),
        "nonmember_score_mean": safe_float(row.get("nonmember_score_mean")),
        "score_gap": safe_float(row.get("score_gap")),
        "source_file": str(SUMMARY_SOURCES[source_name]),
    }

    if (
        normalized["privacy_loss_gap_tighter"] is None
        and epsilon_upper_tighter is not None
        and epsilon_lower_conservative is not None
    ):
        normalized["privacy_loss_gap_tighter"] = (
            epsilon_upper_tighter - epsilon_lower_conservative
        )

    if (
        normalized["tightness_ratio_tighter"] is None
        and epsilon_upper_tighter not in (None, 0.0)
        and epsilon_lower_conservative is not None
    ):
        normalized["tightness_ratio_tighter"] = (
            epsilon_lower_conservative / epsilon_upper_tighter
        )

    if pathology:
        normalized["pathology_preferred_direction"] = pathology.get(
            "preferred_direction"
        )
        normalized["pathology_auc_forward"] = safe_float(
            pathology.get("auc_forward")
        )
        normalized["pathology_auc_negated"] = safe_float(
            pathology.get("auc_negated")
        )

    tags = detect_tags(row, normalized, pathology)
    normalized["diagnostic_tags"] = tags
    normalized["trust_tier"] = assign_trust_tier(normalized, tags)
    normalized["narrative_safe"] = normalized["trust_tier"] in {
        "trusted",
        "provisional",
    }
    normalized["support_profile"] = support_profile(normalized)
    return normalized


def build_pathology_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    index: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("dataset")),
            str(row.get("attack")),
            int(row.get("k_shadows")) if row.get("k_shadows") else None,
        )
        preferred = row.get("preferred_direction")
        existing = index.get(key)
        if existing is None:
            index[key] = row
            continue
        if existing.get("direction_tested") != preferred and row.get(
            "direction_tested"
        ) == preferred:
            index[key] = row
    return index


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: (
                    "|".join(value)
                    if isinstance(value, list)
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(serializable)


def build_markdown_summary(rows: list[dict[str, Any]]) -> str:
    trust_counts = Counter(row["trust_tier"] for row in rows)
    total_rows = len(rows)
    provisional = [
        row
        for row in rows
        if row["trust_tier"] == "provisional" and row["narrative_safe"]
    ]
    invalidated = [row for row in rows if row["trust_tier"] == "invalidated"]
    exploratory = [row for row in rows if row["trust_tier"] == "exploratory"]

    lines = [
        "# Framework Ledger Summary",
        "",
        "## Scope",
        "",
        "- merged sidecar audit outputs from `smoke_matrix`, `support_scaled_pilot`, and `raw_lira_pilot`",
        "- attached `raw_lira_pathology` metadata where available",
        "- excluded pure training rows from the audit-comparison ledger",
        "",
        "## Counts",
        "",
        f"- total audit rows: `{total_rows}`",
        f"- provisional: `{trust_counts.get('provisional', 0)}`",
        f"- exploratory: `{trust_counts.get('exploratory', 0)}`",
        f"- invalidated: `{trust_counts.get('invalidated', 0)}`",
        f"- trusted: `{trust_counts.get('trusted', 0)}`",
        "",
    ]

    if not any(row["trust_tier"] == "trusted" for row in rows):
        lines.extend(
            [
                "## Trusted Status",
                "",
                "- no audit row currently qualifies as fully trusted",
                "- the best current rows are still pilot-scale and should be treated as provisional, not final",
                "",
            ]
        )

    if provisional:
        lines.extend(["## Best Provisional Rows", ""])
        provisional_sorted = sorted(
            provisional,
            key=lambda row: (
                -(row.get("tightness_ratio_tighter") or 0.0),
                row["dataset"],
                row["attack_name"],
            ),
        )
        for row in provisional_sorted[:6]:
            lines.append(
                "- "
                f"`{row['dataset']} + {row['attack_name']}` "
                f"({row['support_profile']}): "
                f"`eps_lower_cons={row.get('epsilon_lower_conservative'):.6f}`, "
                f"`eps_upper={row.get('epsilon_upper_tighter'):.6f}`, "
                f"`tightness={row.get('tightness_ratio_tighter', 0.0):.3f}`"
            )
        lines.append("")

    if invalidated:
        lines.extend(["## Invalidated / Pathological Rows", ""])
        for row in sorted(invalidated, key=lambda row: (row["dataset"], row["attack_name"])):
            lines.append(
                "- "
                f"`{row['dataset']} + {row['attack_name']}`: "
                f"`trust={row['trust_tier']}`, "
                f"`tags={'|'.join(row['diagnostic_tags'])}`"
            )
        lines.append("")

    lines.extend(["## Main Missing Pieces", ""])
    if any("score_direction_sensitive" in row["diagnostic_tags"] for row in rows):
        lines.append(
            "- score direction is now known to be framework-critical, but it is still not consistently logged by the original pipeline outputs"
        )
    if exploratory:
        lines.append(
            "- most rows remain exploratory because support is still limited, thresholds are tail-driven, or the attack family is not yet stable enough"
        )
    if any("pathological_distribution" in row["diagnostic_tags"] for row in rows):
        lines.append(
            "- `adult + raw_lira` still needs root-cause analysis before the framework can claim cross-dataset robustness"
        )
    lines.append(
        "- the next non-pilot step should be one canonical larger-scale matrix with the exact-PLD backend, explicit score direction, and a fixed trust-tagging protocol"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pathology_rows = load_rows(PATHOLOGY_SOURCE)
    pathology_index = build_pathology_index(pathology_rows)

    ledger_rows: list[dict[str, Any]] = []
    for source_name, path in SUMMARY_SOURCES.items():
        for row in load_rows(path):
            normalized = normalize_row(source_name, row, pathology_index)
            if normalized is not None:
                ledger_rows.append(normalized)

    trust_rank = {"trusted": 0, "provisional": 1, "exploratory": 2, "invalidated": 3}
    ledger_rows.sort(
        key=lambda row: (
            trust_rank.get(row["trust_tier"], 9),
            row["dataset"],
            row["attack_family"],
            row["attack_name"],
            str(row.get("support_label") or ""),
            row.get("k_shadows") or 0,
        )
    )

    payload = {
        "scope": "sidecar canonical audit ledger",
        "sources": {name: str(path) for name, path in SUMMARY_SOURCES.items()},
        "pathology_source": str(PATHOLOGY_SOURCE),
        "row_count": len(ledger_rows),
        "rows": ledger_rows,
    }
    write_json(OUTPUT_DIR / "framework_ledger.json", payload)
    write_csv(OUTPUT_DIR / "framework_ledger.csv", ledger_rows)
    (OUTPUT_DIR / "framework_ledger_summary.md").write_text(
        build_markdown_summary(ledger_rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
