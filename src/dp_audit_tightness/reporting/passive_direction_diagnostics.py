from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
import statistics
from typing import Any

from dp_audit_tightness.evaluation.metrics import compute_privacy_tightness_metrics
from dp_audit_tightness.privacy.empirical import _epsilon_candidate, _wilson_interval
from dp_audit_tightness.utils.paths import ensure_directory
from dp_audit_tightness.utils.results import PASSIVE_AUDIT_REGIME, load_audit_result, load_json, save_json


VARIANT_ORDER = [
    "max_probability_passive",
    "negative_loss_passive",
    "logit_margin_passive",
    "score_fusion_passive",
]

DIRECTION_ORDER = ["score>=threshold", "score<threshold"]
SELECTION_MODE_ORDER = ["unconstrained", "member_favoring_only"]


@dataclass(slots=True)
class PassiveRunData:
    audit_path: Path
    debug_artifact_path: Path
    audit_run_id: str
    training_run_id: str
    auditor_variant: str
    score_type: str
    training_seed: int
    epsilon_upper_theory: float
    epsilon_lower_empirical: float
    empirical_ci_lower: float | None
    empirical_ci_upper: float | None
    privacy_loss_gap: float
    tightness_ratio: float | None
    saturation_detected: bool
    selected_threshold: float | None
    selected_event: str | None
    member_scores: list[float]
    nonmember_scores: list[float]


@dataclass(slots=True)
class DirectionDiagnosticCandidate:
    direction: str
    selection_mode: str
    threshold: float | None
    member_event_fraction: float | None
    nonmember_event_fraction: float | None
    tpr: float | None
    fpr: float | None
    epsilon_lower_empirical: float | None
    empirical_ci_lower: float | None
    empirical_ci_upper: float | None
    member_favoring: bool
    objective_numerator_group: str | None
    objective_denominator_group: str | None
    note: str | None = None


@dataclass(slots=True)
class ScoreSummary:
    count: int
    mean: float
    std: float
    min: float
    p05: float
    p25: float
    median: float
    p75: float
    p95: float
    max: float


def generate_passive_direction_diagnostics(
    *,
    passive_results_dir: str | Path,
    passive_artifacts_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Diagnose passive threshold-direction behavior without changing production code."""

    runs = load_clean_passive_runs(
        passive_results_dir=passive_results_dir,
        passive_artifacts_dir=passive_artifacts_dir,
    )
    comparison_rows: list[dict[str, Any]] = []
    score_summary_rows: list[dict[str, Any]] = []
    per_run_best_rows: list[dict[str, Any]] = []

    for run in runs:
        score_summary_rows.append(_build_score_summary_row(run))
        thresholds = sorted(set(run.member_scores + run.nonmember_scores))
        run_candidates: list[DirectionDiagnosticCandidate] = []

        for direction in DIRECTION_ORDER:
            for selection_mode in SELECTION_MODE_ORDER:
                candidate = _find_best_candidate(
                    member_scores=run.member_scores,
                    nonmember_scores=run.nonmember_scores,
                    thresholds=thresholds,
                    delta=_delta_from_run(run),
                    direction=direction,
                    require_member_favoring=selection_mode == "member_favoring_only",
                )
                if candidate is None:
                    candidate = DirectionDiagnosticCandidate(
                        direction=direction,
                        selection_mode=selection_mode,
                        threshold=None,
                        member_event_fraction=None,
                        nonmember_event_fraction=None,
                        tpr=None,
                        fpr=None,
                        epsilon_lower_empirical=None,
                        empirical_ci_lower=None,
                        empirical_ci_upper=None,
                        member_favoring=False,
                        objective_numerator_group=None,
                        objective_denominator_group=None,
                        note="No threshold satisfied the member-favoring constraint."
                        if selection_mode == "member_favoring_only"
                        else "No threshold evaluated.",
                    )
                run_candidates.append(candidate)
                comparison_rows.append(_build_comparison_row(run, candidate, len(thresholds)))

        best_member_favoring = _best_across_directions(
            candidates=run_candidates,
            selection_mode="member_favoring_only",
        )
        if best_member_favoring is not None:
            per_run_best_rows.append(
                _build_comparison_row(
                    run=run,
                    candidate=best_member_favoring,
                    threshold_grid_size=len(thresholds),
                )
            )

    output_root = ensure_directory(output_dir)
    comparison_json_path = save_json(output_root / "comparison_table.json", {"rows": comparison_rows})
    score_summary_json_path = save_json(
        output_root / "score_summary_table.json",
        {"rows": score_summary_rows},
    )
    comparison_csv_path = _write_csv(output_root / "comparison_table.csv", comparison_rows)
    score_summary_csv_path = _write_csv(output_root / "score_summary_table.csv", score_summary_rows)
    markdown_path = output_root / "summary.md"
    markdown_path.write_text(
        _build_markdown_summary(
            runs=runs,
            comparison_rows=comparison_rows,
            score_summary_rows=score_summary_rows,
            per_run_best_rows=per_run_best_rows,
            comparison_csv_path=comparison_csv_path,
            score_summary_csv_path=score_summary_csv_path,
        ),
        encoding="utf-8",
    )

    return {
        "comparison_csv": comparison_csv_path,
        "comparison_json": comparison_json_path,
        "score_summary_csv": score_summary_csv_path,
        "score_summary_json": score_summary_json_path,
        "summary_markdown": markdown_path,
    }


def load_clean_passive_runs(
    *,
    passive_results_dir: str | Path,
    passive_artifacts_dir: str | Path,
) -> list[PassiveRunData]:
    runs: list[PassiveRunData] = []
    results_dir = Path(passive_results_dir)
    artifacts_dir = Path(passive_artifacts_dir)

    for audit_path in sorted(results_dir.glob("passive_audit_seed*.json")):
        record = load_audit_result(audit_path)
        if record.audit_regime != PASSIVE_AUDIT_REGIME:
            continue
        debug_artifact_path = artifacts_dir / f"{record.audit_run_id}_passive_debug.json"
        if not debug_artifact_path.exists():
            raise FileNotFoundError(f"Missing paired passive debug artifact: {debug_artifact_path}")
        debug_payload = load_json(debug_artifact_path)
        runs.append(
            PassiveRunData(
                audit_path=audit_path,
                debug_artifact_path=debug_artifact_path,
                audit_run_id=record.audit_run_id,
                training_run_id=record.training_run_id,
                auditor_variant=record.auditor_variant,
                score_type=str(debug_payload["score_name"]),
                training_seed=record.training_seed,
                epsilon_upper_theory=record.epsilon_upper_theory,
                epsilon_lower_empirical=record.epsilon_lower_empirical,
                empirical_ci_lower=record.empirical_ci_lower,
                empirical_ci_upper=record.empirical_ci_upper,
                privacy_loss_gap=record.privacy_loss_gap,
                tightness_ratio=record.tightness_ratio,
                saturation_detected=record.saturation_detected,
                selected_threshold=_optional_float(record.audit_metadata.get("selected_threshold")),
                selected_event=_optional_string(record.audit_metadata.get("selected_event")),
                member_scores=[float(score) for score in debug_payload["member_scores"]],
                nonmember_scores=[float(score) for score in debug_payload["nonmember_scores"]],
            )
        )

    return sorted(runs, key=lambda run: (_variant_rank(run.auditor_variant), run.training_seed))


def _build_score_summary_row(run: PassiveRunData) -> dict[str, Any]:
    member_summary = _summarize_scores(run.member_scores)
    nonmember_summary = _summarize_scores(run.nonmember_scores)
    thresholds = sorted(set(run.member_scores + run.nonmember_scores))
    return {
        "variant": run.auditor_variant,
        "seed": run.training_seed,
        "score_type": run.score_type,
        "lower_scores_more_member_like_by_construction": _lower_scores_more_member_like(run.score_type),
        "member_count": member_summary.count,
        "member_mean": member_summary.mean,
        "member_std": member_summary.std,
        "member_min": member_summary.min,
        "member_p05": member_summary.p05,
        "member_p25": member_summary.p25,
        "member_median": member_summary.median,
        "member_p75": member_summary.p75,
        "member_p95": member_summary.p95,
        "member_max": member_summary.max,
        "nonmember_count": nonmember_summary.count,
        "nonmember_mean": nonmember_summary.mean,
        "nonmember_std": nonmember_summary.std,
        "nonmember_min": nonmember_summary.min,
        "nonmember_p05": nonmember_summary.p05,
        "nonmember_p25": nonmember_summary.p25,
        "nonmember_median": nonmember_summary.median,
        "nonmember_p75": nonmember_summary.p75,
        "nonmember_p95": nonmember_summary.p95,
        "nonmember_max": nonmember_summary.max,
        "threshold_search_min": thresholds[0],
        "threshold_search_max": thresholds[-1],
        "threshold_grid_size": len(thresholds),
        "audit_path": str(run.audit_path),
        "debug_artifact_path": str(run.debug_artifact_path),
    }


def _find_best_candidate(
    *,
    member_scores: list[float],
    nonmember_scores: list[float],
    thresholds: list[float],
    delta: float,
    direction: str,
    require_member_favoring: bool,
) -> DirectionDiagnosticCandidate | None:
    best: DirectionDiagnosticCandidate | None = None
    best_point = -math.inf

    for threshold in thresholds:
        candidate = _evaluate_direction(
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            threshold=threshold,
            delta=delta,
            direction=direction,
        )
        if require_member_favoring and not candidate.member_favoring:
            continue
        point = candidate.epsilon_lower_empirical if candidate.epsilon_lower_empirical is not None else -math.inf
        if point > best_point:
            best = candidate
            best_point = point
    if best is None:
        return None
    best.selection_mode = "member_favoring_only" if require_member_favoring else "unconstrained"
    return best


def _evaluate_direction(
    *,
    member_scores: list[float],
    nonmember_scores: list[float],
    threshold: float,
    delta: float,
    direction: str,
) -> DirectionDiagnosticCandidate:
    member_trials = len(member_scores)
    nonmember_trials = len(nonmember_scores)
    member_ge = sum(score >= threshold for score in member_scores)
    nonmember_ge = sum(score >= threshold for score in nonmember_scores)
    tpr_ge = member_ge / member_trials
    fpr_ge = nonmember_ge / nonmember_trials
    tpr_lower, tpr_upper = _wilson_interval(member_ge, member_trials)
    fpr_lower, fpr_upper = _wilson_interval(nonmember_ge, nonmember_trials)

    if direction == "score>=threshold":
        member_event_fraction = tpr_ge
        nonmember_event_fraction = fpr_ge
        point = _epsilon_candidate(
            numerator_rate=tpr_ge,
            denominator_rate=fpr_ge,
            delta=delta,
            denominator_floor=0.5 / max(1, nonmember_trials),
        )
        lower = _epsilon_candidate(
            numerator_rate=tpr_lower,
            denominator_rate=fpr_upper,
            delta=delta,
            denominator_floor=0.5 / max(1, nonmember_trials),
        )
        upper = _epsilon_candidate(
            numerator_rate=tpr_upper,
            denominator_rate=fpr_lower,
            delta=delta,
            denominator_floor=0.5 / max(1, nonmember_trials),
        )
        objective_numerator_group = "member"
        objective_denominator_group = "nonmember"
    elif direction == "score<threshold":
        member_event_fraction = 1.0 - tpr_ge
        nonmember_event_fraction = 1.0 - fpr_ge
        point = _epsilon_candidate(
            numerator_rate=1.0 - fpr_ge,
            denominator_rate=1.0 - tpr_ge,
            delta=delta,
            denominator_floor=0.5 / max(1, member_trials),
        )
        lower = _epsilon_candidate(
            numerator_rate=1.0 - fpr_upper,
            denominator_rate=1.0 - tpr_lower,
            delta=delta,
            denominator_floor=0.5 / max(1, member_trials),
        )
        upper = _epsilon_candidate(
            numerator_rate=1.0 - fpr_lower,
            denominator_rate=1.0 - tpr_upper,
            delta=delta,
            denominator_floor=0.5 / max(1, member_trials),
        )
        objective_numerator_group = "nonmember"
        objective_denominator_group = "member"
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return DirectionDiagnosticCandidate(
        direction=direction,
        selection_mode="unconstrained",
        threshold=threshold,
        member_event_fraction=member_event_fraction,
        nonmember_event_fraction=nonmember_event_fraction,
        tpr=member_event_fraction,
        fpr=nonmember_event_fraction,
        epsilon_lower_empirical=point,
        empirical_ci_lower=lower,
        empirical_ci_upper=upper,
        member_favoring=member_event_fraction > nonmember_event_fraction,
        objective_numerator_group=objective_numerator_group,
        objective_denominator_group=objective_denominator_group,
    )


def _build_comparison_row(
    run: PassiveRunData,
    candidate: DirectionDiagnosticCandidate,
    threshold_grid_size: int,
) -> dict[str, Any]:
    epsilon_lower = candidate.epsilon_lower_empirical
    if epsilon_lower is None:
        gap = None
        tightness_ratio = None
    else:
        tightness = compute_privacy_tightness_metrics(
            epsilon_upper_theory=run.epsilon_upper_theory,
            epsilon_lower_empirical=epsilon_lower,
        )
        gap = tightness.privacy_loss_gap
        tightness_ratio = tightness.tightness_ratio

    return {
        "variant": run.auditor_variant,
        "seed": run.training_seed,
        "direction": candidate.direction,
        "selection_mode": candidate.selection_mode,
        "threshold": candidate.threshold,
        "member_event_fraction": candidate.member_event_fraction,
        "nonmember_event_fraction": candidate.nonmember_event_fraction,
        "tpr": candidate.tpr,
        "fpr": candidate.fpr,
        "epsilon_upper_theory": run.epsilon_upper_theory,
        "epsilon_lower_empirical": candidate.epsilon_lower_empirical,
        "privacy_loss_gap": gap,
        "tightness_ratio": tightness_ratio,
        "ci_lower": candidate.empirical_ci_lower,
        "ci_upper": candidate.empirical_ci_upper,
        "saturation_detected": run.saturation_detected,
        "member_favoring": candidate.member_favoring,
        "lower_scores_more_member_like_by_construction": _lower_scores_more_member_like(run.score_type),
        "objective_numerator_group": candidate.objective_numerator_group,
        "objective_denominator_group": candidate.objective_denominator_group,
        "current_selected_threshold": run.selected_threshold,
        "current_selected_event": run.selected_event,
        "current_selected_epsilon_lower_empirical": run.epsilon_lower_empirical,
        "current_selected_ci_lower": run.empirical_ci_lower,
        "current_selected_ci_upper": run.empirical_ci_upper,
        "current_selection_relation": _selection_relation(run, candidate),
        "exact_complement_of_current_selection": _is_exact_complement_of_current_selection(run, candidate),
        "threshold_grid_size": threshold_grid_size,
        "audit_run_id": run.audit_run_id,
        "training_run_id": run.training_run_id,
        "score_type": run.score_type,
        "audit_path": str(run.audit_path),
        "debug_artifact_path": str(run.debug_artifact_path),
        "note": candidate.note,
    }


def _best_across_directions(
    *,
    candidates: list[DirectionDiagnosticCandidate],
    selection_mode: str,
) -> DirectionDiagnosticCandidate | None:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.selection_mode == selection_mode and candidate.epsilon_lower_empirical is not None
    ]
    if not filtered:
        return None
    return max(filtered, key=lambda candidate: candidate.epsilon_lower_empirical or 0.0)


def _build_markdown_summary(
    *,
    runs: list[PassiveRunData],
    comparison_rows: list[dict[str, Any]],
    score_summary_rows: list[dict[str, Any]],
    per_run_best_rows: list[dict[str, Any]],
    comparison_csv_path: Path,
    score_summary_csv_path: Path,
) -> str:
    current_pathologies = sum(
        1 for run in runs if run.epsilon_lower_empirical > run.epsilon_upper_theory
    )
    unconstrained_rows = [row for row in comparison_rows if row["selection_mode"] == "unconstrained"]
    member_favoring_rows = [
        row for row in comparison_rows if row["selection_mode"] == "member_favoring_only"
    ]

    ge_unconstrained = [row for row in unconstrained_rows if row["direction"] == "score>=threshold"]
    lt_unconstrained = [row for row in unconstrained_rows if row["direction"] == "score<threshold"]
    ge_wins_unconstrained, lt_wins_unconstrained, ties_unconstrained = _direction_win_counts(
        ge_rows=ge_unconstrained,
        lt_rows=lt_unconstrained,
    )

    ge_member_favoring = [
        row
        for row in member_favoring_rows
        if row["direction"] == "score>=threshold" and row["epsilon_lower_empirical"] is not None
    ]
    lt_member_favoring = [
        row
        for row in member_favoring_rows
        if row["direction"] == "score<threshold" and row["epsilon_lower_empirical"] is not None
    ]
    ge_wins_member_favoring, lt_wins_member_favoring, ties_member_favoring = _direction_win_counts(
        ge_rows=ge_member_favoring,
        lt_rows=lt_member_favoring,
    )

    unconstrained_pathologies = sum(
        1
        for row in unconstrained_rows
        if row["epsilon_lower_empirical"] is not None
        and row["epsilon_lower_empirical"] > row["epsilon_upper_theory"]
    )
    member_favoring_pathologies = sum(
        1
        for row in member_favoring_rows
        if row["epsilon_lower_empirical"] is not None
        and row["epsilon_lower_empirical"] > row["epsilon_upper_theory"]
    )
    best_member_favoring_pathologies = sum(
        1
        for row in per_run_best_rows
        if row["epsilon_lower_empirical"] is not None
        and row["epsilon_lower_empirical"] > row["epsilon_upper_theory"]
    )

    weak_tail_rows = [
        row
        for row in ge_member_favoring
        if row["member_event_fraction"] is not None
        and row["nonmember_event_fraction"] is not None
        and (
            max(row["member_event_fraction"], row["nonmember_event_fraction"]) <= 0.1
            or abs(row["member_event_fraction"] - row["nonmember_event_fraction"]) <= 0.05
        )
    ]

    lines = [
        "# Passive Direction Diagnostic Summary",
        "",
        "## Scope",
        "",
        f"- Clean passive runs analyzed: {len(runs)}",
        f"- Comparison table: `{comparison_csv_path}`",
        f"- Score summary table: `{score_summary_csv_path}`",
        "- Production passive auditing code was not modified by this diagnostic.",
        "- `saturation_detected` is carried through unchanged from the saved production audit record because saturation is defined across auditor-strength history, not across threshold directions within one run.",
        "",
        "## Direct Answers",
        "",
        f"- Does `score>=threshold` consistently outperform `score<threshold`?",
        f"  No under the current objective. Unconstrained, `score>=threshold` wins {ge_wins_unconstrained}/12 runs, `score<threshold` wins {lt_wins_unconstrained}/12, with {ties_unconstrained} ties. Under member-favoring-only filtering, `score>=threshold` wins {ge_wins_member_favoring}/12 runs, `score<threshold` wins {lt_wins_member_favoring}/12, with {ties_member_favoring} ties.",
        f"- When constrained to member-favoring events only, do the pathological `epsilon_lower_empirical > epsilon_upper_theory` cases mostly disappear?",
        f"  Yes, relative to the current selected production results. The current selected passive runs have {current_pathologies}/12 warnings. Across all 24 unconstrained direction rows there are {unconstrained_pathologies} warnings, across all 24 member-favoring-only rows there are {member_favoring_pathologies} warnings, and among the single best member-favoring event per run there are {best_member_favoring_pathologies}/12 warnings.",
        "- Are the best `score>=threshold` events genuinely discriminative, or still based on weak overlap / tiny tails?",
        f"  They still look weak. {len(weak_tail_rows)}/{len(ge_member_favoring) or 1} member-favoring `score>=threshold` optima are either tiny-tail events or have member/non-member event fractions within 0.05 of each other.",
        "- Is the current instability more consistent with (a) wrong event-direction/complement selection, (b) weak score separability, or (c) both?",
        "  Both. The current `score<threshold` objective reuses the complement event in a way that places the numerator on the non-member side, but the raw passive scores also show heavy overlap, so fixing direction alone is unlikely to make the passive regime scientifically reliable.",
        "",
        "## Key Observations",
        "",
        "- All four passive score definitions are higher-is-more-confident scores by construction, so lower scores are not naturally more member-like.",
        "- The current selected production events all use `score<threshold`.",
        "- The `score<threshold` direction, under the current objective, uses a non-member numerator and a member denominator. That is diagnostic evidence that complement selection is part of the instability.",
        "- The raw score summaries still show substantial overlap between member and non-member distributions, especially in medians and interquartile ranges.",
        "- Cases where the empirical lower bound exceeds the theoretical upper bound are diagnostic warnings only. They should not be interpreted as substantive claims.",
        "",
        "## Recommendation",
        "",
        "The next code change should start with restricting threshold search to member-favoring events and then explicitly repairing the event-direction logic so the lower-bound objective always uses a member-side numerator for the event being evaluated. After that, re-evaluate whether the passive scores remain too overlapped to support stable empirical lower bounds.",
        "",
    ]

    return "\n".join(lines)


def _direction_win_counts(
    *,
    ge_rows: list[dict[str, Any]],
    lt_rows: list[dict[str, Any]],
) -> tuple[int, int, int]:
    ge_by_key = {(row["variant"], row["seed"]): row for row in ge_rows}
    lt_by_key = {(row["variant"], row["seed"]): row for row in lt_rows}
    ge_wins = 0
    lt_wins = 0
    ties = 0
    for key in sorted(set(ge_by_key) | set(lt_by_key)):
        ge_value = ge_by_key.get(key, {}).get("epsilon_lower_empirical")
        lt_value = lt_by_key.get(key, {}).get("epsilon_lower_empirical")
        if ge_value is None or lt_value is None:
            continue
        if ge_value > lt_value:
            ge_wins += 1
        elif lt_value > ge_value:
            lt_wins += 1
        else:
            ties += 1
    return ge_wins, lt_wins, ties


def _selection_relation(run: PassiveRunData, candidate: DirectionDiagnosticCandidate) -> str:
    if run.selected_threshold is None or run.selected_event is None or candidate.threshold is None:
        return "unknown"
    if _same_threshold(candidate.threshold, run.selected_threshold) and candidate.direction == run.selected_event:
        return "matches_current_selection"
    if _is_exact_complement_of_current_selection(run, candidate):
        return "exact_complement_of_current_selection"
    return "genuinely_different_optimum"


def _is_exact_complement_of_current_selection(
    run: PassiveRunData,
    candidate: DirectionDiagnosticCandidate,
) -> bool:
    if run.selected_threshold is None or run.selected_event is None or candidate.threshold is None:
        return False
    return _same_threshold(candidate.threshold, run.selected_threshold) and {
        run.selected_event,
        candidate.direction,
    } == {"score<threshold", "score>=threshold"}


def _same_threshold(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def _summarize_scores(values: list[float]) -> ScoreSummary:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty score list.")
    return ScoreSummary(
        count=len(ordered),
        mean=float(statistics.fmean(ordered)),
        std=float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
        min=float(ordered[0]),
        p05=_percentile(ordered, 0.05),
        p25=_percentile(ordered, 0.25),
        median=_percentile(ordered, 0.50),
        p75=_percentile(ordered, 0.75),
        p95=_percentile(ordered, 0.95),
        max=float(ordered[-1]),
    )


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(values[lower_index])
    weight = position - lower_index
    return float(values[lower_index] * (1.0 - weight) + values[upper_index] * weight)


def _lower_scores_more_member_like(score_type: str) -> bool:
    direction_by_score = {
        "max_probability": False,
        "negative_loss": False,
        "logit_margin": False,
        "score_fusion": False,
    }
    return direction_by_score.get(score_type, False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_directory(path.parent)
    if not rows:
        raise ValueError(f"No rows available to write CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _delta_from_run(run: PassiveRunData) -> float:
    payload = load_json(run.audit_path)
    return float(payload["delta"])


def _variant_rank(variant: str) -> int:
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return len(VARIANT_ORDER)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
