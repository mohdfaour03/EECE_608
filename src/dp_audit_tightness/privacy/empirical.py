from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(slots=True)
class EmpiricalEpsilonEstimate:
    epsilon_lower_empirical: float
    epsilon_lower_empirical_point_estimate: float
    epsilon_lower_empirical_conservative: float
    empirical_ci_lower: float | None
    empirical_ci_upper: float | None
    estimation_method: str
    delta: float
    selected_threshold: float | None = None
    selected_event: str | None = None
    selected_event_direction: str | None = None
    selected_tpr: float | None = None
    selected_fpr: float | None = None
    member_event_fraction: float | None = None
    nonmember_event_fraction: float | None = None
    member_favoring: bool | None = None
    no_member_favoring_event_found: bool = False
    selected_event_is_tiny_tail: bool | None = None
    selected_member_event_count: int | None = None
    selected_nonmember_event_count: int | None = None
    warning_message: str | None = None
    num_member_samples: int = 0
    num_nonmember_samples: int = 0


def estimate_empirical_lower_bound(
    *,
    member_scores: Sequence[float] | None = None,
    nonmember_scores: Sequence[float] | None = None,
    audit_statistics: Mapping[str, float] | None = None,
    delta: float,
    score_direction: str = "higher",
    align_event_to_score_direction: bool = False,
    require_member_favoring: bool = False,
    report_confidence_supported_lower_bound: bool = False,
    tiny_tail_fraction_threshold: float = 0.125,
    tiny_tail_min_event_count: int = 5,
) -> EmpiricalEpsilonEstimate:
    """Estimate an empirical lower bound on privacy loss from auditor outputs.

    For first-pass implementations, this uses a threshold sweep over member and
    non-member score distributions and converts the best observed hypothesis-testing
    event into a demonstrated lower bound on privacy loss.
    """

    if member_scores is not None and nonmember_scores is not None:
        return _estimate_from_score_distributions(
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            delta=delta,
            score_direction=score_direction,
            align_event_to_score_direction=align_event_to_score_direction,
            require_member_favoring=require_member_favoring,
            report_confidence_supported_lower_bound=report_confidence_supported_lower_bound,
            tiny_tail_fraction_threshold=tiny_tail_fraction_threshold,
            tiny_tail_min_event_count=tiny_tail_min_event_count,
        )
    if audit_statistics is None:
        raise ValueError("Provide either score distributions or audit_statistics.")

    epsilon_candidate = max(0.0, float(audit_statistics.get("epsilon_candidate", 0.0)))
    ci_half_width = max(0.0, float(audit_statistics.get("ci_half_width", 0.0)))
    return EmpiricalEpsilonEstimate(
        epsilon_lower_empirical=epsilon_candidate,
        epsilon_lower_empirical_point_estimate=epsilon_candidate,
        epsilon_lower_empirical_conservative=max(0.0, epsilon_candidate - ci_half_width),
        empirical_ci_lower=max(0.0, epsilon_candidate - ci_half_width),
        empirical_ci_upper=epsilon_candidate + ci_half_width,
        estimation_method="direct_candidate_passthrough",
        delta=delta,
    )


def _estimate_from_score_distributions(
    *,
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    delta: float,
    score_direction: str,
    align_event_to_score_direction: bool,
    require_member_favoring: bool,
    report_confidence_supported_lower_bound: bool,
    tiny_tail_fraction_threshold: float,
    tiny_tail_min_event_count: int,
) -> EmpiricalEpsilonEstimate:
    member = [float(score) for score in member_scores]
    nonmember = [float(score) for score in nonmember_scores]
    if not member or not nonmember:
        raise ValueError("Member and non-member score lists must both be non-empty.")

    if score_direction == "lower":
        transformed_member = [-score for score in member]
        transformed_nonmember = [-score for score in nonmember]
        selected_event_direction = "score<threshold"
        threshold_transform = lambda value: -value
    elif score_direction != "higher":
        raise ValueError(f"Unsupported score_direction: {score_direction}")
    else:
        transformed_member = member
        transformed_nonmember = nonmember
        selected_event_direction = "score>=threshold"
        threshold_transform = lambda value: value

    thresholds = sorted(set(transformed_member + transformed_nonmember))
    if not thresholds:
        raise ValueError("At least one score threshold is required.")

    if align_event_to_score_direction:
        return _estimate_member_aligned_threshold_sweep(
            member_scores=transformed_member,
            nonmember_scores=transformed_nonmember,
            delta=delta,
            thresholds=thresholds,
            selected_event_direction=selected_event_direction,
            threshold_transform=threshold_transform,
            require_member_favoring=require_member_favoring,
            report_confidence_supported_lower_bound=report_confidence_supported_lower_bound,
            tiny_tail_fraction_threshold=tiny_tail_fraction_threshold,
            tiny_tail_min_event_count=tiny_tail_min_event_count,
        )
    return _estimate_legacy_threshold_sweep(
        member_scores=transformed_member,
        nonmember_scores=transformed_nonmember,
        delta=delta,
        thresholds=thresholds,
        report_confidence_supported_lower_bound=report_confidence_supported_lower_bound,
        tiny_tail_fraction_threshold=tiny_tail_fraction_threshold,
        tiny_tail_min_event_count=tiny_tail_min_event_count,
    )


def _estimate_member_aligned_threshold_sweep(
    *,
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    delta: float,
    thresholds: Sequence[float],
    selected_event_direction: str,
    threshold_transform,
    require_member_favoring: bool,
    report_confidence_supported_lower_bound: bool,
    tiny_tail_fraction_threshold: float,
    tiny_tail_min_event_count: int,
) -> EmpiricalEpsilonEstimate:
    best_point = -math.inf
    best_lower = 0.0
    best_upper = 0.0
    best_threshold: float | None = None
    best_tpr = 0.0
    best_fpr = 0.0
    best_member_favoring = False
    best_member_count = 0
    best_nonmember_count = 0

    for threshold in thresholds:
        member_successes = sum(score >= threshold for score in member_scores)
        nonmember_successes = sum(score >= threshold for score in nonmember_scores)
        candidate = _evaluate_member_aligned_threshold(
            member_successes=member_successes,
            member_trials=len(member_scores),
            nonmember_successes=nonmember_successes,
            nonmember_trials=len(nonmember_scores),
            delta=delta,
        )
        if require_member_favoring and not candidate["member_favoring"]:
            continue
        if candidate["point"] > best_point:
            best_point = candidate["point"]
            best_lower = candidate["lower"]
            best_upper = candidate["upper"]
            best_threshold = threshold
            best_tpr = candidate["tpr"]
            best_fpr = candidate["fpr"]
            best_member_favoring = bool(candidate["member_favoring"])
            best_member_count = member_successes
            best_nonmember_count = nonmember_successes

    if best_threshold is None:
        warning_message = (
            "No member-favoring event found on the scanned threshold grid; "
            "returning a conservative zero empirical lower bound."
        )
        return EmpiricalEpsilonEstimate(
            epsilon_lower_empirical=0.0,
            epsilon_lower_empirical_point_estimate=0.0,
            epsilon_lower_empirical_conservative=0.0,
            empirical_ci_lower=0.0,
            empirical_ci_upper=0.0,
            estimation_method="threshold_sweep_member_aligned_no_member_favoring_event",
            delta=delta,
            selected_threshold=None,
            selected_event=None,
            selected_event_direction=selected_event_direction,
            selected_tpr=None,
            selected_fpr=None,
            member_event_fraction=None,
            nonmember_event_fraction=None,
            member_favoring=False,
            no_member_favoring_event_found=True,
            selected_event_is_tiny_tail=None,
            selected_member_event_count=None,
            selected_nonmember_event_count=None,
            warning_message=warning_message,
            num_member_samples=len(member_scores),
            num_nonmember_samples=len(nonmember_scores),
        )

    point_estimate = max(0.0, best_point)
    conservative_estimate = max(0.0, best_lower)
    selected_event_is_tiny_tail = _is_tiny_tail_event(
        member_event_fraction=best_tpr,
        nonmember_event_fraction=best_fpr,
        member_event_count=best_member_count,
        nonmember_event_count=best_nonmember_count,
        member_trials=len(member_scores),
        nonmember_trials=len(nonmember_scores),
        fraction_threshold=tiny_tail_fraction_threshold,
        min_event_count=tiny_tail_min_event_count,
    )
    warning_message = _compose_warning_message(
        report_confidence_supported_lower_bound=report_confidence_supported_lower_bound,
        point_estimate=point_estimate,
        conservative_estimate=conservative_estimate,
        selected_event_is_tiny_tail=selected_event_is_tiny_tail,
        existing_warning=None,
    )

    return EmpiricalEpsilonEstimate(
        epsilon_lower_empirical=conservative_estimate
        if report_confidence_supported_lower_bound
        else point_estimate,
        epsilon_lower_empirical_point_estimate=point_estimate,
        epsilon_lower_empirical_conservative=conservative_estimate,
        empirical_ci_lower=max(0.0, best_lower),
        empirical_ci_upper=max(0.0, best_upper),
        estimation_method="threshold_sweep_member_aligned_member_favoring"
        if require_member_favoring
        else "threshold_sweep_member_aligned",
        delta=delta,
        selected_threshold=threshold_transform(best_threshold),
        selected_event=selected_event_direction,
        selected_event_direction=selected_event_direction,
        selected_tpr=best_tpr,
        selected_fpr=best_fpr,
        member_event_fraction=best_tpr,
        nonmember_event_fraction=best_fpr,
        member_favoring=best_member_favoring,
        no_member_favoring_event_found=False,
        selected_event_is_tiny_tail=selected_event_is_tiny_tail,
        selected_member_event_count=best_member_count,
        selected_nonmember_event_count=best_nonmember_count,
        warning_message=warning_message,
        num_member_samples=len(member_scores),
        num_nonmember_samples=len(nonmember_scores),
    )


def _estimate_legacy_threshold_sweep(
    *,
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    delta: float,
    thresholds: Sequence[float],
    report_confidence_supported_lower_bound: bool,
    tiny_tail_fraction_threshold: float,
    tiny_tail_min_event_count: int,
) -> EmpiricalEpsilonEstimate:
    best_point = -math.inf
    best_lower = 0.0
    best_upper = 0.0
    best_threshold = thresholds[0]
    best_event = "score>=threshold"
    best_tpr = 0.0
    best_fpr = 0.0
    best_member_count = 0
    best_nonmember_count = 0

    for threshold in thresholds:
        member_successes = sum(score >= threshold for score in member_scores)
        nonmember_successes = sum(score >= threshold for score in nonmember_scores)
        candidate = _evaluate_legacy_threshold(
            member_successes=member_successes,
            member_trials=len(member_scores),
            nonmember_successes=nonmember_successes,
            nonmember_trials=len(nonmember_scores),
            delta=delta,
        )
        if candidate["point"] > best_point:
            best_point = candidate["point"]
            best_lower = candidate["lower"]
            best_upper = candidate["upper"]
            best_threshold = threshold
            best_event = candidate["event"]
            best_tpr = candidate["tpr"]
            best_fpr = candidate["fpr"]
            best_member_count = member_successes
            best_nonmember_count = nonmember_successes

    member_event_fraction = best_tpr if best_event == "score>=threshold" else 1.0 - best_tpr
    nonmember_event_fraction = best_fpr if best_event == "score>=threshold" else 1.0 - best_fpr
    selected_member_event_count = best_member_count if best_event == "score>=threshold" else len(member_scores) - best_member_count
    selected_nonmember_event_count = best_nonmember_count if best_event == "score>=threshold" else len(nonmember_scores) - best_nonmember_count
    point_estimate = max(0.0, best_point)
    conservative_estimate = max(0.0, best_lower)
    selected_event_is_tiny_tail = _is_tiny_tail_event(
        member_event_fraction=member_event_fraction,
        nonmember_event_fraction=nonmember_event_fraction,
        member_event_count=selected_member_event_count,
        nonmember_event_count=selected_nonmember_event_count,
        member_trials=len(member_scores),
        nonmember_trials=len(nonmember_scores),
        fraction_threshold=tiny_tail_fraction_threshold,
        min_event_count=tiny_tail_min_event_count,
    )
    warning_message = _compose_warning_message(
        report_confidence_supported_lower_bound=report_confidence_supported_lower_bound,
        point_estimate=point_estimate,
        conservative_estimate=conservative_estimate,
        selected_event_is_tiny_tail=selected_event_is_tiny_tail,
        existing_warning=None,
    )

    return EmpiricalEpsilonEstimate(
        epsilon_lower_empirical=conservative_estimate
        if report_confidence_supported_lower_bound
        else point_estimate,
        epsilon_lower_empirical_point_estimate=point_estimate,
        epsilon_lower_empirical_conservative=conservative_estimate,
        empirical_ci_lower=max(0.0, best_lower),
        empirical_ci_upper=max(0.0, best_upper),
        estimation_method="threshold_sweep_binary_hypothesis_test",
        delta=delta,
        selected_threshold=best_threshold,
        selected_event=best_event,
        selected_event_direction=best_event,
        selected_tpr=best_tpr,
        selected_fpr=best_fpr,
        member_event_fraction=member_event_fraction,
        nonmember_event_fraction=nonmember_event_fraction,
        member_favoring=(member_event_fraction > nonmember_event_fraction),
        no_member_favoring_event_found=False,
        selected_event_is_tiny_tail=selected_event_is_tiny_tail,
        selected_member_event_count=selected_member_event_count,
        selected_nonmember_event_count=selected_nonmember_event_count,
        warning_message=warning_message,
        num_member_samples=len(member_scores),
        num_nonmember_samples=len(nonmember_scores),
    )


def _evaluate_member_aligned_threshold(
    *,
    member_successes: int,
    member_trials: int,
    nonmember_successes: int,
    nonmember_trials: int,
    delta: float,
) -> dict[str, float | str]:
    tpr = member_successes / max(1, member_trials)
    fpr = nonmember_successes / max(1, nonmember_trials)
    tpr_lower, tpr_upper = _wilson_interval(member_successes, member_trials)
    fpr_lower, fpr_upper = _wilson_interval(nonmember_successes, nonmember_trials)

    event_positive_point = _epsilon_candidate(
        numerator_rate=tpr,
        denominator_rate=fpr,
        delta=delta,
        denominator_floor=0.5 / max(1, nonmember_trials),
    )
    event_positive_lower = _epsilon_candidate(
        numerator_rate=tpr_lower,
        denominator_rate=fpr_upper,
        delta=delta,
        denominator_floor=0.5 / max(1, nonmember_trials),
    )
    event_positive_upper = _epsilon_candidate(
        numerator_rate=tpr_upper,
        denominator_rate=fpr_lower,
        delta=delta,
        denominator_floor=0.5 / max(1, nonmember_trials),
    )

    return {
        "point": event_positive_point,
        "lower": event_positive_lower,
        "upper": event_positive_upper,
        "tpr": tpr,
        "fpr": fpr,
        "member_favoring": tpr > fpr,
    }


def _evaluate_legacy_threshold(
    *,
    member_successes: int,
    member_trials: int,
    nonmember_successes: int,
    nonmember_trials: int,
    delta: float,
) -> dict[str, float | str]:
    tpr = member_successes / max(1, member_trials)
    fpr = nonmember_successes / max(1, nonmember_trials)
    tpr_lower, tpr_upper = _wilson_interval(member_successes, member_trials)
    fpr_lower, fpr_upper = _wilson_interval(nonmember_successes, nonmember_trials)

    event_positive_point = _epsilon_candidate(
        numerator_rate=tpr,
        denominator_rate=fpr,
        delta=delta,
        denominator_floor=0.5 / max(1, nonmember_trials),
    )
    event_positive_lower = _epsilon_candidate(
        numerator_rate=tpr_lower,
        denominator_rate=fpr_upper,
        delta=delta,
        denominator_floor=0.5 / max(1, nonmember_trials),
    )
    event_positive_upper = _epsilon_candidate(
        numerator_rate=tpr_upper,
        denominator_rate=fpr_lower,
        delta=delta,
        denominator_floor=0.5 / max(1, nonmember_trials),
    )

    event_negative_point = _epsilon_candidate(
        numerator_rate=1.0 - fpr,
        denominator_rate=1.0 - tpr,
        delta=delta,
        denominator_floor=0.5 / max(1, member_trials),
    )
    event_negative_lower = _epsilon_candidate(
        numerator_rate=1.0 - fpr_upper,
        denominator_rate=1.0 - tpr_lower,
        delta=delta,
        denominator_floor=0.5 / max(1, member_trials),
    )
    event_negative_upper = _epsilon_candidate(
        numerator_rate=1.0 - fpr_lower,
        denominator_rate=1.0 - tpr_upper,
        delta=delta,
        denominator_floor=0.5 / max(1, member_trials),
    )

    if event_positive_point >= event_negative_point:
        return {
            "point": event_positive_point,
            "lower": event_positive_lower,
            "upper": event_positive_upper,
            "event": "score>=threshold",
            "tpr": tpr,
            "fpr": fpr,
        }
    return {
        "point": event_negative_point,
        "lower": event_negative_lower,
        "upper": event_negative_upper,
        "event": "score<threshold",
        "tpr": tpr,
        "fpr": fpr,
    }


def _epsilon_candidate(
    *,
    numerator_rate: float,
    denominator_rate: float,
    delta: float,
    denominator_floor: float,
) -> float:
    numerator = numerator_rate - delta
    if numerator <= 0.0:
        return 0.0
    denominator = max(denominator_rate, denominator_floor)
    if denominator <= 0.0:
        return 0.0
    return max(0.0, math.log(numerator / denominator))


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    phat = successes / trials
    denom = 1.0 + (z**2 / trials)
    center = (phat + (z**2 / (2.0 * trials))) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) / trials) + (z**2 / (4.0 * trials**2)))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _is_tiny_tail_event(
    *,
    member_event_fraction: float,
    nonmember_event_fraction: float,
    member_event_count: int,
    nonmember_event_count: int,
    member_trials: int,
    nonmember_trials: int,
    fraction_threshold: float,
    min_event_count: int,
) -> bool:
    max_fraction = max(member_event_fraction, nonmember_event_fraction)
    return (
        max_fraction <= fraction_threshold
        or member_event_count < min_event_count
        or nonmember_event_count < min_event_count
        or member_event_count == member_trials
        or nonmember_event_count == nonmember_trials
    )


def _compose_warning_message(
    *,
    report_confidence_supported_lower_bound: bool,
    point_estimate: float,
    conservative_estimate: float,
    selected_event_is_tiny_tail: bool,
    existing_warning: str | None,
) -> str | None:
    warnings: list[str] = []
    if existing_warning:
        warnings.append(existing_warning)
    if selected_event_is_tiny_tail:
        warnings.append("Selected threshold lies in a sparse tail region.")
    if report_confidence_supported_lower_bound and conservative_estimate < point_estimate:
        warnings.append(
            "Reported empirical lower bound uses the confidence-supported lower side rather than the optimistic point estimate."
        )
    if not warnings:
        return None
    return " ".join(warnings)
