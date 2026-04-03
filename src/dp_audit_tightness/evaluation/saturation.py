from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from dp_audit_tightness.config import SaturationConfig


@dataclass(slots=True)
class SaturationDecision:
    saturation_detected: bool
    reason: str | None
    absolute_improvement: float | None
    relative_improvement: float | None
    tightness_ratio_change: float | None
    confidence_interval_overlap: bool | None


def detect_saturation(
    history: Sequence[Mapping[str, float | None]],
    config: SaturationConfig,
) -> SaturationDecision:
    """Detect whether stronger auditors have stopped materially improving the lower bound.

    The intended use is to compare successive auditor variants ordered by strength.
    Saturation is treated as an interpretable result, not as an execution failure.
    """

    if len(history) < 2:
        return SaturationDecision(
            saturation_detected=False,
            reason=None,
            absolute_improvement=None,
            relative_improvement=None,
            tightness_ratio_change=None,
            confidence_interval_overlap=None,
        )

    previous = history[-2]
    current = history[-1]
    previous_lower = float(previous["epsilon_lower_empirical"] or 0.0)
    current_lower = float(current["epsilon_lower_empirical"] or 0.0)
    absolute_improvement = current_lower - previous_lower
    relative_improvement = None
    if previous_lower > 0.0:
        relative_improvement = absolute_improvement / previous_lower

    previous_ratio = previous.get("tightness_ratio")
    current_ratio = current.get("tightness_ratio")
    tightness_ratio_change = None
    if previous_ratio is not None and current_ratio is not None:
        tightness_ratio_change = float(current_ratio) - float(previous_ratio)

    confidence_interval_overlap = None
    if config.use_confidence_interval_overlap:
        confidence_interval_overlap = _intervals_overlap(
            previous.get("empirical_ci_lower"),
            previous.get("empirical_ci_upper"),
            current.get("empirical_ci_lower"),
            current.get("empirical_ci_upper"),
        )

    absolute_small = absolute_improvement < config.min_absolute_improvement
    relative_small = (
        relative_improvement is None or relative_improvement < config.min_relative_improvement
    )
    tightness_stable = (
        tightness_ratio_change is None
        or abs(tightness_ratio_change) < config.tightness_ratio_tolerance
    )

    improvement_small = absolute_small and relative_small
    ci_supports_saturation = True
    if config.use_confidence_interval_overlap and confidence_interval_overlap is not None:
        ci_supports_saturation = confidence_interval_overlap
    saturation_detected = improvement_small and tightness_stable and ci_supports_saturation

    reasons: list[str] = []
    if absolute_small:
        reasons.append("absolute improvement below threshold")
    if relative_small:
        reasons.append("relative improvement below threshold")
    if tightness_stable:
        reasons.append("tightness ratio stable")
    if confidence_interval_overlap:
        reasons.append("successive confidence intervals overlap")

    return SaturationDecision(
        saturation_detected=saturation_detected,
        reason="; ".join(reasons) if reasons else None,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        tightness_ratio_change=tightness_ratio_change,
        confidence_interval_overlap=confidence_interval_overlap,
    )


def _intervals_overlap(
    lower_a: float | None,
    upper_a: float | None,
    lower_b: float | None,
    upper_b: float | None,
) -> bool | None:
    values = [lower_a, upper_a, lower_b, upper_b]
    if any(value is None for value in values):
        return None
    return not (float(upper_a) < float(lower_b) or float(upper_b) < float(lower_a))
