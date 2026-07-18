from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrivacyTightnessMetrics:
    privacy_loss_gap: float
    tightness_ratio: float | None
    valid_empirical_lower_bound: bool
    sanity_warning: str | None = None


def compute_privacy_loss_gap(epsilon_upper_theory: float, epsilon_lower_empirical: float) -> float:
    return epsilon_upper_theory - epsilon_lower_empirical


def compute_tightness_ratio(
    epsilon_upper_theory: float,
    epsilon_lower_empirical: float,
) -> float | None:
    if epsilon_upper_theory <= 0.0:
        return None
    return epsilon_lower_empirical / epsilon_upper_theory


def empirical_lower_bound_is_valid(
    epsilon_upper_theory: float,
    epsilon_lower_empirical: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    return epsilon_lower_empirical <= epsilon_upper_theory + tolerance


def empirical_lower_bound_sanity_warning(
    epsilon_upper_theory: float,
    epsilon_lower_empirical: float,
) -> str | None:
    if empirical_lower_bound_is_valid(epsilon_upper_theory, epsilon_lower_empirical):
        return None
    return (
        "Empirical lower bound exceeds the theoretical upper bound; treat this "
        "run as a diagnostic warning, not a validated privacy lower-bound claim."
    )


def compute_privacy_tightness_metrics(
    epsilon_upper_theory: float,
    epsilon_lower_empirical: float,
) -> PrivacyTightnessMetrics:
    return PrivacyTightnessMetrics(
        privacy_loss_gap=compute_privacy_loss_gap(epsilon_upper_theory, epsilon_lower_empirical),
        tightness_ratio=compute_tightness_ratio(epsilon_upper_theory, epsilon_lower_empirical),
        valid_empirical_lower_bound=empirical_lower_bound_is_valid(
            epsilon_upper_theory,
            epsilon_lower_empirical,
        ),
        sanity_warning=empirical_lower_bound_sanity_warning(
            epsilon_upper_theory,
            epsilon_lower_empirical,
        ),
    )
