from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrivacyTightnessMetrics:
    privacy_loss_gap: float
    tightness_ratio: float | None


def compute_privacy_loss_gap(epsilon_upper_theory: float, epsilon_lower_empirical: float) -> float:
    return epsilon_upper_theory - epsilon_lower_empirical


def compute_tightness_ratio(
    epsilon_upper_theory: float,
    epsilon_lower_empirical: float,
) -> float | None:
    if epsilon_upper_theory <= 0.0:
        return None
    return epsilon_lower_empirical / epsilon_upper_theory


def compute_privacy_tightness_metrics(
    epsilon_upper_theory: float,
    epsilon_lower_empirical: float,
) -> PrivacyTightnessMetrics:
    return PrivacyTightnessMetrics(
        privacy_loss_gap=compute_privacy_loss_gap(epsilon_upper_theory, epsilon_lower_empirical),
        tightness_ratio=compute_tightness_ratio(epsilon_upper_theory, epsilon_lower_empirical),
    )

