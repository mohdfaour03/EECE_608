"""Gap decomposition: attribute the tightness gap to its sources.

The total gap between epsilon_upper_theory and epsilon_lower_empirical
can be decomposed into attributable components:

  total_gap = accounting_gap + threat_model_gap + attack_gap + residual

Where:
    accounting_gap   = eps_upper_rdp - eps_upper_pld
        How much of the gap comes from using a looser privacy accountant.

    threat_model_gap = eps_lower_canary - eps_lower_passive
        How much the evaluator-controlled threat model recovers beyond
        the passive threat model.

    attack_gap       = eps_lower_best_possible - eps_lower_best_achieved
        How much stronger attacks could close the remaining gap.
        (Only computable when a stronger attack is available.)

    residual         = eps_upper_pld - eps_lower_canary
        The gap that cannot be attributed to accounting or threat model.
        Captures: worst-case vs. actual data, last-iterate advantage,
        subsampling mismatch, and fundamental limits of the auditor.

This module computes the decomposition from available data and flags
which components are available vs. estimated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class GapDecomposition:
    """Structured decomposition of the tightness gap for one experiment."""

    # Identifiers
    training_run_id: str
    dataset: str
    model: str

    # Upper bounds
    epsilon_upper_rdp: float
    epsilon_upper_pld: float | None  # None if PLD accounting not available

    # Lower bounds
    epsilon_lower_canary: float | None
    epsilon_lower_passive: float | None

    # Decomposition components
    total_gap: float                  # eps_upper_rdp - best_lower
    accounting_gap: float | None      # eps_upper_rdp - eps_upper_pld
    threat_model_gap: float | None    # eps_lower_canary - eps_lower_passive
    residual_gap: float | None        # tightest_upper - eps_lower_canary

    # Ratios (for cross-experiment comparison)
    tightness_ratio_canary: float | None
    tightness_ratio_passive: float | None

    # Metadata
    components_available: list[str]   # Which decomposition pieces are computed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decompose_gap(
    *,
    training_run_id: str,
    dataset: str,
    model: str,
    epsilon_upper_rdp: float,
    epsilon_upper_pld: float | None = None,
    epsilon_lower_canary: float | None = None,
    epsilon_lower_passive: float | None = None,
) -> GapDecomposition:
    """Compute the gap decomposition from available measurements.

    Parameters
    ----------
    epsilon_upper_rdp : float
        Theoretical upper bound from RDP accountant (always available).
    epsilon_upper_pld : float or None
        Theoretical upper bound from PLD accountant (tighter, optional).
    epsilon_lower_canary : float or None
        Best empirical lower bound from canary auditing.
    epsilon_lower_passive : float or None
        Best empirical lower bound from passive auditing.

    Returns
    -------
    GapDecomposition
        Structured decomposition with all computable components.
    """
    components: list[str] = []

    # Accounting gap: how much looser is RDP vs PLD?
    accounting_gap = None
    if epsilon_upper_pld is not None:
        accounting_gap = round(epsilon_upper_rdp - epsilon_upper_pld, 6)
        components.append("accounting_gap")

    # Threat model gap: how much more does canary recover vs passive?
    threat_model_gap = None
    if epsilon_lower_canary is not None and epsilon_lower_passive is not None:
        threat_model_gap = round(epsilon_lower_canary - epsilon_lower_passive, 6)
        components.append("threat_model_gap")

    # Best available lower bound
    lowers = [v for v in (epsilon_lower_canary, epsilon_lower_passive) if v is not None]
    best_lower = max(lowers) if lowers else 0.0

    # Total gap (always computable)
    total_gap = round(epsilon_upper_rdp - best_lower, 6)
    components.append("total_gap")

    # Residual: tightest upper - best canary lower
    residual_gap = None
    tightest_upper = epsilon_upper_pld if epsilon_upper_pld is not None else epsilon_upper_rdp
    if epsilon_lower_canary is not None:
        residual_gap = round(tightest_upper - epsilon_lower_canary, 6)
        components.append("residual_gap")

    # Tightness ratios
    tightness_canary = None
    tightness_passive = None
    if epsilon_upper_rdp > 0:
        if epsilon_lower_canary is not None:
            tightness_canary = round(epsilon_lower_canary / epsilon_upper_rdp, 6)
        if epsilon_lower_passive is not None:
            tightness_passive = round(epsilon_lower_passive / epsilon_upper_rdp, 6)

    return GapDecomposition(
        training_run_id=training_run_id,
        dataset=dataset,
        model=model,
        epsilon_upper_rdp=epsilon_upper_rdp,
        epsilon_upper_pld=epsilon_upper_pld,
        epsilon_lower_canary=epsilon_lower_canary,
        epsilon_lower_passive=epsilon_lower_passive,
        total_gap=total_gap,
        accounting_gap=accounting_gap,
        threat_model_gap=threat_model_gap,
        residual_gap=residual_gap,
        tightness_ratio_canary=tightness_canary,
        tightness_ratio_passive=tightness_passive,
        components_available=components,
    )
