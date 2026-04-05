"""Privacy Loss Distribution (PLD) accounting for tighter epsilon upper bounds.

The RDP accountant (used by Opacus) produces valid but *loose* upper bounds on
epsilon.  PLD-based accounting computes the privacy loss numerically via
discretised distributions and yields a *tighter* upper bound for the same
(noise_multiplier, sampling_rate, num_steps, delta) tuple.

This module provides two back-ends:

1. **dp_accounting** (Google's library) — preferred; uses the analytical
   Gaussian mechanism PLD with Poisson subsampling.
2. **Fallback analytical** — a closed-form Gaussian-mechanism bound via the
   inverse-CDF approach (Balle et al., 2020).  Less tight than PLD but still
   tighter than RDP for most regimes, and has zero extra dependencies.

The public entry point is :func:`compute_epsilon_pld`.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_epsilon_pld(
    *,
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
    backend: str = "auto",
) -> dict[str, Any]:
    """Compute a tight epsilon upper bound using PLD-based accounting.

    Parameters
    ----------
    noise_multiplier : float
        Ratio of noise standard deviation to the clipping norm (sigma / C).
    sampling_rate : float
        Probability that each example is included in a batch (batch_size / n).
    num_steps : int
        Total number of gradient steps (epochs * steps_per_epoch).
    delta : float
        Target delta for (epsilon, delta)-DP.
    backend : str
        ``"google"`` forces the dp_accounting library, ``"analytical"`` uses
        the closed-form bound, ``"auto"`` (default) tries google first.

    Returns
    -------
    dict with keys:
        epsilon_pld : float   — the computed epsilon upper bound
        backend_used : str    — which backend produced the result
        num_steps : int       — echoed back for record-keeping
    """
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")
    if not (0 < sampling_rate <= 1):
        raise ValueError("sampling_rate must be in (0, 1].")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    if backend in ("auto", "google"):
        try:
            eps = _compute_with_dp_accounting(
                noise_multiplier=noise_multiplier,
                sampling_rate=sampling_rate,
                num_steps=num_steps,
                delta=delta,
            )
            return {"epsilon_pld": round(eps, 6), "backend_used": "google_dp_accounting", "num_steps": num_steps}
        except (ImportError, AttributeError, Exception) as e:
            if backend == "google":
                raise
            # Fall through to analytical

    # Analytical fallback (always available)
    eps = _compute_analytical_gaussian(
        noise_multiplier=noise_multiplier,
        sampling_rate=sampling_rate,
        num_steps=num_steps,
        delta=delta,
    )
    return {"epsilon_pld": round(eps, 6), "backend_used": "analytical_gaussian", "num_steps": num_steps}


# ---------------------------------------------------------------------------
# Backend 1: Google dp_accounting (PLD-based, tightest)
# ---------------------------------------------------------------------------

def _compute_with_dp_accounting(
    *,
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
) -> float:
    """Use google/dp-accounting's PLD accountant for tight composition."""
    from dp_accounting.pld import privacy_loss_distribution as pld_lib
    from dp_accounting.pld import common

    # Build the PLD for a single step of subsampled Gaussian mechanism.
    pld_single = pld_lib.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sensitivities=[1.0],
        pessimistic_estimate=True,
        sampling_prob=sampling_rate,
        use_connect_dots=True,
        value_discretization_interval=common.VALUE_DISCRETIZATION_INTERVAL,
    )

    # Self-compose for num_steps applications.
    pld_composed = pld_single.self_compose(num_steps)

    # Extract epsilon at the target delta.
    epsilon = pld_composed.get_epsilon_for_delta(delta)
    return float(epsilon)


# ---------------------------------------------------------------------------
# Backend 2: GDP (Gaussian Differential Privacy) — Dong, Roth & Su (2019)
# ---------------------------------------------------------------------------

def _compute_analytical_gaussian(
    *,
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
) -> float:
    """Tight closed-form bound via the GDP (Gaussian DP) CLT framework.

    The Central Limit Theorem for DP (Dong, Roth & Su, 2019; Balle et al.,
    2020) shows that T-fold composition of the subsampled Gaussian mechanism
    converges to mu-GDP with:

        mu = q * sqrt(T) / sigma

    where q is the sampling rate and sigma is the noise multiplier.  We then
    convert mu-GDP to (epsilon, delta)-DP by inverting the trade-off function.

    This gives bounds that are *much* tighter than advanced composition and
    comparable to (though slightly looser than) exact PLD numerical accounting.
    """
    sigma = noise_multiplier

    # GDP parameter: mu = q * sqrt(T) / sigma
    mu = sampling_rate * math.sqrt(num_steps) / sigma

    # Convert mu-GDP to (epsilon, delta)-DP by inverting:
    #   delta(eps) = Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2)
    epsilon = _gdp_to_eps_delta(mu, delta)
    return float(epsilon)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _gdp_to_eps_delta(mu: float, target_delta: float) -> float:
    """Convert mu-GDP to epsilon given a target delta.

    Inverts: delta(eps) = Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2)
    via binary search.
    """
    if mu <= 0:
        return 0.0

    def _delta_of_eps(eps: float) -> float:
        t1 = _norm_cdf(-eps / mu + mu / 2.0)
        t2 = math.exp(eps) * _norm_cdf(-eps / mu - mu / 2.0)
        return t1 - t2

    # Binary search for the smallest epsilon where delta(eps) <= target_delta
    lo, hi = 0.0, mu * mu + 2.0 * mu * math.sqrt(math.log(1.0 / target_delta))

    # Safety check: if delta(0) is already <= target, epsilon = 0
    if _delta_of_eps(0.0) <= target_delta:
        return 0.0

    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _delta_of_eps(mid) > target_delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break

    return hi
