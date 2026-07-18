"""Privacy Loss Distribution (PLD) accounting for tighter epsilon upper bounds.

The RDP accountant (used by Opacus) produces valid but *loose* upper bounds on
epsilon.  PLD-based accounting computes the privacy loss numerically via
discretised distributions and yields a *tighter* upper bound for the same
(noise_multiplier, sampling_rate, num_steps, delta) tuple.

This module provides two back-ends:

1. **dp_accounting** (Google's library) — preferred; uses the analytical
   Gaussian mechanism PLD with Poisson subsampling.
2. **Fallback analytical** — a GDP central-limit approximation (mu = q*sqrt(T)/sigma).
   WARNING: this is an *asymptotic approximation*, NOT a valid tight upper bound.
   In the low-noise / low-composition regime (small sigma, few steps) it is
   *anti-conservative*: it can sit far BELOW the true PLD and even below what any
   valid accountant would report (e.g. at sigma=0.5, ~211 steps it returns ~0.47
   while the true PLD is ~5.11). It converges to true PLD only at high sigma.
   Never report it as an upper bound; use it only as a rough diagnostic. Mislabeling
   this fallback as "PLD" produced the invalid "93% accounting share" artifact
   (see research/VALIDATION_2026-07-06.md).

The public entry point is :func:`compute_epsilon_pld`.
"""

from __future__ import annotations

import math
from typing import Any

_VALUE_DISCRETIZATION_INTERVAL = 1e-4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_epsilon_pld(
    *,
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
    backend: str = "google",
) -> dict[str, Any]:
    """Compute a tight epsilon upper bound using PLD-based accounting.

    ``backend`` defaults to ``"google"`` (hard default as of 2026-07-06 fix #5): if the
    ``dp_accounting`` PLD library is unavailable this RAISES rather than silently
    returning the anti-conservative GDP-CLT analytical approximation. Use
    ``backend="auto"`` to opt into the (warned) fallback, or ``backend="analytical"``
    to force the GDP-CLT approximation for diagnostic diffs only. Mislabeling the
    analytical fallback as "PLD" is exactly what invalidated Finding 1 (see
    research/VALIDATION_2026-07-06.md).

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
            # Fall through to analytical -- but this is anti-conservative at low
            # sigma and must NEVER be silently recorded as "PLD". Warn loudly.
            import warnings
            warnings.warn(
                "compute_epsilon_pld: dp_accounting PLD backend unavailable "
                f"({type(e).__name__}: {e}); falling back to the GDP-CLT analytical "
                "approximation, which is NOT a valid tight upper bound at low sigma "
                "(anti-conservative). Do not report this value as PLD. "
                "See research/VALIDATION_2026-07-06.md.",
                RuntimeWarning,
                stacklevel=2,
            )

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

    # Build the PLD for a single step of subsampled Gaussian mechanism.
    kwargs = {
        "standard_deviation": noise_multiplier,
        "pessimistic_estimate": True,
        "sampling_prob": sampling_rate,
        "use_connect_dots": True,
        "value_discretization_interval": _VALUE_DISCRETIZATION_INTERVAL,
    }
    try:
        pld_single = pld_lib.from_gaussian_mechanism(
            sensitivity=1.0,
            **kwargs,
        )
    except TypeError:
        # Older dp-accounting builds used a plural sensitivities argument.
        pld_single = pld_lib.from_gaussian_mechanism(
            sensitivities=[1.0],
            **kwargs,
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
