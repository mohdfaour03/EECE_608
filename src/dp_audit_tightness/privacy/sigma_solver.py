"""Solve for the noise multiplier sigma that achieves a target epsilon.

Rationale (professor feedback, 2026-07-08 meeting): per-epsilon hyperparameter
optimization requires holding the *privacy guarantee* fixed while the search
varies epsilon-affecting hyperparameters (batch size -> sampling rate; epochs ->
num_steps). For each trial configuration we therefore re-solve sigma so that
eps_PLD(sigma, q, T, delta) == target_epsilon, and the tuner compares utility
at a genuinely constant privacy level.

Design notes
------------
- The accountant is INJECTED as a callable so this module has no hard
  dependency on dp_accounting (CPU sandboxes without it can still unit-test the
  solver against synthetic monotone accountants). The default accountant wraps
  ``compute_epsilon_pld`` with the hard "google" backend, consistent with
  Fix 5 (HANDOFF_OPUS_FIXES_2026-07-06): no silent analytical fallback.
- epsilon is strictly decreasing in sigma for the subsampled Gaussian
  mechanism at fixed (q, T, delta); bisection is therefore exact up to
  tolerance. We still verify the bracket and monotone direction defensively
  and raise instead of extrapolating.
- Everything is deterministic. No randomness anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "SigmaSolution",
    "SigmaSolverError",
    "solve_sigma_for_epsilon",
    "make_pld_accountant",
]

# accountant(sigma, sampling_rate, num_steps, delta) -> epsilon (float)
Accountant = Callable[[float, float, int, float], float]


class SigmaSolverError(RuntimeError):
    """Raised when the solver cannot bracket or converge on a valid sigma."""


@dataclass(slots=True)
class SigmaSolution:
    """Record of a solved noise multiplier (kept JSON-friendly for snapshots)."""

    sigma: float
    achieved_epsilon: float
    target_epsilon: float
    sampling_rate: float
    num_steps: int
    delta: float
    iterations: int
    epsilon_abs_error: float
    bracket_low: float
    bracket_high: float

    def to_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "achieved_epsilon": self.achieved_epsilon,
            "target_epsilon": self.target_epsilon,
            "sampling_rate": self.sampling_rate,
            "num_steps": self.num_steps,
            "delta": self.delta,
            "iterations": self.iterations,
            "epsilon_abs_error": self.epsilon_abs_error,
            "bracket_low": self.bracket_low,
            "bracket_high": self.bracket_high,
        }


def make_pld_accountant() -> Accountant:
    """Default accountant: true PLD via the google backend (raises if missing).

    Imported lazily so that importing this module never requires dp_accounting.
    """

    from dp_audit_tightness.privacy.pld_accounting import compute_epsilon_pld

    def _accountant(sigma: float, sampling_rate: float, num_steps: int, delta: float) -> float:
        result = compute_epsilon_pld(
            noise_multiplier=sigma,
            sampling_rate=sampling_rate,
            num_steps=num_steps,
            delta=delta,
            backend="google",
        )
        return float(result["epsilon_pld"])

    return _accountant


def solve_sigma_for_epsilon(
    target_epsilon: float,
    sampling_rate: float,
    num_steps: int,
    delta: float,
    accountant: Accountant | None = None,
    sigma_low: float = 0.3,
    sigma_high: float = 64.0,
    epsilon_tolerance: float = 1e-3,
    max_iterations: int = 80,
) -> SigmaSolution:
    """Bisection solve for sigma such that accountant(sigma, q, T, delta) ~= target.

    Parameters mirror the accounting call; ``sigma_low``/``sigma_high`` give the
    initial bracket, which is widened geometrically (up to hard caps) if the
    target lies outside it. Raises :class:`SigmaSolverError` when the target is
    unreachable (e.g. absurdly small epsilon that would need sigma > 512, or a
    non-monotone/buggy accountant).
    """

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive.")
    if not (0 < sampling_rate <= 1):
        raise ValueError("sampling_rate must be in (0, 1].")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1).")
    if accountant is None:
        accountant = make_pld_accountant()

    lo, hi = float(sigma_low), float(sigma_high)
    if not (0 < lo < hi):
        raise ValueError("Require 0 < sigma_low < sigma_high.")

    # eps(sigma) is decreasing: eps(lo) is the largest value in the bracket.
    eps_lo = accountant(lo, sampling_rate, num_steps, delta)   # big epsilon
    eps_hi = accountant(hi, sampling_rate, num_steps, delta)   # small epsilon
    if eps_lo < eps_hi:
        raise SigmaSolverError(
            f"Accountant is not decreasing in sigma (eps({lo})={eps_lo:.4f} < "
            f"eps({hi})={eps_hi:.4f}); refusing to solve."
        )

    # Widen the bracket if the target falls outside it.
    widen = 0
    while eps_lo < target_epsilon and lo > 0.05 and widen < 8:
        lo = max(0.05, lo / 2.0)
        eps_lo = accountant(lo, sampling_rate, num_steps, delta)
        widen += 1
    while eps_hi > target_epsilon and hi < 512.0 and widen < 16:
        hi = min(512.0, hi * 2.0)
        eps_hi = accountant(hi, sampling_rate, num_steps, delta)
        widen += 1
    if not (eps_hi <= target_epsilon <= eps_lo):
        raise SigmaSolverError(
            f"target_epsilon={target_epsilon} outside achievable range "
            f"[{eps_hi:.5f}, {eps_lo:.5f}] for sigma in [{lo}, {hi}] "
            f"(q={sampling_rate}, T={num_steps}, delta={delta})."
        )

    bracket_low, bracket_high = lo, hi
    mid, eps_mid = lo, eps_lo
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        mid = 0.5 * (lo + hi)
        eps_mid = accountant(mid, sampling_rate, num_steps, delta)
        if abs(eps_mid - target_epsilon) <= epsilon_tolerance:
            break
        if eps_mid > target_epsilon:
            lo = mid   # too much epsilon -> need more noise
        else:
            hi = mid   # too little epsilon -> less noise
    else:
        raise SigmaSolverError(
            f"No convergence after {max_iterations} iterations "
            f"(best |eps-target|={abs(eps_mid - target_epsilon):.2e}, tol={epsilon_tolerance})."
        )

    return SigmaSolution(
        sigma=float(mid),
        achieved_epsilon=float(eps_mid),
        target_epsilon=float(target_epsilon),
        sampling_rate=float(sampling_rate),
        num_steps=int(num_steps),
        delta=float(delta),
        iterations=iterations,
        epsilon_abs_error=float(abs(eps_mid - target_epsilon)),
        bracket_low=float(bracket_low),
        bracket_high=float(bracket_high),
    )
