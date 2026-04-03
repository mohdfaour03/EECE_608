"""GDP (Gaussian Differential Privacy) estimation for tighter lower bounds.

Instead of the threshold sweep + Wilson CI approach in empirical.py, this module:
1. Computes the empirical AUC between member and non-member scores
2. Estimates the GDP mu parameter from AUC: mu = sqrt(2) * Phi^{-1}(AUC)
3. Converts mu to (epsilon, delta)-DP via the standard GDP-to-DP conversion
4. Uses bootstrap confidence intervals on mu for conservative bounds

The AUC-based estimator avoids threshold selection bias (picking the "best"
threshold inflates mu by cherry-picking noise). Under GDP(mu), the ROC curve
is fully determined by mu, and AUC = Phi(mu / sqrt(2)), giving a clean
single-parameter estimate from the entire score distribution.

Reference: Dong, Roth & Su "Gaussian Differential Privacy" (2019)
Reference: Balle et al. "Hypothesis Testing Interpretations..." (2020)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class GDPEstimate:
    """Result of GDP-based privacy estimation."""
    epsilon_lower: float
    mu_lower: float
    mu_point: float
    mu_ci_lower: float
    delta: float
    method: str
    auc: float = 0.0
    n_member: int = 0
    n_nonmember: int = 0
    warning: str | None = None


def estimate_epsilon_gdp(
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    delta: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> GDPEstimate:
    """Estimate epsilon lower bound using GDP framework.

    Computes the empirical AUC between member and non-member scores,
    estimates mu via mu = sqrt(2) * Phi^{-1}(AUC), then converts to epsilon.
    Uses bootstrap for conservative confidence intervals.

    Parameters
    ----------
    member_scores : array-like
        Scores for members (higher = more likely member).
    nonmember_scores : array-like
        Scores for non-members.
    delta : float
        Privacy parameter delta.
    n_bootstrap : int
        Number of bootstrap resamples for CI.
    ci_level : float
        Confidence level for bootstrap CI.
    """
    m = np.array(member_scores, dtype=np.float64)
    n = np.array(nonmember_scores, dtype=np.float64)

    if len(m) < 5 or len(n) < 5:
        return GDPEstimate(
            epsilon_lower=0.0, mu_lower=0.0, mu_point=0.0,
            mu_ci_lower=0.0, delta=delta, method="gdp_auc",
            n_member=len(m), n_nonmember=len(n),
            warning="Too few samples")

    # Point estimate of mu from AUC
    auc = _compute_auc(m, n)
    mu_point = _auc_to_mu(auc)

    # Bootstrap CI on mu
    rng = np.random.RandomState(42)
    mu_boots = []
    for _ in range(n_bootstrap):
        m_boot = rng.choice(m, size=len(m), replace=True)
        n_boot = rng.choice(n, size=len(n), replace=True)
        auc_b = _compute_auc(m_boot, n_boot)
        mu_boots.append(_auc_to_mu(auc_b))

    alpha = 1.0 - ci_level
    mu_ci_lower = float(np.percentile(mu_boots, 100 * alpha / 2))
    mu_ci_lower = max(0.0, mu_ci_lower)

    # Convert mu to epsilon (use conservative CI lower bound)
    eps_conservative = _mu_to_epsilon(mu_ci_lower, delta) if mu_ci_lower > 0 else 0.0

    return GDPEstimate(
        epsilon_lower=max(0.0, eps_conservative),
        mu_lower=mu_ci_lower,
        mu_point=mu_point,
        mu_ci_lower=mu_ci_lower,
        delta=delta,
        method="gdp_auc_bootstrap",
        auc=auc,
        n_member=len(m),
        n_nonmember=len(n),
    )


def _compute_auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    """Compute AUC via the Mann-Whitney U statistic.

    AUC = P(member_score > nonmember_score) + 0.5 * P(member_score == nonmember_score)
    This is equivalent to U / (n_m * n_n) where U is the Mann-Whitney U statistic.
    """
    n_m = len(member)
    n_n = len(nonmember)
    # Vectorized: count how many nonmember scores each member score beats
    # For large arrays, use sorted-rank approach for O(n log n)
    all_scores = np.concatenate([member, nonmember])
    labels = np.concatenate([np.ones(n_m), np.zeros(n_n)])

    # Sort by score descending
    order = np.argsort(-all_scores)
    sorted_labels = labels[order]

    # AUC = (sum of ranks of positives - n_m*(n_m+1)/2) / (n_m * n_n)
    ranks = np.arange(1, len(all_scores) + 1, dtype=np.float64)
    # Handle ties: assign average rank
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[i:j])
            ranks[i:j] = avg_rank
        i = j

    pos_rank_sum = ranks[sorted_labels == 1].sum()
    auc = (pos_rank_sum - n_m * (n_m + 1) / 2) / (n_m * n_n)
    return float(auc)


def _auc_to_mu(auc: float) -> float:
    """Convert AUC to GDP mu parameter.

    Under GDP(mu): AUC = Phi(mu / sqrt(2))
    So mu = sqrt(2) * Phi^{-1}(AUC)

    Returns 0.0 if AUC <= 0.5 (no discrimination).
    """
    from scipy.stats import norm

    if auc <= 0.5:
        return 0.0
    # Clip to avoid Phi^{-1}(1) = inf
    auc_clipped = min(auc, 1.0 - 1e-10)
    mu = math.sqrt(2) * norm.ppf(auc_clipped)
    return max(0.0, mu)


def _mu_to_epsilon(mu: float, delta: float) -> float:
    """Convert GDP mu to (epsilon, delta)-DP via binary search."""
    if mu <= 0:
        return 0.0

    lo, hi = 0.0, mu * mu + 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        d = _gdp_delta(mid, mu)
        if d > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break
    return lo


def _gdp_delta(epsilon: float, mu: float) -> float:
    """Compute delta for given epsilon under GDP(mu)."""
    from scipy.stats import norm
    if mu <= 0:
        return 0.0
    term1 = norm.cdf(-epsilon / mu + mu / 2)
    term2 = math.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2)
    return max(0.0, term1 - term2)
