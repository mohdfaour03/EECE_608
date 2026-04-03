from __future__ import annotations

from typing import Any


def compute_theoretical_upper_bound(*, privacy_engine: Any, delta: float) -> float:
    """Return the theoretical upper bound on privacy loss from the configured accountant.

    This function should only return a quantity with the semantics of a guaranteed upper
    bound under the accountant and mechanism actually used in training.
    """

    accountant = getattr(privacy_engine, "accountant", privacy_engine)
    if not hasattr(accountant, "get_epsilon"):
        raise TypeError("privacy_engine/accountant must expose get_epsilon(delta).")
    epsilon_upper_theory = float(accountant.get_epsilon(delta))
    if epsilon_upper_theory < 0.0:
        raise ValueError("Theoretical upper bound on privacy loss must be non-negative.")
    return epsilon_upper_theory

