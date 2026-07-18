from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.privacy.sigma_solver import (
    SigmaSolverError,
    make_pld_accountant,
    solve_sigma_for_epsilon,
)


def synthetic_accountant(sigma: float, sampling_rate: float, num_steps: int, delta: float) -> float:
    """Strictly decreasing in sigma, increasing in q and T — GDP-CLT-shaped.

    Used ONLY as a test double so the solver is unit-testable without
    dp_accounting. Shape: eps ~ mu*(mu/2 + z) with mu = q*sqrt(T)/sigma.
    """

    mu = sampling_rate * math.sqrt(num_steps) / sigma
    return mu * (mu / 2.0 + 1.2815515655446004)  # z_{0.9}; arbitrary fixed constant


class SigmaSolverTests(unittest.TestCase):
    def test_default_accountant_reads_epsilon_pld_result_key(self) -> None:
        with patch(
            "dp_audit_tightness.privacy.pld_accounting.compute_epsilon_pld",
            return_value={
                "epsilon_pld": 1.234,
                "backend_used": "google_dp_accounting",
                "num_steps": 10,
            },
        ):
            accountant = make_pld_accountant()
            self.assertEqual(accountant(1.0, 0.1, 10, 1e-5), 1.234)

    # q chosen so the synthetic accountant's reachable range comfortably covers
    # the full target grid 0.3..8.0 within the solver's sigma bracket.
    kwargs = dict(sampling_rate=0.1, num_steps=211, delta=1e-5,
                  accountant=synthetic_accountant)

    def test_roundtrip_hits_target_within_tolerance(self) -> None:
        for target in (0.3, 0.5, 1.0, 2.0, 4.0, 8.0):
            solution = solve_sigma_for_epsilon(target_epsilon=target, **self.kwargs)
            self.assertLessEqual(solution.epsilon_abs_error, 1e-3)
            recomputed = synthetic_accountant(
                solution.sigma, solution.sampling_rate, solution.num_steps, solution.delta
            )
            self.assertAlmostEqual(recomputed, solution.achieved_epsilon, places=12)

    def test_sigma_decreases_as_target_epsilon_grows(self) -> None:
        sigmas = [
            solve_sigma_for_epsilon(target_epsilon=t, **self.kwargs).sigma
            for t in (0.3, 0.5, 1.0, 2.0, 4.0, 8.0)
        ]
        self.assertEqual(sigmas, sorted(sigmas, reverse=True))

    def test_determinism(self) -> None:
        a = solve_sigma_for_epsilon(target_epsilon=1.0, **self.kwargs)
        b = solve_sigma_for_epsilon(target_epsilon=1.0, **self.kwargs)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_unreachable_target_raises_instead_of_extrapolating(self) -> None:
        with self.assertRaises(SigmaSolverError):
            solve_sigma_for_epsilon(target_epsilon=1e-9, **self.kwargs)

    def test_non_monotone_accountant_is_rejected(self) -> None:
        def increasing(sigma: float, q: float, t: int, d: float) -> float:
            return sigma

        with self.assertRaises(SigmaSolverError):
            solve_sigma_for_epsilon(
                target_epsilon=1.0, sampling_rate=0.01, num_steps=10, delta=1e-5,
                accountant=increasing,
            )

    def test_input_validation(self) -> None:
        for bad in (dict(target_epsilon=0.0), dict(sampling_rate=0.0), dict(num_steps=0), dict(delta=1.0)):
            params = dict(target_epsilon=1.0, sampling_rate=0.01, num_steps=10, delta=1e-5,
                          accountant=synthetic_accountant)
            params.update(bad)
            with self.assertRaises(ValueError):
                solve_sigma_for_epsilon(**params)


class HpoConfigTests(unittest.TestCase):
    def test_hpo_yaml_configs_parse(self) -> None:
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "tune_hyperparams", root / "experiments" / "tune_hyperparams.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["tune_hyperparams"] = module   # required for slots dataclasses
        try:
            spec.loader.exec_module(module)
            for name in ("mnist_eps_grid.yaml", "diabetes_eps_grid.yaml"):
                config = module.HpoConfig.from_yaml(root / "configs" / "hpo" / name)
                self.assertEqual(len(config.target_epsilons), 6)
                self.assertGreaterEqual(config.n_trials, 1)
                self.assertTrue(config.search_space.batch_sizes)
                self.assertLess(config.search_space.learning_rate_min,
                                config.search_space.learning_rate_max)
        finally:
            sys.modules.pop("tune_hyperparams", None)


if __name__ == "__main__":
    unittest.main()
