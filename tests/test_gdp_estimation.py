from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.privacy.gdp_estimation import (
    _auc_to_mu,
    _compute_auc,
    _mu_to_epsilon,
    estimate_epsilon_gdp,
)


class GDPEstimationTests(unittest.TestCase):
    def test_auc_uses_member_greater_than_nonmember_with_ties(self) -> None:
        auc = _compute_auc(
            member=np.array([0.5, 0.7, 0.7]),
            nonmember=np.array([0.4, 0.7, 0.8]),
        )
        self.assertAlmostEqual(auc, 4.0 / 9.0)

    def test_higher_score_direction_produces_positive_signal(self) -> None:
        estimate = estimate_epsilon_gdp(
            member_scores=[0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95, 0.97],
            nonmember_scores=[0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.27],
            delta=1e-5,
            n_bootstrap=100,
            score_direction="higher",
        )
        self.assertEqual(estimate.score_direction, "higher")
        self.assertGreater(estimate.auc, 0.5)
        self.assertGreater(estimate.mu_point, 0.0)
        self.assertGreater(estimate.epsilon_lower_point, 0.0)

    def test_lower_score_direction_is_negated_before_auc(self) -> None:
        estimate = estimate_epsilon_gdp(
            member_scores=[0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.27],
            nonmember_scores=[0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95, 0.97],
            delta=1e-5,
            n_bootstrap=100,
            score_direction="lower",
        )
        self.assertEqual(estimate.score_direction, "lower")
        self.assertGreater(estimate.auc, 0.5)
        self.assertGreater(estimate.mu_point, 0.0)
        self.assertGreater(estimate.epsilon_lower_point, 0.0)

    def test_no_signal_maps_to_zero_mu_and_epsilon(self) -> None:
        self.assertEqual(_auc_to_mu(0.5), 0.0)
        estimate = estimate_epsilon_gdp(
            member_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            nonmember_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            delta=1e-5,
            n_bootstrap=100,
        )
        self.assertEqual(estimate.mu_point, 0.0)
        self.assertEqual(estimate.epsilon_lower_point, 0.0)

    def test_mu_to_epsilon_is_monotone(self) -> None:
        eps_small = _mu_to_epsilon(mu=0.25, delta=1e-5)
        eps_large = _mu_to_epsilon(mu=0.75, delta=1e-5)
        self.assertGreater(eps_small, 0.0)
        self.assertGreater(eps_large, eps_small)

    def test_too_few_samples_returns_zero_with_warning(self) -> None:
        estimate = estimate_epsilon_gdp(
            member_scores=[0.9, 0.8, 0.7, 0.6],
            nonmember_scores=[0.1, 0.2, 0.3, 0.4],
            delta=1e-5,
        )
        self.assertEqual(estimate.epsilon_lower, 0.0)
        self.assertEqual(estimate.epsilon_lower_point, 0.0)
        self.assertIsNotNone(estimate.warning)


if __name__ == "__main__":
    unittest.main()
