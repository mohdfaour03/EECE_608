from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.evaluation.metrics import (
    compute_privacy_tightness_metrics,
    empirical_lower_bound_is_valid,
)


class PrivacyTightnessMetricsTests(unittest.TestCase):
    def test_valid_lower_bound_has_no_sanity_warning(self) -> None:
        metrics = compute_privacy_tightness_metrics(
            epsilon_upper_theory=1.0,
            epsilon_lower_empirical=0.25,
        )
        self.assertEqual(metrics.privacy_loss_gap, 0.75)
        self.assertEqual(metrics.tightness_ratio, 0.25)
        self.assertTrue(metrics.valid_empirical_lower_bound)
        self.assertIsNone(metrics.sanity_warning)

    def test_lower_bound_above_upper_is_marked_diagnostic(self) -> None:
        metrics = compute_privacy_tightness_metrics(
            epsilon_upper_theory=1.0,
            epsilon_lower_empirical=1.25,
        )
        self.assertFalse(metrics.valid_empirical_lower_bound)
        self.assertIsNotNone(metrics.sanity_warning)

    def test_sanity_check_allows_tiny_roundoff(self) -> None:
        self.assertTrue(
            empirical_lower_bound_is_valid(
                epsilon_upper_theory=1.0,
                epsilon_lower_empirical=1.0 + 5e-13,
            )
        )


if __name__ == "__main__":
    unittest.main()
