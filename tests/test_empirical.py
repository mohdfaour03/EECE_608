from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound
from dp_audit_tightness.utils.results import load_audit_result, load_json


class EmpiricalLowerBoundTests(unittest.TestCase):
    def test_member_favoring_event_can_be_selected(self) -> None:
        estimate = estimate_empirical_lower_bound(
            member_scores=[0.95, 0.9, 0.85, 0.8],
            nonmember_scores=[0.45, 0.4, 0.35, 0.3],
            delta=1e-5,
            align_event_to_score_direction=True,
            require_member_favoring=True,
        )
        self.assertEqual(estimate.selected_event_direction, "score>=threshold")
        self.assertTrue(estimate.member_favoring)
        self.assertFalse(estimate.no_member_favoring_event_found)
        self.assertGreater(estimate.epsilon_lower_empirical, 0.0)

    def test_non_member_favoring_events_are_rejected(self) -> None:
        estimate = estimate_empirical_lower_bound(
            member_scores=[0.3, 0.35, 0.4],
            nonmember_scores=[0.8, 0.85, 0.9],
            delta=1e-5,
            align_event_to_score_direction=True,
            require_member_favoring=True,
        )
        self.assertEqual(estimate.epsilon_lower_empirical, 0.0)
        self.assertEqual(estimate.empirical_ci_lower, 0.0)
        self.assertEqual(estimate.empirical_ci_upper, 0.0)
        self.assertIsNone(estimate.selected_threshold)
        self.assertFalse(estimate.member_favoring)
        self.assertTrue(estimate.no_member_favoring_event_found)
        self.assertIsNotNone(estimate.warning_message)

    def test_lower_score_direction_maps_to_score_less_than_threshold(self) -> None:
        estimate = estimate_empirical_lower_bound(
            member_scores=[0.1, 0.15, 0.2, 0.25],
            nonmember_scores=[0.6, 0.65, 0.7, 0.75],
            delta=1e-5,
            score_direction="lower",
            align_event_to_score_direction=True,
            require_member_favoring=True,
        )
        self.assertEqual(estimate.selected_event_direction, "score<threshold")
        self.assertTrue(estimate.member_favoring)
        self.assertGreater(estimate.epsilon_lower_empirical, 0.0)

    def test_clean_passive_artifacts_no_longer_yield_pathological_lower_bounds(self) -> None:
        passive_dir = ROOT / "results" / "substantive_smoke_3seed" / "audits" / "passive"
        artifact_dir = passive_dir / "artifacts"
        for audit_path in sorted(passive_dir.glob("passive_audit_seed*.json")):
            record = load_audit_result(audit_path)
            debug_payload = load_json(artifact_dir / f"{record.audit_run_id}_passive_debug.json")
            estimate = estimate_empirical_lower_bound(
                member_scores=debug_payload["member_scores"],
                nonmember_scores=debug_payload["nonmember_scores"],
                delta=record.delta,
                align_event_to_score_direction=True,
                require_member_favoring=True,
            )
            self.assertLessEqual(
                estimate.epsilon_lower_empirical,
                record.epsilon_upper_theory,
                msg=f"Pathological lower>upper persisted for {audit_path.name}",
            )
            self.assertTrue(
                bool(estimate.member_favoring) or estimate.no_member_favoring_event_found,
                msg=f"Expected member-favoring selection or conservative zero result for {audit_path.name}",
            )

    def test_confidence_supported_reporting_shrinks_sparse_tail_point_estimates(self) -> None:
        estimate = estimate_empirical_lower_bound(
            member_scores=[1.0] * 7 + [0.0] * 57,
            nonmember_scores=[0.0] * 64,
            delta=1e-5,
            align_event_to_score_direction=True,
            require_member_favoring=True,
            report_confidence_supported_lower_bound=True,
        )
        self.assertGreater(estimate.epsilon_lower_empirical_point_estimate, 0.0)
        self.assertEqual(estimate.empirical_ci_lower, 0.0)
        self.assertEqual(estimate.epsilon_lower_empirical, 0.0)
        self.assertEqual(estimate.epsilon_lower_empirical_conservative, 0.0)
        self.assertTrue(estimate.selected_event_is_tiny_tail)
        self.assertIsNotNone(estimate.warning_message)

    def test_metadata_separates_point_and_conservative_values(self) -> None:
        estimate = estimate_empirical_lower_bound(
            member_scores=[0.9, 0.9, 0.9, 0.2, 0.2, 0.2],
            nonmember_scores=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            delta=1e-5,
            align_event_to_score_direction=True,
            require_member_favoring=True,
            report_confidence_supported_lower_bound=True,
        )
        self.assertGreaterEqual(
            estimate.epsilon_lower_empirical_point_estimate,
            estimate.epsilon_lower_empirical_conservative,
        )
        self.assertEqual(estimate.epsilon_lower_empirical, estimate.epsilon_lower_empirical_conservative)

    def test_clean_canary_artifacts_conservative_reporting_avoids_pathological_lower_bounds(self) -> None:
        canary_dir = ROOT / "results" / "canary_seedfix_3seed" / "audits" / "canary"
        artifact_dir = canary_dir / "artifacts"
        for audit_path in sorted(canary_dir.glob("canary_audit_seed*.json")):
            record = load_audit_result(audit_path)
            debug_payload = load_json(artifact_dir / f"{record.audit_run_id}_canary_debug.json")
            estimate = estimate_empirical_lower_bound(
                member_scores=debug_payload["member_scores"],
                nonmember_scores=debug_payload["nonmember_scores"],
                delta=record.delta,
                align_event_to_score_direction=True,
                require_member_favoring=True,
                report_confidence_supported_lower_bound=True,
            )
            self.assertLessEqual(
                estimate.epsilon_lower_empirical,
                record.epsilon_upper_theory,
                msg=f"Conservative canary lower bound still exceeds theory for {audit_path.name}",
            )
            self.assertEqual(
                estimate.epsilon_lower_empirical,
                estimate.epsilon_lower_empirical_conservative,
            )


class HoldoutThresholdSelectionTests(unittest.TestCase):
    """Tests for the sample-split holdout path (2026-07-06 fix #1)."""

    KW = dict(
        delta=1e-5,
        align_event_to_score_direction=True,
        require_member_favoring=True,
        report_confidence_supported_lower_bound=True,
        score_direction="higher",
    )

    def _est(self, m, n, ts):
        return estimate_empirical_lower_bound(
            member_scores=m, nonmember_scores=n, threshold_selection=ts, **self.KW
        )

    def test_null_coverage_holdout_is_zero_in_at_least_95pct(self) -> None:
        import random

        holdout_zero = 0
        insample_nonzero = 0
        for rep in range(200):
            rng = random.Random(1000 + rep)
            m = [rng.gauss(0, 1) for _ in range(640)]
            n = [rng.gauss(0, 1) for _ in range(640)]
            if self._est(m, n, "holdout").epsilon_lower_empirical_conservative == 0.0:
                holdout_zero += 1
            if self._est(m, n, "in_sample").epsilon_lower_empirical_conservative > 0.0:
                insample_nonzero += 1
        # Holdout must certify 0 under the null in >= 95% of reps (expect ~100%).
        self.assertGreaterEqual(holdout_zero, 190)
        # In-sample is anti-conservative: it can manufacture nonzero eps from noise.
        # (Documents why the fix exists.)
        self.assertGreaterEqual(insample_nonzero, 1)

    def test_signal_holdout_positive_and_below_insample_point(self) -> None:
        import random

        rng = random.Random(7)
        m = [rng.gauss(3, 1) for _ in range(640)]
        n = [rng.gauss(0, 1) for _ in range(640)]
        ho = self._est(m, n, "holdout")
        ins = self._est(m, n, "in_sample")
        self.assertGreater(ho.epsilon_lower_empirical_conservative, 0.0)
        self.assertLessEqual(
            ho.epsilon_lower_empirical_conservative,
            ins.epsilon_lower_empirical_point_estimate,
        )
        self.assertEqual(ho.threshold_selection, "holdout")

    def test_determinism(self) -> None:
        import random

        rng = random.Random(2)
        m = [rng.gauss(3, 1) for _ in range(320)]
        n = [rng.gauss(0, 1) for _ in range(320)]
        self.assertEqual(self._est(m, n, "holdout"), self._est(m, n, "holdout"))

    def test_tiny_inputs_return_conservative_zero_without_crash(self) -> None:
        est = self._est([0.1, 0.2, 0.3], [0.0, 0.05, 0.1], "holdout")
        self.assertEqual(est.epsilon_lower_empirical_conservative, 0.0)
        self.assertEqual(est.estimation_method, "threshold_sweep_holdout_insufficient_samples")

    def test_odd_length_inputs_do_not_crash(self) -> None:
        import random

        rng = random.Random(3)
        m = [rng.gauss(1, 1) for _ in range(41)]
        n = [rng.gauss(0, 1) for _ in range(37)]
        est = self._est(m, n, "holdout")  # must not raise
        self.assertGreaterEqual(est.epsilon_lower_empirical_conservative, 0.0)

    def test_in_sample_path_is_default_and_unchanged(self) -> None:
        import random

        rng = random.Random(9)
        m = [rng.gauss(2, 1) for _ in range(200)]
        n = [rng.gauss(0, 1) for _ in range(200)]
        default = estimate_empirical_lower_bound(member_scores=m, nonmember_scores=n, **self.KW)
        self.assertEqual(default.threshold_selection, "in_sample")
        self.assertIn("member_aligned", default.estimation_method)


if __name__ == "__main__":
    unittest.main()
