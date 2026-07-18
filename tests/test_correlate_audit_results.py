from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.correlate_audit_results import correlate_results
from dp_audit_tightness.utils.results import CANARY_AUDIT_REGIME, PASSIVE_AUDIT_REGIME


class CorrelateAuditResultsTests(unittest.TestCase):
    def test_correlation_reads_regime_subdirectories_and_prefers_valid_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "comparisons"
            self._write_json(
                root / "training" / "train_demo.json",
                {
                    "training_run_id": "train_demo",
                    "dataset": "mnist",
                    "model_name": "simple_mlp",
                    "epsilon_upper_theory": 1.0,
                    "epsilon_upper_pld": 0.8,
                    "pld_accounting_backend": "google_dp_accounting",
                    "noise_multiplier": 1.1,
                    "clipping_norm": 1.0,
                    "batch_size": 256,
                    "epochs": 1,
                    "delta": 1e-5,
                    "sampling_rate": 0.01,
                    "utility_metrics": {"accuracy": 0.8},
                },
            )

            self._write_audit(root, "canary", "bad_canary", CANARY_AUDIT_REGIME, 1.2, False)
            self._write_audit(root, "canary", "good_canary", CANARY_AUDIT_REGIME, 0.4, True)
            self._write_audit(root, "passive", "bad_passive", PASSIVE_AUDIT_REGIME, 1.1, False)
            self._write_audit(root, "passive", "good_passive", PASSIVE_AUDIT_REGIME, 0.2, True)

            written = correlate_results(root, output_dir)
            self.assertEqual(len(written), 1)

            comparison = json.loads(written[0].read_text(encoding="utf-8"))
            self.assertEqual(comparison["canary_audit"]["auditor_variant"], "good_canary")
            self.assertEqual(comparison["passive_audit"]["auditor_variant"], "good_passive")
            self.assertTrue(comparison["canary_audit"]["valid_empirical_lower_bound"])
            self.assertTrue(comparison["passive_audit"]["valid_empirical_lower_bound"])

    def _write_audit(
        self,
        root: Path,
        subdir: str,
        variant: str,
        regime: str,
        epsilon_lower: float,
        valid: bool,
    ) -> None:
        self._write_json(
            root / "audits" / subdir / f"{variant}.json",
            {
                "audit_run_id": variant,
                "training_run_id": "train_demo",
                "audit_regime": regime,
                "auditor_variant": variant,
                "epsilon_upper_theory": 1.0,
                "epsilon_lower_empirical": epsilon_lower,
                "empirical_ci_lower": 0.0,
                "empirical_ci_upper": epsilon_lower,
                "privacy_loss_gap": 1.0 - epsilon_lower,
                "tightness_ratio": epsilon_lower,
                "audit_metadata": {
                    "valid_empirical_lower_bound": valid,
                    "sanity_warning": None if valid else "invalid",
                    "raw_statistics": {},
                },
            },
        )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
