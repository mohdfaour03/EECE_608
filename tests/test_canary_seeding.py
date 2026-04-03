from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.auditing.canary.generation import generate_canaries
from dp_audit_tightness.auditing.canary.seeding import build_canary_seed_plan
from dp_audit_tightness.config import CanaryConfig


class CanarySeedingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canary_config = CanaryConfig(
            strategy="random_canaries",
            num_canaries=4,
            insertion_rate=0.01,
            optimize_steps=0,
        )

    def test_seed_plan_is_reproducible(self) -> None:
        left = build_canary_seed_plan(experiment_seed=123, audit_seed=101, dataset_split_seed=11)
        right = build_canary_seed_plan(experiment_seed=123, audit_seed=101, dataset_split_seed=11)
        self.assertEqual(left.to_dict(), right.to_dict())

    def test_different_experiment_seeds_change_derived_seeds(self) -> None:
        left = build_canary_seed_plan(experiment_seed=123, audit_seed=101, dataset_split_seed=11)
        right = build_canary_seed_plan(experiment_seed=124, audit_seed=101, dataset_split_seed=11)
        self.assertNotEqual(left.canary_generation_seed, right.canary_generation_seed)
        self.assertNotEqual(left.canary_insertion_seed, right.canary_insertion_seed)
        self.assertNotEqual(left.retrain_seed, right.retrain_seed)

    def test_canary_generation_is_reproducible_for_fixed_seed(self) -> None:
        seed_plan = build_canary_seed_plan(experiment_seed=123, audit_seed=101, dataset_split_seed=11)
        first = generate_canaries(self.canary_config, seed_plan.canary_generation_seed, num_classes=10)
        second = generate_canaries(self.canary_config, seed_plan.canary_generation_seed, num_classes=10)
        self.assertEqual(
            [(payload.identifier, payload.target_label, payload.descriptor) for payload in first],
            [(payload.identifier, payload.target_label, payload.descriptor) for payload in second],
        )

    def test_different_experiment_seeds_can_change_canary_payloads(self) -> None:
        left_plan = build_canary_seed_plan(experiment_seed=123, audit_seed=101, dataset_split_seed=11)
        right_plan = build_canary_seed_plan(experiment_seed=124, audit_seed=101, dataset_split_seed=11)
        left = generate_canaries(self.canary_config, left_plan.canary_generation_seed, num_classes=10)
        right = generate_canaries(self.canary_config, right_plan.canary_generation_seed, num_classes=10)
        self.assertNotEqual(
            [(payload.target_label, payload.descriptor) for payload in left],
            [(payload.target_label, payload.descriptor) for payload in right],
        )

    def test_seed_metadata_records_actual_seed_fields(self) -> None:
        seed_plan = build_canary_seed_plan(experiment_seed=125, audit_seed=102, dataset_split_seed=11)
        self.assertEqual(
            set(seed_plan.to_dict().keys()),
            {
                "experiment_seed",
                "audit_seed",
                "dataset_split_seed",
                "canary_generation_seed",
                "canary_insertion_seed",
                "retrain_seed",
            },
        )


if __name__ == "__main__":
    unittest.main()
