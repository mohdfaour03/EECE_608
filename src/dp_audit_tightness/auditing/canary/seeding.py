from __future__ import annotations

from dataclasses import asdict, dataclass


_SEED_MODULUS = 2_147_483_647


@dataclass(slots=True)
class CanarySeedPlan:
    experiment_seed: int
    audit_seed: int
    dataset_split_seed: int
    canary_generation_seed: int
    canary_insertion_seed: int
    retrain_seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def build_canary_seed_plan(
    *,
    experiment_seed: int,
    audit_seed: int,
    dataset_split_seed: int,
) -> CanarySeedPlan:
    """Derive deterministic canary seeds from the experiment seed and audit seed.

    This ensures that repeated-run canary audits remain reproducible while still
    changing materially across different experiment seeds.
    """

    return CanarySeedPlan(
        experiment_seed=experiment_seed,
        audit_seed=audit_seed,
        dataset_split_seed=dataset_split_seed,
        canary_generation_seed=_derive_seed(experiment_seed, audit_seed, stream=11),
        canary_insertion_seed=_derive_seed(experiment_seed, audit_seed, stream=23),
        retrain_seed=_derive_seed(experiment_seed, audit_seed, stream=37),
    )


def _derive_seed(experiment_seed: int, audit_seed: int, *, stream: int) -> int:
    seed = (
        abs(experiment_seed) * 1_000_003
        + abs(audit_seed) * 9_176
        + stream * 104_729
        + 17
    ) % _SEED_MODULUS
    return seed if seed != 0 else stream + 1
