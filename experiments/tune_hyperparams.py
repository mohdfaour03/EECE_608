"""Per-epsilon hyperparameter optimization with Optuna (professor feedback, 2026-07-08).

For each target epsilon in the config grid, runs an Optuna TPE study over
(learning_rate, batch_size, epochs, clipping_norm). Because batch size and
epochs change the privacy cost, every trial RE-SOLVES the noise multiplier so
that eps_PLD(sigma, q, T, delta) == target_epsilon (see
dp_audit_tightness.privacy.sigma_solver). The tuner therefore compares utility
at a genuinely constant privacy level — "different hyperparameters for every
epsilon" done rigorously.

Outputs per (dataset, target_epsilon):
  results_root/hpo/<study_name>.json           full study record (all trials)
  results_root/hpo/best_configs/<study>.yaml   winning config, ready for run_train.py
                                               and the audit runners

Determinism: Optuna TPESampler is seeded from the config; each trial's
training seed is derived as base_training_seed + trial.number so no two trials
share torch randomness, and the winning config freezes the ORIGINAL seed list
(123/124/125 by default) so final audited models remain comparable across
epsilon cells.

Honesty note for the paper: hyperparameter search itself consumes privacy
budget in principle (Papernot & Steinke, 2022). We follow standard auditing
practice and treat tuning as out-of-band (the audited guarantee covers the
final training run), but the paper must state this explicitly, and the
Empirical Privacy Variance phenomenon (Hu et al., 2025) means the tuned
configuration is part of what we are measuring.

Usage:
  python experiments/tune_hyperparams.py --config configs/hpo/mnist_eps_grid.yaml
  python experiments/tune_hyperparams.py --config configs/hpo/mnist_eps_grid.yaml --epsilon 1.0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import logging
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from dp_audit_tightness.config import TrainExperimentConfig, config_to_dict, load_train_config
from dp_audit_tightness.privacy.sigma_solver import SigmaSolverError, solve_sigma_for_epsilon
from dp_audit_tightness.utils.logging_utils import configure_logging
from dp_audit_tightness.utils.seeds import set_global_seed


@dataclass(slots=True)
class HpoSearchSpace:
    learning_rate_min: float = 0.01
    learning_rate_max: float = 1.0
    batch_sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    epochs_min: int = 1
    epochs_max: int = 15
    clipping_norms: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])


@dataclass(slots=True)
class HpoConfig:
    base_train_config: str
    target_epsilons: list[float]
    n_trials: int = 30
    sampler_seed: int = 20260708
    base_trial_training_seed: int = 900_000
    study_name_prefix: str = "hpo"
    results_root: str = "results"
    search_space: HpoSearchSpace = field(default_factory=HpoSearchSpace)

    @classmethod
    def from_yaml(cls, path: Path) -> "HpoConfig":
        payload = yaml.safe_load(path.read_text())
        space = HpoSearchSpace(**payload.pop("search_space", {}))
        return cls(search_space=space, **payload)


def _steps_per_epoch(train_size: int, batch_size: int) -> int:
    return max(1, math.ceil(train_size / batch_size))


def build_trial_config(
    base: TrainExperimentConfig,
    *,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    clipping_norm: float,
    sigma: float,
    sampling_rate: float,
    training_seed: int,
) -> TrainExperimentConfig:
    return replace(
        base,
        training=replace(
            base.training,
            batch_size=batch_size,
            epochs=epochs,
            clipping_norm=clipping_norm,
            noise_multiplier=sigma,
            sampling_rate=sampling_rate,
            optimizer=replace(base.training.optimizer, learning_rate=learning_rate),
        ),
        run=replace(
            base.run,
            training_seed=training_seed,
            split_seeds=None,
            training_seeds=None,
            save_checkpoint=False,
            save_debug_artifacts=False,
        ),
    )


def run_study(
    hpo: HpoConfig,
    target_epsilon: float,
    logger: logging.Logger,
    *,
    storage: str | None = None,
    checkpoint_command: str | None = None,
) -> dict[str, Any]:
    import optuna
    from dp_audit_tightness.data.datasets import load_dataset_bundle
    from dp_audit_tightness.training.dp_sgd import train_dp_sgd

    base = load_train_config(Path(hpo.base_train_config))
    bundle = load_dataset_bundle(base.dataset, split_seed=base.run.split_seed)
    train_size = bundle.train_size
    delta = base.privacy.delta
    space = hpo.search_space

    def objective(trial: "optuna.trial.Trial") -> float:
        learning_rate = trial.suggest_float("learning_rate", space.learning_rate_min, space.learning_rate_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", space.batch_sizes)
        epochs = trial.suggest_int("epochs", space.epochs_min, space.epochs_max)
        clipping_norm = trial.suggest_categorical("clipping_norm", space.clipping_norms)

        sampling_rate = batch_size / train_size
        num_steps = _steps_per_epoch(train_size, batch_size) * epochs
        try:
            solution = solve_sigma_for_epsilon(
                target_epsilon=target_epsilon,
                sampling_rate=sampling_rate,
                num_steps=num_steps,
                delta=delta,
            )
        except SigmaSolverError as exc:
            # Configuration cannot reach the target epsilon -> prune, don't crash.
            raise optuna.TrialPruned(f"sigma unsolvable: {exc}") from exc
        trial.set_user_attr("sigma", solution.sigma)
        trial.set_user_attr("achieved_epsilon", solution.achieved_epsilon)
        trial.set_user_attr("num_steps", num_steps)

        trial_seed = hpo.base_trial_training_seed + trial.number
        set_global_seed(trial_seed)
        config = build_trial_config(
            base,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            clipping_norm=clipping_norm,
            sigma=solution.sigma,
            sampling_rate=sampling_rate,
            training_seed=trial_seed,
        )
        outcome = train_dp_sgd(
            config=config,
            logger=logger,
            dataset_bundle=bundle,
            save_checkpoint=False,
            run_descriptor=f"hpo_eps{target_epsilon}_trial{trial.number}",
        )
        accuracy = float(outcome.record.utility_metrics.get("accuracy", 0.0))
        trial.set_user_attr("eval_loss", float(outcome.record.utility_metrics.get("loss", float("nan"))))
        return accuracy

    storage_url = storage
    if storage_url and "://" not in storage_url:
        storage_path = Path(storage_url).expanduser().resolve()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=hpo.sampler_seed)
    study = optuna.create_study(
        study_name=f"{hpo.study_name_prefix}_{base.dataset.name}_eps{target_epsilon}",
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=storage_url is not None,
    )
    counted_states = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    finished_trials = sum(trial.state in counted_states for trial in study.trials)
    remaining_trials = max(0, hpo.n_trials - finished_trials)
    logger.info(
        "study=%s finished_trials=%d remaining_trials=%d storage=%s",
        study.study_name,
        finished_trials,
        remaining_trials,
        storage_url or "in-memory",
    )
    if remaining_trials:
        callbacks = []
        if checkpoint_command:
            checkpoint_args = shlex.split(checkpoint_command)

            def checkpoint_after_trial(
                _study: "optuna.study.Study",
                trial: "optuna.trial.FrozenTrial",
            ) -> None:
                env = os.environ.copy()
                env["OPTUNA_TRIAL_NUMBER"] = str(trial.number)
                env["OPTUNA_TRIAL_STATE"] = trial.state.name
                logger.info(
                    "trial=%d state=%s checkpoint_command=%s",
                    trial.number,
                    trial.state.name,
                    checkpoint_args[0],
                )
                subprocess.run(checkpoint_args, check=True, env=env)

            callbacks.append(checkpoint_after_trial)
        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=True,
            callbacks=callbacks,
        )

    best = study.best_trial
    best_sampling_rate = best.params["batch_size"] / train_size
    winning_config = build_trial_config(
        base,
        learning_rate=best.params["learning_rate"],
        batch_size=best.params["batch_size"],
        epochs=best.params["epochs"],
        clipping_norm=best.params["clipping_norm"],
        sigma=best.user_attrs["sigma"],
        sampling_rate=best_sampling_rate,
        training_seed=base.run.training_seed,   # restore canonical seeds for the audited runs
    )
    winning_config = replace(
        winning_config,
        experiment_name=f"{base.experiment_name}_tuned_eps{target_epsilon}",
        run=replace(
            winning_config.run,
            save_checkpoint=True,
            save_debug_artifacts=True,
            split_seeds=base.run.split_seeds,
            training_seeds=base.run.training_seeds,
            notes=(
                f"Optuna-tuned for target_epsilon={target_epsilon} "
                f"(study={study.study_name}, n_trials={hpo.n_trials}, sampler_seed={hpo.sampler_seed}); "
                f"sigma solved to achieve eps_PLD={best.user_attrs['achieved_epsilon']:.4f}."
            ),
        ),
    )

    record = {
        "study_name": study.study_name,
        "dataset": base.dataset.name,
        "target_epsilon": target_epsilon,
        "delta": delta,
        "train_size": train_size,
        "n_trials": hpo.n_trials,
        "sampler": "TPESampler",
        "sampler_seed": hpo.sampler_seed,
        "search_space": asdict(hpo.search_space),
        "best_value_eval_accuracy": study.best_value,
        "best_params": best.params,
        "best_user_attrs": dict(best.user_attrs),
        "trials": [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
                "user_attrs": dict(t.user_attrs),
            }
            for t in study.trials
        ],
    }

    hpo_dir = Path(hpo.results_root) / "hpo"
    (hpo_dir / "best_configs").mkdir(parents=True, exist_ok=True)
    record_path = hpo_dir / f"{study.study_name}.json"
    record_path.write_text(json.dumps(record, indent=2))
    config_path = hpo_dir / "best_configs" / f"{study.study_name}.yaml"
    config_path.write_text(yaml.safe_dump(config_to_dict(winning_config), sort_keys=False))
    logger.info("study=%s best_acc=%.4f -> %s", study.study_name, study.best_value, config_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an HPO YAML config (configs/hpo/).")
    parser.add_argument("--epsilon", type=float, default=None, help="Run only this target epsilon from the grid.")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL or SQLite path used to resume interrupted studies.",
    )
    parser.add_argument(
        "--checkpoint-command",
        default=None,
        help=(
            "Optional command run after every Optuna trial. The command receives "
            "OPTUNA_TRIAL_NUMBER and OPTUNA_TRIAL_STATE environment variables; a "
            "non-zero exit stops the study so uncheckpointed work is not continued."
        ),
    )
    args = parser.parse_args()

    logger = configure_logging()
    hpo = HpoConfig.from_yaml(Path(args.config))
    grid = [args.epsilon] if args.epsilon is not None else hpo.target_epsilons
    for target in grid:
        run_study(
            hpo,
            float(target),
            logger,
            storage=args.storage,
            checkpoint_command=args.checkpoint_command,
        )


if __name__ == "__main__":
    main()
