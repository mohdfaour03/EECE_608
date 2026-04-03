from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.config import config_to_dict, load_train_config
from dp_audit_tightness.training.dp_sgd import train_dp_sgd
from dp_audit_tightness.utils.logging_utils import configure_logging, format_kv_fields
from dp_audit_tightness.utils.results import save_json, save_training_result
from dp_audit_tightness.utils.seeds import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one DP-SGD model and save run metadata.")
    parser.add_argument("--config", required=True, help="Path to a training YAML config.")
    parser.add_argument("--training-seeds", nargs="*", type=int, help="Optional training-seed sweep.")
    parser.add_argument("--split-seeds", nargs="*", type=int, help="Optional split-seed sweep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    config_path = Path(args.config)
    config = load_train_config(config_path)
    for split_seed, training_seed in _resolve_seed_pairs(config, args):
        active_config = replace(
            config,
            run=replace(
                config.run,
                split_seed=split_seed,
                training_seed=training_seed,
            ),
        )
        set_global_seed(training_seed)
        logger.info(
            "Loaded training config %s",
            format_kv_fields(
                dataset=active_config.dataset.name,
                model=active_config.model.name,
                noise_multiplier=active_config.training.noise_multiplier,
                delta=active_config.privacy.delta,
                split_seed=split_seed,
                training_seed=training_seed,
            ),
        )

        outcome = train_dp_sgd(config=active_config, logger=logger)
        config_snapshot_path = Path(active_config.run.results_root) / "training" / "configs" / (
            f"{outcome.record.training_run_id}_config.json"
        )
        outcome.record.config_snapshot_path = str(config_snapshot_path)
        save_json(config_snapshot_path, config_to_dict(active_config))
        training_result_path = save_training_result(outcome.record, results_root=active_config.run.results_root)

        logger.info("Saved training result to %s", training_result_path)
        logger.info("Saved config snapshot to %s", config_snapshot_path)
        if outcome.checkpoint_path is not None:
            logger.info("Saved model artifact to %s", outcome.checkpoint_path)


def _resolve_seed_pairs(config, args) -> list[tuple[int, int]]:
    split_seeds = args.split_seeds or config.run.split_seeds or [config.run.split_seed]
    training_seeds = args.training_seeds or config.run.training_seeds or [config.run.training_seed]
    if len(split_seeds) == 1 and len(training_seeds) > 1:
        split_seeds = split_seeds * len(training_seeds)
    elif len(training_seeds) == 1 and len(split_seeds) > 1:
        training_seeds = training_seeds * len(split_seeds)
    elif len(split_seeds) != len(training_seeds):
        raise ValueError("split seeds and training seeds must have matching lengths or broadcast from length 1.")
    return list(zip(split_seeds, training_seeds, strict=False))


if __name__ == "__main__":
    main()
