"""Generate training configs for an epsilon sweep.

Given a template config and a list of noise multiplier values, writes one
YAML config file per (noise_multiplier, seed) combination.  This is the
entry point for producing the tightness-vs-epsilon curves.

Usage
-----
    python experiments/generate_sweep_configs.py \
        --template configs/train/cifar10_cnn_dp_sgd_substantive.yaml \
        --noise-multipliers 0.5 0.8 1.1 1.5 2.0 \
        --seeds 123 124 125 \
        --output-dir configs/sweep/cifar10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def generate_sweep_configs(
    template_path: str | Path,
    noise_multipliers: list[float],
    training_seeds: list[int],
    output_dir: str | Path,
    *,
    epochs_override: int | None = None,
) -> list[Path]:
    """Write one config per (noise_multiplier, seed).  Returns written paths."""
    template_path = Path(template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with template_path.open("r", encoding="utf-8") as fh:
        template = yaml.safe_load(fh)

    written: list[Path] = []
    for sigma in noise_multipliers:
        for seed in training_seeds:
            cfg = _deep_copy_dict(template)
            cfg["training"]["noise_multiplier"] = sigma
            cfg["run"]["training_seed"] = seed
            # Remove seed lists — each config is a single-seed run.
            cfg["run"].pop("training_seeds", None)
            cfg["run"].pop("split_seeds", None)
            if epochs_override is not None:
                cfg["training"]["epochs"] = epochs_override
            tag = f"sigma{sigma}_seed{seed}"
            cfg["experiment_name"] = f"{template['experiment_name']}_sweep_{tag}"
            cfg["run"]["notes"] = (
                f"Epsilon sweep: noise_multiplier={sigma}, seed={seed}. "
                f"Generated from {template_path.name}."
            )
            out_path = output_dir / f"{cfg['experiment_name']}.yaml"
            with out_path.open("w", encoding="utf-8") as fh:
                yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
            written.append(out_path)
    return written


def _deep_copy_dict(d: dict) -> dict:
    """Simple recursive dict copy (avoids importing copy)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate epsilon sweep configs.")
    parser.add_argument("--template", required=True, help="Path to template YAML config.")
    parser.add_argument(
        "--noise-multipliers",
        nargs="+",
        type=float,
        required=True,
        help="List of noise multiplier values to sweep.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[123, 124, 125],
        help="Training seeds (default: 123 124 125).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write generated configs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    args = parser.parse_args()

    written = generate_sweep_configs(
        template_path=args.template,
        noise_multipliers=args.noise_multipliers,
        training_seeds=args.seeds,
        output_dir=args.output_dir,
        epochs_override=args.epochs,
    )
    for path in written:
        print(f"  wrote {path}")
    print(f"\nGenerated {len(written)} configs in {args.output_dir}")


if __name__ == "__main__":
    main()
