from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.reporting.passive_direction_diagnostics import (
    generate_passive_direction_diagnostics,
)
from dp_audit_tightness.utils.logging_utils import configure_logging, format_kv_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose passive threshold-direction behavior on saved audit artifacts."
    )
    parser.add_argument(
        "--passive-results-dir",
        default="results/substantive_smoke_3seed/audits/passive",
        help="Directory containing clean passive audit JSON files.",
    )
    parser.add_argument(
        "--passive-artifacts-dir",
        default="results/substantive_smoke_3seed/audits/passive/artifacts",
        help="Directory containing paired passive debug score dumps.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/substantive_smoke_3seed/audits/passive_direction_diagnostics",
        help="Directory where diagnostic outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    outputs = generate_passive_direction_diagnostics(
        passive_results_dir=Path(args.passive_results_dir),
        passive_artifacts_dir=Path(args.passive_artifacts_dir),
        output_dir=Path(args.output_dir),
    )
    logger.info(
        "Saved passive direction diagnostics %s",
        format_kv_fields(
            comparison_csv=outputs["comparison_csv"],
            comparison_json=outputs["comparison_json"],
            score_summary_csv=outputs["score_summary_csv"],
            score_summary_json=outputs["score_summary_json"],
            summary_markdown=outputs["summary_markdown"],
        ),
    )


if __name__ == "__main__":
    main()
