from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_audit_tightness.reporting.canary_estimator_diagnostics import (
    generate_canary_estimator_diagnostics,
)
from dp_audit_tightness.utils.logging_utils import configure_logging, format_kv_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose canary empirical lower-bound threshold selection on saved audit artifacts."
    )
    parser.add_argument(
        "--canary-results-dir",
        default="results/canary_seedfix_3seed/audits/canary",
        help="Directory containing post-seedfix canary audit JSON files.",
    )
    parser.add_argument(
        "--canary-artifacts-dir",
        default="results/canary_seedfix_3seed/audits/canary/artifacts",
        help="Directory containing paired canary debug score dumps.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/substantive_smoke_3seed/audits/canary_estimator_diagnostics",
        help="Directory where the diagnostic outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    outputs = generate_canary_estimator_diagnostics(
        canary_results_dir=Path(args.canary_results_dir),
        canary_artifacts_dir=Path(args.canary_artifacts_dir),
        output_dir=Path(args.output_dir),
    )
    logger.info(
        "Saved canary estimator diagnostics %s",
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
