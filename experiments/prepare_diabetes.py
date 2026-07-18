"""Prepare the CDC Diabetes Health Indicators dataset for the audit pipeline.

Converts the BRFSS-2015 CSV (diabetes_binary_health_indicators_BRFSS2015.csv,
253,680 rows x 22 columns, first column = binary label 'Diabetes_binary') into
the .npz format the loader expects: 'features' float32 [N, 21], 'labels' int64 [N].

Sources (either works, identical file):
  - UCI ML Repository, dataset id 891 ("CDC Diabetes Health Indicators")
  - Kaggle: alexteboul/diabetes-health-indicators-dataset

Preprocessing (deterministic, no randomness):
  - Continuous columns (BMI, GenHlth, MentHlth, PhysHlth, Age, Education, Income)
    are z-score standardized with statistics computed over the FULL file; the
    member/non-member split happens later, seeded, inside load_dataset_bundle,
    and standardization statistics are not a per-record privacy surface we audit.
  - Binary indicator columns pass through unchanged.
  - Labels: Diabetes_binary cast to int64.

Usage:
  python experiments/prepare_diabetes.py --csv data/raw/diabetes_binary_health_indicators_BRFSS2015.csv \
      --out data/raw/cdc_diabetes.npz

Optionally fetch via ucimlrepo (pip install ucimlrepo) with --fetch-uci.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LABEL_COLUMN = "Diabetes_binary"
CONTINUOUS_COLUMNS = ("BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income")
EXPECTED_FEATURES = 21
EXPECTED_ROWS = 253_680


def build_npz(csv_path: Path, out_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if LABEL_COLUMN not in frame.columns:
        raise SystemExit(
            f"Column '{LABEL_COLUMN}' missing — got {list(frame.columns)[:5]}... "
            "Use the *binary* BRFSS2015 CSV (not the 012 multiclass variant)."
        )
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
    features_frame = frame.drop(columns=[LABEL_COLUMN])
    if features_frame.shape[1] != EXPECTED_FEATURES:
        raise SystemExit(
            f"Expected {EXPECTED_FEATURES} feature columns, found {features_frame.shape[1]}."
        )
    for column in CONTINUOUS_COLUMNS:
        col = features_frame[column].astype(np.float64)
        std = float(col.std())
        if std == 0.0:
            raise SystemExit(f"Zero-variance column {column}; refusing to standardize.")
        features_frame[column] = (col - float(col.mean())) / std
    features = features_frame.to_numpy(dtype=np.float32)

    if len(labels) != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows, found {len(labels)} (continuing).")
    positive_rate = float(labels.mean())
    print(f"rows={len(labels)}  features={features.shape[1]}  positive_rate={positive_rate:.4f}")
    if not (0.10 <= positive_rate <= 0.20):
        print("WARNING: positive rate outside the expected ~13.9% for the binary variant.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=features, labels=labels)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def fetch_uci(csv_path: Path) -> None:
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo

    bundle = fetch_ucirepo(id=891)
    frame = bundle.data.features.copy()
    frame.insert(0, LABEL_COLUMN, bundle.data.targets.iloc[:, 0].to_numpy())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    print(f"fetched UCI id=891 -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/diabetes_binary_health_indicators_BRFSS2015.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/cdc_diabetes.npz"))
    parser.add_argument("--fetch-uci", action="store_true", help="download via ucimlrepo first")
    args = parser.parse_args()
    if args.fetch_uci and not args.csv.exists():
        fetch_uci(args.csv)
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv} (pass --fetch-uci or download manually).")
    build_npz(args.csv, args.out)


if __name__ == "__main__":
    main()
