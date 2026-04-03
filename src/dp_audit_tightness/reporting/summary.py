from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any

from dp_audit_tightness.utils.paths import build_run_id
from dp_audit_tightness.utils.results import AggregatedSummaryRecord, AuditRunRecord, TrainingRunRecord


def aggregate_audit_records(records: list[AuditRunRecord]) -> list[AggregatedSummaryRecord]:
    grouped: dict[tuple[str, str, str, str], list[AuditRunRecord]] = defaultdict(list)
    for record in records:
        key = (record.dataset, record.model_name, record.audit_regime, record.auditor_variant)
        grouped[key].append(record)

    summaries: list[AggregatedSummaryRecord] = []
    for (dataset, model_name, audit_regime, auditor_variant), group in grouped.items():
        lower_values = [record.epsilon_lower_empirical for record in group]
        upper_values = [record.epsilon_upper_theory for record in group]
        gap_values = [record.privacy_loss_gap for record in group]
        ratio_values = [record.tightness_ratio for record in group if record.tightness_ratio is not None]
        saturation_rate = sum(record.saturation_detected for record in group) / len(group)
        summaries.append(
            AggregatedSummaryRecord(
                summary_id=build_run_id("summary", f"{audit_regime}_{auditor_variant}"),
                dataset=dataset,
                model_name=model_name,
                audit_regime=audit_regime,
                auditor_variant=auditor_variant,
                num_runs=len(group),
                mean_epsilon_upper_theory=statistics.fmean(upper_values),
                mean_epsilon_lower_empirical=statistics.fmean(lower_values),
                std_epsilon_lower_empirical=statistics.pstdev(lower_values) if len(group) > 1 else 0.0,
                mean_privacy_loss_gap=statistics.fmean(gap_values),
                mean_tightness_ratio=statistics.fmean(ratio_values) if ratio_values else None,
                saturation_rate=saturation_rate,
            )
        )
    return summaries


def summarize_saturation(records: list[AuditRunRecord]) -> dict[str, float]:
    if not records:
        return {"saturation_rate": math.nan}
    return {"saturation_rate": sum(record.saturation_detected for record in records) / len(records)}


def build_audit_run_summary_rows(
    training_records: dict[str, TrainingRunRecord],
    audit_records: list[AuditRunRecord],
) -> list[dict[str, Any]]:
    utility_keys = sorted(
        {
            key
            for record in audit_records
            for key in record.utility_metrics.keys()
        }
    )
    rows: list[dict[str, Any]] = []
    for record in audit_records:
        if record.training_run_id not in training_records:
            raise ValueError(
                f"Missing training record for audit run {record.audit_run_id}: {record.training_run_id}"
            )
        row: dict[str, Any] = {
            "run_id": record.audit_run_id,
            "seed": record.training_seed,
            "audit_regime": record.audit_regime,
            "auditor_variant": record.auditor_variant,
            "epsilon_upper_theory": record.epsilon_upper_theory,
            "epsilon_lower_empirical": record.epsilon_lower_empirical,
            "privacy_loss_gap": record.privacy_loss_gap,
            "tightness_ratio": record.tightness_ratio,
            "saturation_detected": record.saturation_detected,
        }
        for key in utility_keys:
            row[f"utility_{key}"] = record.utility_metrics.get(key)
        rows.append(row)
    return rows
