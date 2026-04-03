from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AuditObservation:
    audit_regime: str
    auditor_variant: str
    epsilon_candidate: float | None = None
    ci_half_width: float | None = None
    raw_statistics: dict[str, float] = field(default_factory=dict)
    artifact_payload: dict[str, Any] = field(default_factory=dict)
    member_scores: list[float] = field(default_factory=list)
    nonmember_scores: list[float] = field(default_factory=list)
    score_name: str | None = None
    epsilon_upper_theory: float | None = None
    utility_metrics: dict[str, float] = field(default_factory=dict)
    audited_model_run_id: str | None = None
