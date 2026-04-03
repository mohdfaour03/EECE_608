from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("dp_audit_tightness")


def format_kv_fields(**fields: Any) -> str:
    return " ".join(f"{key}={value}" for key, value in fields.items())

