from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re


def utc_timestamp_slug() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_id(prefix: str, descriptor: str, seed: int | None = None) -> str:
    clean_descriptor = re.sub(r"[^a-zA-Z0-9]+", "_", descriptor).strip("_").lower()
    suffix = utc_timestamp_slug()
    if seed is None:
        return f"{prefix}_{clean_descriptor}_{suffix}"
    return f"{prefix}_{clean_descriptor}_seed{seed}_{suffix}"


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

