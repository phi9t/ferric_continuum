from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping


def new_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def create_run_dir(repo_root: Path, run_id: str) -> Path:
    run_dir = repo_root / ".ferric" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, data: Mapping[str, object]) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def write_text(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
