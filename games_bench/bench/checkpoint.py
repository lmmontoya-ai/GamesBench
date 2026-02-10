from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_VERSION = "v1"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "execution_state.json"


def compute_job_plan_hash(job_signatures: list[dict[str, Any]]) -> str:
    return _json_hash(job_signatures)


def build_execution_state(
    *,
    run_id: str,
    job_plan_hash: str,
    total_jobs: int,
    completed_episode_ids: list[int] | None = None,
) -> dict[str, Any]:
    completed = sorted(set(int(x) for x in (completed_episode_ids or [])))
    next_uncommitted = 0
    completed_set = set(completed)
    while next_uncommitted in completed_set:
        next_uncommitted += 1
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_id": str(run_id),
        "job_plan_hash": str(job_plan_hash),
        "total_jobs": int(total_jobs),
        "completed_episode_ids": completed,
        "next_uncommitted_episode_id": next_uncommitted,
        "last_updated": _now_iso_utc(),
    }


def load_execution_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid checkpoint format: {path}")
    return data


def save_execution_state(path: Path, state: dict[str, Any]) -> None:
    payload = dict(state)
    payload["last_updated"] = _now_iso_utc()
    path.write_text(json.dumps(payload, indent=2))


def update_execution_state(
    state: dict[str, Any],
    *,
    committed_episode_id: int,
) -> dict[str, Any]:
    updated = dict(state)
    completed = sorted(
        set(int(x) for x in updated.get("completed_episode_ids", []))
        | {int(committed_episode_id)}
    )
    updated["completed_episode_ids"] = completed

    next_uncommitted = 0
    completed_set = set(completed)
    while next_uncommitted in completed_set:
        next_uncommitted += 1
    updated["next_uncommitted_episode_id"] = next_uncommitted
    updated["last_updated"] = _now_iso_utc()
    return updated


def recover_jsonl_records(
    path: Path,
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if not data:
        return []

    lines = data.splitlines(keepends=True)
    valid_rows: list[dict[str, Any]] = []
    valid_bytes = 0

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            valid_bytes += len(raw_line)
            continue
        try:
            parsed = json.loads(line.decode("utf-8"))
        except Exception as exc:
            is_last = idx == (len(lines) - 1)
            if strict or not is_last:
                raise SystemExit(
                    f"Invalid JSONL while resuming: {path} line {idx + 1}: {exc}"
                ) from exc
            break
        if not isinstance(parsed, dict):
            raise SystemExit(
                f"Invalid JSONL row in {path} line {idx + 1}: expected object"
            )
        valid_rows.append(parsed)
        valid_bytes += len(raw_line)

    if valid_bytes < len(data):
        path.write_bytes(data[:valid_bytes])
    return valid_rows


def recover_text_log(path: Path, *, strict: bool) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data:
        return

    last_newline = data.rfind(b"\n")
    if last_newline == len(data) - 1:
        return

    if strict:
        raise SystemExit(f"Non-newline-terminated log during strict resume: {path}")

    if last_newline < 0:
        path.write_bytes(b"")
    else:
        path.write_bytes(data[: last_newline + 1])
