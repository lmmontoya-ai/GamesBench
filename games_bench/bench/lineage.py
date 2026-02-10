from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_git(args: list[str]) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return output or None


def _git_metadata() -> dict[str, Any] | None:
    inside = _run_git(["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        return None

    status = _run_git(["status", "--porcelain"]) or ""
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": bool(status.strip()),
        "remote_origin": _run_git(["config", "--get", "remote.origin.url"]),
    }


def _bench_version() -> str | None:
    try:
        return importlib.metadata.version("games-bench")
    except Exception:
        return None


def _seed_lineage(run_config: dict[str, Any]) -> dict[str, Any] | None:
    lineage: dict[str, Any] = {}

    for key in ("seed", "random_seed", "procgen_seed"):
        if key in run_config:
            lineage[key] = run_config[key]

    procgen = run_config.get("procgen")
    if isinstance(procgen, dict) and procgen.get("enabled"):
        lineage["procgen"] = {
            "mode": procgen.get("mode"),
            "seed": procgen.get("seed"),
            "grid_sizes": procgen.get("grid_sizes"),
            "box_counts": procgen.get("box_counts"),
        }

    return lineage or None


def build_run_manifest(
    *,
    run_config: dict[str, Any],
    game_config: dict[str, Any] | None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)

    suite_fields = {
        "spec_base": run_config.get("spec_base"),
        "spec": run_config.get("spec"),
        "interaction_mode": run_config.get("interaction_mode"),
        "stateless": run_config.get("stateless"),
    }

    return {
        "run_manifest_version": "v1",
        "run_id": run_config.get("run_id"),
        "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
        "created_at_unix": int(now.timestamp()),
        "game": run_config.get("game"),
        "provider": run_config.get("provider"),
        "model": run_config.get("model"),
        "spec": run_config.get("spec"),
        "interaction_mode": run_config.get("interaction_mode"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_metadata(),
        "bench_version": _bench_version(),
        "parent_run_id": parent_run_id,
        "hashes": {
            "run_config_hash": _sha256_json(run_config),
            "game_config_hash": _sha256_json(game_config or {}),
            "suite_hash": _sha256_json(suite_fields),
            "prompt_hash": _sha256_json(run_config.get("prompt_variants", [])),
            "tool_schema_hash": _sha256_json(run_config.get("tool_schemas", [])),
        },
        "seed_lineage": _seed_lineage(run_config),
    }


def write_run_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
