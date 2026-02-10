from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable

from games_bench.bench.checkpoint import (
    build_execution_state,
    checkpoint_path,
    compute_job_plan_hash,
    load_execution_state,
    recover_jsonl_records,
    recover_text_log,
    save_execution_state,
    update_execution_state,
)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _episode_id_from_output(output: Any) -> int:
    value = getattr(output, "episode_id", None)
    if value is None and isinstance(output, dict):
        value = output.get("episode_id")
    episode_id = _safe_int(value)
    if episode_id is None:
        raise SystemExit("Episode output missing integer 'episode_id'.")
    return episode_id


def _variant_id_from_output(output: Any) -> str:
    value = getattr(output, "variant_id", None)
    if value is None and isinstance(output, dict):
        value = output.get("variant_id")
    return str(value or "unknown")


def _episode_dict_from_output(output: Any) -> dict[str, Any]:
    value = getattr(output, "episode", None)
    if value is None and isinstance(output, dict):
        value = output.get("episode")
    if not isinstance(value, dict):
        raise SystemExit("Episode output missing object 'episode'.")
    return dict(value)


def _events_from_output(output: Any) -> list[dict[str, Any]]:
    value = getattr(output, "events", None)
    if value is None and isinstance(output, dict):
        value = output.get("events")
    if isinstance(value, list):
        return [dict(v) if isinstance(v, dict) else {"value": v} for v in value]
    return []


def _raw_lines_from_output(output: Any) -> list[str]:
    value = getattr(output, "raw_lines", None)
    if value is None and isinstance(output, dict):
        value = output.get("raw_lines")
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _recording_from_output(output: Any) -> dict[str, Any] | None:
    value = getattr(output, "recording", None)
    if value is None and isinstance(output, dict):
        value = output.get("recording")
    if isinstance(value, dict):
        return value
    return None


def _job_signature(job: Any) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for key in ("episode_id", "variant_id", "run_idx"):
        if hasattr(job, key):
            signature[key] = getattr(job, key)

    level = getattr(job, "level", None)
    if level is not None:
        level_id = getattr(level, "level_id", None)
        if level_id is not None:
            signature["level_id"] = level_id

    case = getattr(job, "case", None)
    if case is not None:
        signature["case"] = str(case)

    if not signature:
        if is_dataclass(job):
            signature = {"repr": str(job)}
        else:
            signature = {"repr": repr(job)}
    return signature


def run_episode_jobs(
    *,
    out_dir: Path,
    run_id: str,
    jobs: list[Any],
    run_job: Callable[[Any], Any],
    parallelism: int,
    record: bool,
    record_raw: bool,
    progress_reporter: Any | None,
    resume: bool,
    strict_resume: bool,
    checkpoint_interval: int,
) -> list[dict[str, Any]]:
    episodes_path = out_dir / "episodes.jsonl"
    traces_path = out_dir / "traces.jsonl"
    raw_path = out_dir / "raw_generations.jsonl"
    recordings_dir = out_dir / "recordings"
    if record:
        recordings_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_interval < 1:
        raise SystemExit("checkpoint_interval must be >= 1")

    signatures = [_job_signature(job) for job in jobs]
    job_plan_hash = compute_job_plan_hash(signatures)
    state_file = checkpoint_path(out_dir)

    recovered_episodes: list[dict[str, Any]] = []
    completed_ids: set[int] = set()

    if resume:
        if not out_dir.exists():
            raise SystemExit(
                f"Resume requested but run directory does not exist: {out_dir}"
            )
        recovered_episodes = recover_jsonl_records(episodes_path, strict=strict_resume)
        recover_text_log(traces_path, strict=strict_resume)
        if record_raw:
            recover_text_log(raw_path, strict=strict_resume)

        has_existing_artifacts = (
            state_file.exists() or episodes_path.exists() or traces_path.exists()
        )
        if not has_existing_artifacts:
            raise SystemExit(
                "Resume requested but no existing run artifacts were found. "
                f"Expected checkpoint or JSONL files under {out_dir}."
            )

        for episode in recovered_episodes:
            episode_id = _safe_int(episode.get("episode_id"))
            if episode_id is None:
                raise SystemExit("Recovered episode is missing integer episode_id.")
            if episode_id in completed_ids and strict_resume:
                raise SystemExit(
                    "Duplicate episode_id detected in recovered episodes during strict resume."
                )
            completed_ids.add(episode_id)

        state = load_execution_state(state_file)
        if state is None:
            if not completed_ids:
                raise SystemExit(
                    "Resume requested but execution_state.json is missing and no "
                    "recoverable committed episodes were found."
                )
            state = build_execution_state(
                run_id=run_id,
                job_plan_hash=job_plan_hash,
                total_jobs=len(jobs),
                completed_episode_ids=sorted(completed_ids),
            )
            save_execution_state(state_file, state)
        else:
            existing_hash = str(state.get("job_plan_hash") or "")
            if existing_hash and existing_hash != job_plan_hash and strict_resume:
                raise SystemExit(
                    "Resume job plan mismatch with checkpoint. Use matching inputs or disable --strict-resume."
                )
            if str(state.get("run_id") or run_id) != str(run_id) and strict_resume:
                raise SystemExit(
                    "Resume run_id mismatch with checkpoint. Use matching --run-id or disable --strict-resume."
                )
            merged_completed = set(
                int(x) for x in state.get("completed_episode_ids", [])
            )
            merged_completed |= completed_ids
            state = build_execution_state(
                run_id=run_id,
                job_plan_hash=job_plan_hash,
                total_jobs=len(jobs),
                completed_episode_ids=sorted(merged_completed),
            )
            save_execution_state(state_file, state)
            completed_ids = set(state["completed_episode_ids"])
    else:
        # Fresh run: do not overwrite existing episode artifacts accidentally.
        if episodes_path.exists() and episodes_path.stat().st_size > 0:
            raise SystemExit(
                f"Run artifacts already exist in {out_dir}. Use --resume to continue or choose a different --run-id."
            )
        state = build_execution_state(
            run_id=run_id,
            job_plan_hash=job_plan_hash,
            total_jobs=len(jobs),
            completed_episode_ids=[],
        )
        save_execution_state(state_file, state)

    pending_jobs = [
        job
        for job in jobs
        if _safe_int(getattr(job, "episode_id", None)) not in completed_ids
    ]

    episodes: list[dict[str, Any]] = sorted(
        [dict(ep) for ep in recovered_episodes],
        key=lambda ep: int(ep.get("episode_id", 0)),
    )

    ep_mode = "a" if resume else "w"
    trace_mode = "a" if resume else "w"
    raw_mode = "a" if resume else "w"

    with (
        episodes_path.open(ep_mode) as ep_file,
        traces_path.open(trace_mode) as trace_file,
        (
            raw_path.open(raw_mode) if record_raw else open(os.devnull, raw_mode)
        ) as raw_file,
    ):
        commits_since_checkpoint = 0

        def commit_output(output: Any) -> None:
            nonlocal state, commits_since_checkpoint

            episode_id = _episode_id_from_output(output)
            variant_id = _variant_id_from_output(output)
            episode = _episode_dict_from_output(output)
            events = _events_from_output(output)
            raw_lines = _raw_lines_from_output(output)
            recording = _recording_from_output(output)

            if record and recording is not None:
                recording_path = recordings_dir / f"episode_{episode_id:04d}.json"
                recording_path.write_text(json.dumps(recording, indent=2))
                episode["recording_file"] = str(recording_path)

            ep_file.write(json.dumps(episode) + "\n")
            trace_file.write(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "variant_id": variant_id,
                        "events": events,
                    }
                )
                + "\n"
            )
            for line in raw_lines:
                raw_file.write(str(line) + "\n")

            episodes.append(episode)
            completed_ids.add(episode_id)

            if progress_reporter is not None:
                progress_reporter.on_episode_complete(episode)

            state = update_execution_state(state, committed_episode_id=episode_id)
            commits_since_checkpoint += 1
            if commits_since_checkpoint >= checkpoint_interval:
                save_execution_state(state_file, state)
                commits_since_checkpoint = 0

        if parallelism == 1:
            for job in pending_jobs:
                output = run_job(job)
                commit_output(output)
        else:
            pending_by_episode: dict[int, Any] = {}
            next_episode_id = 0
            while next_episode_id in completed_ids:
                next_episode_id += 1

            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                future_map = {
                    executor.submit(run_job, job): job for job in pending_jobs
                }
                for future in as_completed(future_map):
                    output = future.result()
                    output_episode_id = _episode_id_from_output(output)
                    pending_by_episode[output_episode_id] = output
                    while next_episode_id in pending_by_episode:
                        ordered = pending_by_episode.pop(next_episode_id)
                        commit_output(ordered)
                        next_episode_id += 1
                        while next_episode_id in completed_ids:
                            next_episode_id += 1

        if commits_since_checkpoint > 0:
            save_execution_state(state_file, state)

    return sorted(episodes, key=lambda ep: int(ep.get("episode_id", 0)))
