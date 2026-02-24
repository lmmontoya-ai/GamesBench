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
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        token = value.strip()
        if token and token.lstrip("-").isdigit():
            return int(token)
    return None


def _episode_id_from_job(job: Any) -> int:
    episode_id = _safe_int(getattr(job, "episode_id", None))
    if episode_id is None:
        raise SystemExit("Episode job missing integer 'episode_id'.")
    return episode_id


def _collect_job_episode_ids(jobs: list[Any]) -> list[int]:
    episode_ids: list[int] = []
    seen_episode_ids: set[int] = set()
    for job in jobs:
        episode_id = _episode_id_from_job(job)
        if episode_id in seen_episode_ids:
            raise SystemExit(
                f"Duplicate episode_id detected in job plan: {episode_id}. "
                "Each job must have a unique episode_id."
            )
        seen_episode_ids.add(episode_id)
        episode_ids.append(episode_id)
    return episode_ids


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

    job_episode_ids = _collect_job_episode_ids(jobs)
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
            checkpoint_completed = set(
                int(x) for x in state.get("completed_episode_ids", [])
            )
            checkpoint_only_ids = sorted(checkpoint_completed - completed_ids)
            if checkpoint_only_ids:
                if strict_resume:
                    preview = ", ".join(str(x) for x in checkpoint_only_ids[:10])
                    if len(checkpoint_only_ids) > 10:
                        preview += ", ..."
                    raise SystemExit(
                        "Strict resume mismatch: execution_state.json references "
                        f"episode_id(s) missing from episodes.jsonl ({preview}). "
                        "This can happen if a run was interrupted before JSONL "
                        "buffers flushed. Re-run with --resume (non-strict) to "
                        "rewind to durable episodes and recompute missing outputs."
                    )
                merged_completed = set(completed_ids)
            else:
                merged_completed = set(checkpoint_completed)
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

    pending_job_pairs = [
        (job, episode_id)
        for job, episode_id in zip(jobs, job_episode_ids, strict=False)
        if episode_id not in completed_ids
    ]
    pending_episode_ids = [episode_id for _, episode_id in pending_job_pairs]

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

        def flush_artifacts_for_checkpoint() -> None:
            ep_file.flush()
            trace_file.flush()
            os.fsync(ep_file.fileno())
            os.fsync(trace_file.fileno())
            if record_raw:
                raw_file.flush()
                os.fsync(raw_file.fileno())

        def commit_output(output: Any) -> None:
            nonlocal state, commits_since_checkpoint

            episode_id = _episode_id_from_output(output)
            if episode_id in completed_ids:
                raise SystemExit(
                    f"Duplicate committed episode_id detected: {episode_id}. "
                    "Episode outputs must be unique."
                )
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
                flush_artifacts_for_checkpoint()
                save_execution_state(state_file, state)
                commits_since_checkpoint = 0

        if parallelism == 1:
            for job, expected_episode_id in pending_job_pairs:
                output = run_job(job)
                output_episode_id = _episode_id_from_output(output)
                if output_episode_id != expected_episode_id:
                    raise SystemExit(
                        "Episode output episode_id mismatch for serial execution: "
                        f"expected {expected_episode_id}, got {output_episode_id}."
                    )
                commit_output(output)
        else:
            pending_by_episode: dict[int, Any] = {}
            pending_index = 0

            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                future_map = {
                    executor.submit(run_job, job): expected_episode_id
                    for job, expected_episode_id in pending_job_pairs
                }
                for future in as_completed(future_map):
                    output = future.result()
                    expected_episode_id = future_map[future]
                    output_episode_id = _episode_id_from_output(output)
                    if output_episode_id != expected_episode_id:
                        raise SystemExit(
                            "Episode output episode_id mismatch for parallel execution: "
                            f"expected {expected_episode_id}, got {output_episode_id}."
                        )
                    if output_episode_id in pending_by_episode:
                        raise SystemExit(
                            f"Duplicate episode output detected for episode_id={output_episode_id}."
                        )
                    pending_by_episode[output_episode_id] = output
                    while pending_index < len(pending_episode_ids):
                        next_episode_id = pending_episode_ids[pending_index]
                        if next_episode_id not in pending_by_episode:
                            break
                        ordered = pending_by_episode.pop(next_episode_id)
                        commit_output(ordered)
                        pending_index += 1

                if pending_by_episode:
                    dangling_ids = ", ".join(str(k) for k in sorted(pending_by_episode))
                    raise SystemExit(
                        "Executor failed to commit all pending episode outputs. "
                        f"Uncommitted episode_id(s): {dangling_ids}"
                    )

        if commits_since_checkpoint > 0:
            flush_artifacts_for_checkpoint()
            save_execution_state(state_file, state)

    return sorted(episodes, key=lambda ep: int(ep.get("episode_id", 0)))
