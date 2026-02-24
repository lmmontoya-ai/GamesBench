from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


RUN_RECORD_VERSION = "site-run-v1"
TRAJECTORY_VERSION = "trajectory-v1"
TRAJECTORY_INDEX_VERSION = "trajectory-index-v1"
SITE_INDEX_VERSION = "site-index-v1"
LEADERBOARD_VERSION = "leaderboard-v1"
MODEL_GAME_POINTER_VERSION = "by-model-game-v1"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON object in file: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL file: {path}:{lineno}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid JSONL row at {path}:{lineno}: expected object")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-._")
    return text or "unknown"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        candidate = float(value)
        if candidate == candidate and candidate not in {float("inf"), float("-inf")}:
            return candidate
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_run_key(
    *,
    game: str,
    spec: str,
    interaction_mode: str,
    provider: str,
    model: str,
) -> str:
    return "__".join(
        [
            _slug(game),
            _slug(spec),
            _slug(interaction_mode),
            _slug(provider),
            _slug(model),
        ]
    )


def _read_run_artifacts(run_dir: Path) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    run_dir = run_dir.resolve()
    run_config = _read_json(run_dir / "run_config.json")
    summary = _read_json(run_dir / "summary.json")
    episodes = _read_jsonl(run_dir / "episodes.jsonl")

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    return run_config, summary, episodes, manifest


def _resolve_identity(
    run_config: dict[str, Any],
    summary: dict[str, Any],
    *,
    run_dir: Path,
) -> dict[str, str]:
    game = _nonempty_str(run_config.get("game")) or _nonempty_str(summary.get("game"))
    spec = _nonempty_str(run_config.get("spec")) or _nonempty_str(summary.get("spec"))
    interaction_mode = _nonempty_str(
        run_config.get("interaction_mode")
    ) or _nonempty_str(summary.get("interaction_mode"))
    provider = _nonempty_str(run_config.get("provider")) or _nonempty_str(
        summary.get("provider")
    )
    model = _nonempty_str(run_config.get("model")) or _nonempty_str(
        summary.get("model")
    )

    missing: list[str] = []
    if game is None:
        missing.append("game")
    if spec is None:
        missing.append("spec")
    if interaction_mode is None:
        missing.append("interaction_mode")
    if provider is None:
        missing.append("provider")
    if model is None:
        missing.append("model")
    if missing:
        raise SystemExit(
            "Missing required identity fields in run artifacts "
            f"for {run_dir}: {', '.join(missing)}"
        )

    return {
        "game": str(game),
        "spec": str(spec),
        "interaction_mode": str(interaction_mode),
        "provider": str(provider),
        "model": str(model),
    }


def _coverage_metrics(
    episodes: list[dict[str, Any]],
    *,
    overall: dict[str, Any],
) -> dict[str, Any]:
    episodes_total = len(episodes)
    usage_count = sum(1 for ep in episodes if isinstance(ep.get("usage"), dict))
    cost_count = sum(1 for ep in episodes if ep.get("cost") is not None)

    token_totals = overall.get("token_totals")
    cost_total_value = overall.get("cost_total")

    total_tokens = None
    if isinstance(token_totals, dict):
        total_tokens = _as_float(token_totals.get("total_tokens"))
        if total_tokens is None:
            prompt = _as_float(token_totals.get("prompt_tokens"))
            completion = _as_float(token_totals.get("completion_tokens"))
            if prompt is not None and completion is not None:
                total_tokens = prompt + completion

    if total_tokens is None and usage_count:
        prompt_sum = 0.0
        completion_sum = 0.0
        found_any = False
        for ep in episodes:
            usage = ep.get("usage")
            if not isinstance(usage, dict):
                continue
            prompt = _as_float(usage.get("prompt_tokens") or usage.get("input_tokens"))
            completion = _as_float(
                usage.get("completion_tokens") or usage.get("output_tokens")
            )
            total = _as_float(usage.get("total_tokens"))
            if total is not None:
                prompt_sum += total
                found_any = True
            elif prompt is not None and completion is not None:
                prompt_sum += prompt + completion
                found_any = True
            else:
                if prompt is not None:
                    prompt_sum += prompt
                    found_any = True
                if completion is not None:
                    completion_sum += completion
                    found_any = True
        candidate_total = prompt_sum + completion_sum
        total_tokens = candidate_total if found_any else None

    cost_total = _as_float(cost_total_value)
    if cost_total is None and cost_count:
        running_total = 0.0
        seen = False
        for ep in episodes:
            cost = _as_float(ep.get("cost"))
            if cost is None:
                continue
            running_total += cost
            seen = True
        cost_total = running_total if seen else None

    solved = _as_float(overall.get("solved"))
    if solved is None:
        solved = float(sum(1 for ep in episodes if bool(ep.get("solved"))))

    token_coverage_rate = (
        float(usage_count) / float(episodes_total) if episodes_total else 0.0
    )
    cost_coverage_rate = (
        float(cost_count) / float(episodes_total) if episodes_total else 0.0
    )

    tokens_per_solved = None
    if solved > 0 and total_tokens is not None:
        tokens_per_solved = total_tokens / solved

    cost_per_solved = None
    if solved > 0 and cost_total is not None:
        cost_per_solved = cost_total / solved

    return {
        "episodes_total": episodes_total,
        "episodes_with_usage": usage_count,
        "episodes_with_cost": cost_count,
        "token_coverage_rate": token_coverage_rate,
        "cost_coverage_rate": cost_coverage_rate,
        "tokens_per_solved": tokens_per_solved,
        "cost_per_solved": cost_per_solved,
    }


def _taxonomy_rollups(
    episodes: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, int]]:
    outcome_counts: dict[str, int] = {}
    failure_tag_counts: dict[str, int] = {}

    for ep in episodes:
        outcome = _nonempty_str(ep.get("outcome_code")) or "unknown"
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        tags = ep.get("failure_tags")
        if isinstance(tags, list):
            for raw_tag in tags:
                tag = _nonempty_str(raw_tag)
                if tag is None:
                    continue
                failure_tag_counts[tag] = failure_tag_counts.get(tag, 0) + 1

    return outcome_counts, failure_tag_counts


def _validate_release_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(
            "release_date must use YYYY-MM-DD format, " f"received: {value!r}"
        ) from exc
    return parsed.isoformat()


def pack_run_record(
    *,
    run_dir: Path,
    release_id: str,
    release_date: str,
    out_root: Path,
) -> Path:
    run_config, summary, episodes, manifest = _read_run_artifacts(run_dir)
    identity = _resolve_identity(run_config, summary, run_dir=run_dir)

    run_key = _build_run_key(**identity)
    run_id = _nonempty_str(run_config.get("run_id")) or run_dir.name
    git_commit = None
    if manifest is not None:
        git_payload = manifest.get("git")
        if isinstance(git_payload, dict):
            git_commit = _nonempty_str(git_payload.get("commit"))

    overall = summary.get("overall")
    if not isinstance(overall, dict):
        raise SystemExit(
            f"Invalid summary format: expected object at overall in {run_dir}"
        )

    variants = summary.get("variants")
    if variants is None:
        variants = {}
    if not isinstance(variants, dict):
        raise SystemExit(
            f"Invalid summary format: expected object at variants in {run_dir}"
        )

    coverage = _coverage_metrics(episodes, overall=overall)
    outcome_counts, failure_tag_counts = _taxonomy_rollups(episodes)

    payload = {
        "record_version": RUN_RECORD_VERSION,
        "generated_at": _now_iso_utc(),
        "release_id": release_id,
        "release_date": release_date,
        "run_key": run_key,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "game": identity["game"],
        "spec_base": _nonempty_str(run_config.get("spec_base"))
        or _nonempty_str(summary.get("spec_base")),
        "spec": identity["spec"],
        "interaction_mode": identity["interaction_mode"],
        "provider": identity["provider"],
        "model": identity["model"],
        "model_slug": _slug(identity["model"]),
        "score_version": _nonempty_str(summary.get("score_version")),
        "taxonomy_version": _nonempty_str(summary.get("taxonomy_version")),
        "scored_at": _nonempty_str(summary.get("scored_at")),
        "git_commit": git_commit,
        "overall": overall,
        "variants": variants,
        "derived": coverage,
        "outcome_counts": outcome_counts,
        "failure_tag_counts": failure_tag_counts,
    }

    out_path = out_root.resolve() / release_id / f"{run_key}.json"
    return _write_json(out_path, payload)


def _coerce_turn_index(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _coerce_action_index(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _extract_text_from_response_content(content: Any) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = _nonempty_str(item.get("type"))
            if item_type == "text":
                text_value = _nonempty_str(item.get("text"))
                if text_value:
                    chunks.append(text_value)
        joined = "\n".join(chunks).strip()
        return joined or None
    return None


def _extract_text_from_provider_raw(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return _extract_text_from_provider_raw(parsed)

    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    text = _extract_text_from_response_content(message.get("content"))
                    if text:
                        return text

        output_text = _nonempty_str(raw.get("output_text"))
        if output_text:
            return output_text

        output = raw.get("output")
        if isinstance(output, list):
            output_chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if _nonempty_str(item.get("type")) == "message":
                    text = _extract_text_from_response_content(item.get("content"))
                    if text:
                        output_chunks.append(text)
                        continue
                text_candidate = _nonempty_str(item.get("text"))
                if text_candidate:
                    output_chunks.append(text_candidate)
            joined_output = "\n".join(output_chunks).strip()
            if joined_output:
                return joined_output

        for key in ("text", "content", "message"):
            text = _extract_text_from_response_content(raw.get(key))
            if text:
                return text

    return None


def _extract_reasoning(events: list[dict[str, Any]]) -> tuple[str | None, str]:
    chunks: list[str] = []
    for event in events:
        if _nonempty_str(event.get("type")) != "provider_result":
            continue
        text = _extract_text_from_provider_raw(event.get("raw"))
        if text:
            chunks.append(text)
    if not chunks:
        return None, "none"
    return "\n\n".join(chunks), "provider_raw"


def _extract_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: dict[tuple[int, int], dict[str, Any]] = {}
    first_seen: dict[tuple[int, int], int] = {}
    current_state: Any = None
    order_counter = 0

    for event in events:
        event_type = _nonempty_str(event.get("type"))
        if event_type == "state_snapshot":
            current_state = event.get("state")
            continue
        if event_type == "state" and current_state is None:
            current_state = event.get("state")
            continue

        if event_type in {"tool_call", "tool_result", "action_state"}:
            turn_index = _coerce_turn_index(event.get("turn_index"))
            action_index = _coerce_action_index(event.get("action_index"))
            key = (turn_index, action_index)
            if key not in actions:
                actions[key] = {
                    "turn_index": turn_index,
                    "action_index": action_index,
                    "state_before": current_state,
                    "action": None,
                    "tool_result": None,
                    "state_after": None,
                }
                first_seen[key] = order_counter
                order_counter += 1
            step = actions[key]

            if event_type == "tool_call":
                step["action"] = {
                    "name": event.get("name"),
                    "arguments": event.get("arguments"),
                }
                if step.get("state_before") is None:
                    step["state_before"] = current_state
                continue

            if event_type == "tool_result":
                step["tool_result"] = event.get("result")
                continue

            if event_type == "action_state":
                step["state_after"] = event.get("state")
                if step.get("state_after") is not None:
                    current_state = step["state_after"]
                continue

    sorted_steps = sorted(
        actions.values(),
        key=lambda row: (
            int(row.get("turn_index", 0)),
            int(row.get("action_index", 0)),
            first_seen.get(
                (
                    int(row.get("turn_index", 0)),
                    int(row.get("action_index", 0)),
                ),
                0,
            ),
        ),
    )

    last_state: Any = None
    final_steps: list[dict[str, Any]] = []
    for step in sorted_steps:
        state_before = step.get("state_before")
        if state_before is None:
            state_before = last_state

        state_after = step.get("state_after")
        if state_after is None:
            result = step.get("tool_result")
            if isinstance(result, dict):
                state_after = result.get("state")

        if state_after is not None:
            last_state = state_after

        final_steps.append(
            {
                "turn_index": int(step.get("turn_index", 0)),
                "state_before": state_before,
                "action": step.get("action"),
                "tool_result": step.get("tool_result"),
                "state_after": state_after,
            }
        )

    return final_steps


def _hanoi_solved_efficiency(ep: dict[str, Any]) -> tuple[float, float]:
    move_count = _as_float(ep.get("move_count"))
    optimal_steps = _as_float(ep.get("optimal_steps"))
    if move_count is not None and optimal_steps is not None and optimal_steps > 0:
        return (move_count / optimal_steps, move_count)
    if move_count is not None:
        return (move_count, move_count)
    return (float("inf"), float("inf"))


def _sokoban_solved_efficiency(ep: dict[str, Any]) -> tuple[float, float]:
    move_ratio = _as_float(ep.get("move_ratio"))
    move_count = _as_float(ep.get("move_count"))
    if move_ratio is not None:
        return (move_ratio, move_count if move_count is not None else float("inf"))
    if move_count is not None:
        return (move_count, move_count)
    return (float("inf"), float("inf"))


def _hanoi_progress(ep: dict[str, Any]) -> tuple[float, float]:
    move_count = _as_float(ep.get("move_count"))
    optimal_steps = _as_float(ep.get("optimal_steps"))
    if move_count is not None and optimal_steps is not None and optimal_steps > 0:
        return (move_count / optimal_steps, move_count)
    if move_count is not None:
        return (move_count, move_count)
    return (0.0, 0.0)


def _sokoban_progress(ep: dict[str, Any]) -> tuple[float, float]:
    ratio = _as_float(ep.get("boxes_on_goals_ratio"))
    move_count = _as_float(ep.get("move_count"))
    return (
        ratio if ratio is not None else 0.0,
        move_count if move_count is not None else 0.0,
    )


def _hard_failure_rank(ep: dict[str, Any]) -> int:
    outcome = _nonempty_str(ep.get("outcome_code")) or "failed_unknown"
    ranking = {
        "failed_deadlock_terminal": 0,
        "failed_budget": 1,
        "failed_stagnation": 2,
        "failed_unknown": 3,
        "failed_provider": 4,
    }
    return ranking.get(outcome, 5)


def _select_trajectory_episodes(
    *,
    episodes: list[dict[str, Any]],
    steps_by_episode: dict[int, list[dict[str, Any]]],
    game: str,
    max_trajectories: int,
) -> list[tuple[str, dict[str, Any]]]:
    if max_trajectories < 1:
        return []

    sorted_episodes = sorted(
        episodes,
        key=lambda ep: (
            _as_int(ep.get("episode_id"))
            if _as_int(ep.get("episode_id")) is not None
            else 10**9
        ),
    )
    solved = [ep for ep in sorted_episodes if bool(ep.get("solved"))]
    failures = [ep for ep in sorted_episodes if not bool(ep.get("solved"))]

    selected: list[tuple[str, dict[str, Any]]] = []
    selected_ids: set[int] = set()

    def _episode_id(ep: dict[str, Any]) -> int:
        episode_id = _as_int(ep.get("episode_id"))
        return episode_id if episode_id is not None else 10**9

    def _append(slot: str, ep: dict[str, Any]) -> None:
        episode_id = _episode_id(ep)
        if episode_id in selected_ids:
            return
        selected.append((slot, ep))
        selected_ids.add(episode_id)

    if solved:
        if game == "sokoban":
            best_solved = min(
                solved,
                key=lambda ep: (*_sokoban_solved_efficiency(ep), _episode_id(ep)),
            )
        else:
            best_solved = min(
                solved,
                key=lambda ep: (*_hanoi_solved_efficiency(ep), _episode_id(ep)),
            )
        _append("best_solved", best_solved)

    if failures:
        if game == "sokoban":
            highest_progress_failure = max(
                failures,
                key=lambda ep: (*_sokoban_progress(ep), -float(_episode_id(ep))),
            )
        else:
            highest_progress_failure = max(
                failures,
                key=lambda ep: (*_hanoi_progress(ep), -float(_episode_id(ep))),
            )
        _append("highest_progress_failure", highest_progress_failure)

    remaining_failures = [ep for ep in failures if _episode_id(ep) not in selected_ids]
    if remaining_failures:
        representative = sorted(
            remaining_failures,
            key=lambda ep: (
                _hard_failure_rank(ep),
                -len(steps_by_episode.get(_episode_id(ep), [])),
                _episode_id(ep),
            ),
        )[0]
        _append("representative_hard_failure", representative)

    if len(selected) < max_trajectories:
        remaining = [
            ep for ep in sorted_episodes if _episode_id(ep) not in selected_ids
        ]
        for ep in remaining:
            slot = f"extra_{len(selected) + 1}"
            _append(slot, ep)
            if len(selected) >= max_trajectories:
                break

    return selected[:max_trajectories]


def pack_trajectories(
    *,
    run_dir: Path,
    release_id: str,
    out_root: Path,
    max_trajectories: int,
) -> Path:
    run_dir = run_dir.resolve()
    run_config, summary, episodes, _manifest = _read_run_artifacts(run_dir)
    identity = _resolve_identity(run_config, summary, run_dir=run_dir)
    run_key = _build_run_key(**identity)

    traces_path = run_dir / "traces.jsonl"
    if not traces_path.exists():
        raise SystemExit(f"Missing required file: {traces_path}")
    trace_rows = _read_jsonl(traces_path)

    trace_by_episode: dict[int, dict[str, Any]] = {}
    for row in trace_rows:
        episode_id = _as_int(row.get("episode_id"))
        if episode_id is None:
            continue
        trace_by_episode[episode_id] = row

    steps_by_episode: dict[int, list[dict[str, Any]]] = {}
    reasoning_by_episode: dict[int, tuple[str | None, str]] = {}
    for ep in episodes:
        episode_id = _as_int(ep.get("episode_id"))
        if episode_id is None:
            continue
        trace = trace_by_episode.get(episode_id, {})
        events = trace.get("events") if isinstance(trace, dict) else None
        if isinstance(events, list):
            typed_events = [event for event in events if isinstance(event, dict)]
        else:
            typed_events = []
        steps_by_episode[episode_id] = _extract_steps(typed_events)
        reasoning_by_episode[episode_id] = _extract_reasoning(typed_events)

    selected = _select_trajectory_episodes(
        episodes=episodes,
        steps_by_episode=steps_by_episode,
        game=identity["game"],
        max_trajectories=max_trajectories,
    )

    out_dir = out_root.resolve() / release_id / run_key
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    for slot, episode in selected:
        episode_id = _as_int(episode.get("episode_id"))
        if episode_id is None:
            continue
        steps = steps_by_episode.get(episode_id, [])
        reasoning, reasoning_source = reasoning_by_episode.get(
            episode_id, (None, "none")
        )

        episode_filename = f"episode_{episode_id:04d}.json"
        episode_payload = {
            "trajectory_version": TRAJECTORY_VERSION,
            "release_id": release_id,
            "run_key": run_key,
            "slot": slot,
            "episode_id": episode_id,
            "variant_id": episode.get("variant_id"),
            "outcome_code": episode.get("outcome_code"),
            "failure_tags": (
                episode.get("failure_tags")
                if isinstance(episode.get("failure_tags"), list)
                else []
            ),
            "solved": bool(episode.get("solved")),
            "usage": (
                episode.get("usage") if isinstance(episode.get("usage"), dict) else None
            ),
            "cost": _as_float(episode.get("cost")),
            "reasoning": reasoning,
            "reasoning_source": reasoning_source,
            "steps": steps,
        }
        _write_json(out_dir / episode_filename, episode_payload)
        index_rows.append(
            {
                "slot": slot,
                "episode_id": episode_id,
                "variant_id": episode.get("variant_id"),
                "solved": bool(episode.get("solved")),
                "path": episode_filename,
            }
        )

    index_payload = {
        "index_version": TRAJECTORY_INDEX_VERSION,
        "generated_at": _now_iso_utc(),
        "release_id": release_id,
        "run_key": run_key,
        "game": identity["game"],
        "spec": identity["spec"],
        "interaction_mode": identity["interaction_mode"],
        "provider": identity["provider"],
        "model": identity["model"],
        "max_trajectories": max_trajectories,
        "selected": index_rows,
    }
    return _write_json(out_dir / "index.json", index_payload)


def _iter_release_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records

    for path in sorted(root.rglob("*.json")):
        if path.name == "index.json":
            continue
        try:
            payload = _read_json(path)
        except SystemExit:
            continue
        if payload.get("record_version") != RUN_RECORD_VERSION:
            continue
        payload["_source_path"] = str(path)
        records.append(payload)
    return records


def _iter_trajectory_indexes(root: Path) -> list[dict[str, Any]]:
    indexes: list[dict[str, Any]] = []
    if not root.exists():
        return indexes

    for path in sorted(root.rglob("index.json")):
        try:
            payload = _read_json(path)
        except SystemExit:
            continue
        if payload.get("index_version") != TRAJECTORY_INDEX_VERSION:
            continue
        payload["_source_path"] = str(path)
        indexes.append(payload)
    return indexes


def _release_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    release_date = _nonempty_str(record.get("release_date")) or "0000-00-00"
    scored_at = _nonempty_str(record.get("scored_at")) or ""
    run_key = _nonempty_str(record.get("run_key")) or ""
    return (release_date, scored_at, run_key)


def _validate_release_record(record: dict[str, Any]) -> list[str]:
    required = [
        "release_id",
        "release_date",
        "run_key",
        "game",
        "spec",
        "interaction_mode",
        "provider",
        "model",
        "overall",
    ]
    missing: list[str] = []
    for key in required:
        value = record.get(key)
        if key not in record or value is None or (isinstance(value, str) and not value):
            missing.append(key)
    if record.get("overall") is not None and not isinstance(
        record.get("overall"), dict
    ):
        missing.append("overall(object)")
    return missing


def _sort_leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _metric(row: dict[str, Any], key: str, default: float) -> float:
        overall = row.get("overall")
        if not isinstance(overall, dict):
            return default
        value = _as_float(overall.get(key))
        return value if value is not None else default

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -_metric(row, "solve_rate", 0.0),
            _metric(row, "avg_illegal_moves", float("inf")),
            _metric(row, "avg_tool_calls", float("inf")),
            _nonempty_str(row.get("model")) or "",
        ),
    )
    ranked: list[dict[str, Any]] = []
    for idx, row in enumerate(sorted_rows, start=1):
        with_rank = dict(row)
        with_rank["rank"] = idx
        ranked.append(with_rank)
    return ranked


def build_site_data(
    *,
    input_releases: Path,
    input_trajectories: Path,
    output_dir: Path,
    strict: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    runs_dir = output_dir / "runs"
    trajectories_dir = output_dir / "trajectories"
    by_model_game_dir = output_dir / "by_model_game"

    records = _iter_release_records(input_releases.resolve())
    if strict and not records:
        raise SystemExit(
            "No release records discovered. "
            f"Expected records under: {input_releases.resolve()}"
        )

    valid_records: list[dict[str, Any]] = []
    for record in records:
        missing = _validate_release_record(record)
        if missing:
            message = (
                "Invalid release record "
                f"{record.get('_source_path')}: missing {', '.join(sorted(set(missing)))}"
            )
            if strict:
                raise SystemExit(message)
            print(message, file=sys.stderr)
            continue
        valid_records.append(record)

    valid_records.sort(key=_release_sort_key, reverse=True)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_index_map: dict[tuple[str, str], dict[str, Any]] = {}
    for index_payload in _iter_trajectory_indexes(input_trajectories.resolve()):
        release_id = _nonempty_str(index_payload.get("release_id"))
        run_key = _nonempty_str(index_payload.get("run_key"))
        if not release_id or not run_key:
            if strict:
                raise SystemExit(
                    f"Invalid trajectory index metadata: {index_payload.get('_source_path')}"
                )
            continue

        source_index_path = Path(str(index_payload.get("_source_path")))
        source_dir = source_index_path.parent
        target_dir = trajectories_dir / release_id / run_key
        target_dir.mkdir(parents=True, exist_ok=True)

        selected_rows = index_payload.get("selected")
        selected_rows = selected_rows if isinstance(selected_rows, list) else []

        for row in selected_rows:
            if not isinstance(row, dict):
                continue
            rel_path = _nonempty_str(row.get("path"))
            if not rel_path:
                continue
            src_file = source_dir / rel_path
            if not src_file.exists():
                if strict:
                    raise SystemExit(
                        f"Trajectory file missing for index {source_index_path}: {src_file}"
                    )
                continue
            dst_file = target_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

        target_index_path = target_dir / "index.json"
        _write_json(
            target_index_path,
            {k: v for k, v in index_payload.items() if not str(k).startswith("_")},
        )
        trajectory_index_map[(release_id, run_key)] = index_payload

    run_index_rows: list[dict[str, Any]] = []
    for record in valid_records:
        release_id = str(record["release_id"])
        run_key = str(record["run_key"])
        out_path = runs_dir / release_id / f"{run_key}.json"
        cleaned = {k: v for k, v in record.items() if not str(k).startswith("_")}
        _write_json(out_path, cleaned)

        run_index_rows.append(
            {
                "release_id": release_id,
                "release_date": record.get("release_date"),
                "run_key": run_key,
                "game": record.get("game"),
                "spec": record.get("spec"),
                "interaction_mode": record.get("interaction_mode"),
                "provider": record.get("provider"),
                "model": record.get("model"),
                "model_slug": record.get("model_slug") or _slug(record.get("model")),
                "scored_at": record.get("scored_at"),
                "run_record_path": str(Path("runs") / release_id / f"{run_key}.json"),
                "overall": record.get("overall"),
                "derived": record.get("derived"),
            }
        )

    index_payload = {
        "index_version": SITE_INDEX_VERSION,
        "generated_at": _now_iso_utc(),
        "run_count": len(run_index_rows),
        "runs": run_index_rows,
    }
    _write_json(output_dir / "index.json", index_payload)

    latest_by_identity: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for record in valid_records:
        identity = (
            str(record.get("game")),
            str(record.get("spec")),
            str(record.get("interaction_mode")),
            str(record.get("provider")),
            str(record.get("model")),
        )
        existing = latest_by_identity.get(identity)
        if existing is None:
            latest_by_identity[identity] = record
            continue
        if _release_sort_key(record) > _release_sort_key(existing):
            latest_by_identity[identity] = record

    leaderboard_rows: list[dict[str, Any]] = []
    for record in latest_by_identity.values():
        release_id = str(record["release_id"])
        run_key = str(record["run_key"])
        leaderboard_rows.append(
            {
                "release_id": release_id,
                "release_date": record.get("release_date"),
                "run_key": run_key,
                "game": record.get("game"),
                "spec": record.get("spec"),
                "interaction_mode": record.get("interaction_mode"),
                "provider": record.get("provider"),
                "model": record.get("model"),
                "model_slug": record.get("model_slug") or _slug(record.get("model")),
                "scored_at": record.get("scored_at"),
                "overall": record.get("overall"),
                "derived": record.get("derived"),
                "run_record_path": str(Path("runs") / release_id / f"{run_key}.json"),
            }
        )

    leaderboard_payload = {
        "leaderboard_version": LEADERBOARD_VERSION,
        "generated_at": _now_iso_utc(),
        "entry_count": len(leaderboard_rows),
        "entries": _sort_leaderboard_rows(leaderboard_rows),
    }
    _write_json(output_dir / "leaderboard.json", leaderboard_payload)

    latest_by_model_game: dict[tuple[str, str], dict[str, Any]] = {}
    for record in valid_records:
        model = str(record.get("model"))
        game = str(record.get("game"))
        key = (_slug(model), _slug(game))
        existing = latest_by_model_game.get(key)
        if existing is None or _release_sort_key(record) > _release_sort_key(existing):
            latest_by_model_game[key] = record

    pointer_count = 0
    for (model_slug, game_slug), record in latest_by_model_game.items():
        release_id = str(record["release_id"])
        run_key = str(record["run_key"])
        selected_files: list[str] = []
        trajectory_index_rel: str | None = None

        trajectory_index = trajectory_index_map.get((release_id, run_key))
        if trajectory_index is not None:
            trajectory_index_rel = str(
                Path("trajectories") / release_id / run_key / "index.json"
            )
            selected = trajectory_index.get("selected")
            if isinstance(selected, list):
                for row in selected:
                    if not isinstance(row, dict):
                        continue
                    rel_name = _nonempty_str(row.get("path"))
                    if not rel_name:
                        continue
                    selected_files.append(
                        str(Path("trajectories") / release_id / run_key / rel_name)
                    )

        pointer_payload = {
            "pointer_version": MODEL_GAME_POINTER_VERSION,
            "generated_at": _now_iso_utc(),
            "model": record.get("model"),
            "model_slug": model_slug,
            "game": record.get("game"),
            "release_id": release_id,
            "run_key": run_key,
            "run_record_path": str(Path("runs") / release_id / f"{run_key}.json"),
            "trajectory_index_path": trajectory_index_rel,
            "trajectory_paths": selected_files,
        }
        _write_json(
            by_model_game_dir / model_slug / game_slug / "latest.json", pointer_payload
        )
        pointer_count += 1

    return {
        "run_count": len(valid_records),
        "leaderboard_count": len(leaderboard_rows),
        "trajectory_run_count": len(trajectory_index_map),
        "model_game_pointer_count": pointer_count,
        "output_dir": str(output_dir),
    }


def _add_pack_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "pack",
        help="Package one scored run into a tracked release record.",
    )
    parser.add_argument("--run-dir", required=True, help="Scored run directory.")
    parser.add_argument(
        "--release-id",
        required=True,
        help="Release identifier (for example: gpt-4.2-2026-02).",
    )
    parser.add_argument(
        "--release-date",
        required=True,
        help="Release date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--out-root",
        default="bench_results/releases",
        help="Output root for tracked release records.",
    )


def _add_pack_trajectories_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "pack-trajectories",
        help="Select and package curated trajectories for one run.",
    )
    parser.add_argument(
        "--run-dir", required=True, help="Run directory with traces.jsonl."
    )
    parser.add_argument(
        "--release-id",
        required=True,
        help="Release identifier matching the packaged run record.",
    )
    parser.add_argument(
        "--out-root",
        default="bench_results/trajectories",
        help="Output root for curated trajectory payloads.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=3,
        help="Maximum number of trajectories to publish (default: 3).",
    )


def _add_build_site_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "build-site",
        help="Build site-ready JSON payloads from tracked release records.",
    )
    parser.add_argument(
        "--input-releases",
        default="bench_results/releases",
        help="Input root containing packaged release records.",
    )
    parser.add_argument(
        "--input-trajectories",
        default="bench_results/trajectories",
        help="Input root containing packaged curated trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for site-ready JSON data.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed or missing required inputs.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and publish GamesBench website data artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_pack_parser(subparsers)
    _add_pack_trajectories_parser(subparsers)
    _add_build_site_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "pack":
        release_date = _validate_release_date(str(args.release_date))
        out_path = pack_run_record(
            run_dir=Path(args.run_dir),
            release_id=str(args.release_id),
            release_date=release_date,
            out_root=Path(args.out_root),
        )
        print(
            json.dumps(
                {
                    "record_path": str(out_path),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "pack-trajectories":
        max_trajectories = int(args.max_trajectories)
        if max_trajectories < 1:
            raise SystemExit("max-trajectories must be >= 1")
        index_path = pack_trajectories(
            run_dir=Path(args.run_dir),
            release_id=str(args.release_id),
            out_root=Path(args.out_root),
            max_trajectories=max_trajectories,
        )
        index_payload = _read_json(index_path)
        selected = index_payload.get("selected")
        selected_count = len(selected) if isinstance(selected, list) else 0
        print(
            json.dumps(
                {
                    "index_path": str(index_path),
                    "selected_count": selected_count,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "build-site":
        payload = build_site_data(
            input_releases=Path(args.input_releases),
            input_trajectories=Path(args.input_trajectories),
            output_dir=Path(args.output_dir),
            strict=bool(args.strict),
        )
        print(json.dumps(payload, indent=2))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
