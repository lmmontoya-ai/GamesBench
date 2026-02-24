from __future__ import annotations

from typing import Any


TAXONOMY_VERSION = "taxonomy-v1"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def classify_episode(
    episode: dict[str, Any],
    *,
    game_name: str | None = None,
) -> tuple[str, list[str]]:
    solved = bool(episode.get("solved"))
    tags: list[str] = []

    reason = str(episode.get("termination_reason") or "").strip().lower()
    max_turns = _as_int(episode.get("max_turns_effective", episode.get("max_turns")))
    turn_count = _as_int(episode.get("turn_count"))
    move_count = _as_int(episode.get("move_count"))
    tool_calls = _as_int(episode.get("tool_calls"))
    illegal_moves = _as_int(episode.get("illegal_moves"))
    provider_error_count = _as_int(episode.get("provider_error_count"))

    outcome_code = "solved" if solved else "failed_unknown"

    if reason.startswith("stagnation"):
        tags.append("stagnation_stop")
        if not solved:
            outcome_code = "failed_stagnation"

    if reason.startswith("loop:"):
        tags.append("loop_stop")
        if not solved:
            outcome_code = "failed_loop"

    if reason == "deadlock_terminal" or reason.startswith("deadlock:"):
        tags.append("deadlock_terminal")
        if reason.startswith("deadlock:"):
            tags.append("deadlock_patience_stop")
        if not solved:
            outcome_code = "failed_deadlock_terminal"

    if not solved and max_turns is not None and turn_count is not None:
        if turn_count >= max_turns:
            tags.append("turn_budget_exhausted")
            if outcome_code == "failed_unknown":
                outcome_code = "failed_budget"

    if provider_error_count is not None and provider_error_count > 0:
        tags.append("provider_error")
        if not solved:
            outcome_code = "failed_provider"

    if illegal_moves is not None and illegal_moves >= 3:
        tags.append("illegal_action_burst")

    if (
        not solved
        and tool_calls is not None
        and tool_calls >= 3
        and (move_count is not None and move_count == 0)
    ):
        tags.append("query_loop")

    if game_name == "sokoban" or "deadlocked" in episode:
        if bool(episode.get("deadlocked", False)):
            tags.append("deadlocked_final")

    if not solved:
        tags.append("unsolved_final")

    # Preserve order but remove duplicates.
    deduped_tags: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped_tags.append(tag)

    return outcome_code, deduped_tags


def annotate_episode_with_taxonomy(
    episode: dict[str, Any],
    *,
    game_name: str | None = None,
) -> dict[str, Any]:
    enriched = dict(episode)
    outcome_code, failure_tags = classify_episode(enriched, game_name=game_name)
    enriched["taxonomy_version"] = TAXONOMY_VERSION
    enriched["outcome_code"] = outcome_code
    enriched["failure_tags"] = failure_tags
    return enriched
