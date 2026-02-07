from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class RecordingStep:
    index: int
    state_before: Any
    state_text: Any | None
    action: dict[str, Any] | None
    state_after: Any
    legal: bool
    totals: dict[str, int]


def _count_action_outcome(
    action: dict[str, Any] | None,
    result: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Return (move_delta, illegal_move_delta) for a tool result.

    Current recordings are Hanoi-oriented. We only count mutating tools:
      - *_move: ok=true => move, ok=false => illegal move
      - *_step: use info.illegal_action when available
    Query tools never affect move totals.
    """

    if isinstance(meta, dict):
        if "counts_as_move" in meta:
            move_delta = 1 if bool(meta.get("counts_as_move")) else 0
            illegal_delta = 1 if bool(meta.get("illegal_action")) else 0
            return (move_delta, illegal_delta)
        if "illegal_action" in meta:
            if bool(meta.get("illegal_action")):
                return (0, 1)
            if bool(meta.get("state_mutating")):
                return (1, 0)
            if "state_mutating" in meta:
                return (0, 0)
            return (0, 0)
        if bool(meta.get("illegal_action")):
            return (0, 1)
        if bool(meta.get("state_mutating")):
            return (1, 0)
        if "illegal_action" in meta or "state_mutating" in meta:
            return (0, 0)

    name = action.get("name") if isinstance(action, dict) else None
    if not isinstance(name, str):
        return (0, 0)

    ok = bool(result.get("ok", False))
    if name.endswith("_move"):
        return (1, 0) if ok else (0, 1)

    if name.endswith("_step"):
        info = result.get("info")
        illegal_action = (
            bool(info.get("illegal_action")) if isinstance(info, dict) else False
        )
        if illegal_action:
            return (0, 1)
        return (1, 0) if ok else (0, 1)

    return (0, 0)


def build_recording(
    *,
    events: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    steps: list[RecordingStep] = []
    current_state_prompt: Any = None
    current_state_snapshot: Any = None
    current_action: dict[str, Any] | None = None
    initial_state_prompt: Any = None
    initial_state_snapshot: Any = None

    moves = 0
    illegal_moves = 0
    tool_calls = 0

    for event in events:
        event_type = event.get("type")
        if event_type == "state_snapshot":
            current_state_snapshot = event.get("state")
            if initial_state_snapshot is None:
                initial_state_snapshot = current_state_snapshot
            continue
        if event_type == "state":
            current_state_prompt = event.get("state_text", event.get("state"))
            if initial_state_prompt is None:
                initial_state_prompt = current_state_prompt
            continue
        if event_type == "tool_call":
            current_action = {
                "name": event.get("name"),
                "arguments": event.get("arguments", {}),
            }
            continue
        if event_type == "tool_result":
            if not steps and (
                initial_state_snapshot is not None or initial_state_prompt is not None
            ):
                initial_state = (
                    initial_state_snapshot
                    if initial_state_snapshot is not None
                    else initial_state_prompt
                )
                steps.append(
                    RecordingStep(
                        index=0,
                        state_before=initial_state,
                        state_text=initial_state_prompt,
                        action=None,
                        state_after=initial_state,
                        legal=True,
                        totals={
                            "moves": moves,
                            "illegal_moves": illegal_moves,
                            "tool_calls": tool_calls,
                        },
                    )
                )

            tool_calls += 1
            result = event.get("result", {})
            move_delta, illegal_delta = _count_action_outcome(
                current_action,
                result,
                meta=event.get("meta"),
            )
            moves += move_delta
            illegal_moves += illegal_delta
            ok = bool(result.get("ok", False))

            state_before = (
                current_state_snapshot
                if current_state_snapshot is not None
                else current_state_prompt
            )
            step = RecordingStep(
                index=len(steps),
                state_before=state_before,
                state_text=current_state_prompt,
                action=current_action,
                state_after=result.get("state"),
                legal=ok,
                totals={
                    "moves": moves,
                    "illegal_moves": illegal_moves,
                    "tool_calls": tool_calls,
                },
            )
            steps.append(step)
            current_action = None

    summary = {
        "solved": bool(metadata.get("solved", False)),
        "total_moves": moves,
        "total_illegal_moves": illegal_moves,
        "total_tool_calls": tool_calls,
        "initial_state_included": bool(steps and steps[0].action is None),
    }

    return {
        "metadata": {
            **metadata,
            "initial_state": initial_state_snapshot,
            "initial_state_text": initial_state_prompt,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "steps": [asdict(step) for step in steps],
    }
