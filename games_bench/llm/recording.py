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
            ok = bool(result.get("ok", False))
            if ok:
                moves += 1
            else:
                illegal_moves += 1

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
