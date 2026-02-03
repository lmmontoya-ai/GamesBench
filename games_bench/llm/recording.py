from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class RecordingStep:
    index: int
    state_before: Any
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
    current_state: Any = None
    current_action: dict[str, Any] | None = None

    moves = 0
    illegal_moves = 0
    tool_calls = 0

    for event in events:
        event_type = event.get("type")
        if event_type == "state":
            current_state = event.get("state")
            continue
        if event_type == "tool_call":
            current_action = {
                "name": event.get("name"),
                "arguments": event.get("arguments", {}),
            }
            continue
        if event_type == "tool_result":
            tool_calls += 1
            result = event.get("result", {})
            ok = bool(result.get("ok", False))
            if ok:
                moves += 1
            else:
                illegal_moves += 1

            step = RecordingStep(
                index=len(steps),
                state_before=current_state,
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
    }

    return {
        "metadata": {
            **metadata,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "steps": [asdict(step) for step in steps],
    }
