from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from games_bench.games.adapter import GameAdapter
from .providers import ProviderResult


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    solved: bool
    game_metrics: dict[str, Any]
    illegal_moves: int
    tool_calls: int
    events: list[dict[str, Any]]
    usage: dict[str, float] | None
    cost: float | None

    @property
    def move_count(self) -> int:
        value = self.game_metrics.get("move_count", 0)
        return int(value) if isinstance(value, (int, float)) else 0

    @property
    def n_disks(self) -> int | None:
        value = self.game_metrics.get("n_disks")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def optimal_steps(self) -> int | None:
        value = self.game_metrics.get("optimal_steps")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def history(self) -> list[Any]:
        value = self.game_metrics.get("history")
        if isinstance(value, list):
            return value
        return []


def _accumulate_usage(total: dict[str, float], usage: dict[str, Any] | None) -> None:
    if not usage:
        return
    mapping = {
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
    }
    for key, target in mapping.items():
        value = usage.get(key)
        if isinstance(value, (int, float)):
            total[target] = total.get(target, 0.0) + float(value)


def run_tool_calling_episode(
    adapter: GameAdapter,
    provider: Any,
    *,
    max_turns: int = 200,
    instructions: str | None = None,
    state_formatter: Callable[[GameAdapter], str] | None = None,
    state_image_renderer: Callable[[GameAdapter], dict[str, Any]] | None = None,
    allowed_tools: list[str] | None = None,
    record_provider_raw: bool = False,
) -> EpisodeResult:
    tools = adapter.tool_schemas()
    if allowed_tools is not None:
        if not allowed_tools:
            raise ValueError("allowed_tools must not be empty")
        allowed_set = set(allowed_tools)
        tools = [tool for tool in tools if tool["name"] in allowed_set]
    else:
        allowed_set = None

    instructions = instructions or adapter.default_instructions()
    state_formatter = state_formatter or (lambda a: a.format_state())
    if state_image_renderer and not getattr(provider, "supports_images", False):
        raise ValueError("Provider does not support image inputs.")

    illegal_moves = 0
    tool_calls = 0
    events: list[dict[str, Any]] = []
    usage_totals: dict[str, float] = {}
    cost_total = 0.0
    cost_seen = False

    for _ in range(max_turns):
        if adapter.is_solved():
            break
        state_text = state_formatter(adapter)
        state_image = state_image_renderer(adapter) if state_image_renderer else None
        snapshot = adapter.get_state_snapshot()
        try:
            state_payload = json.loads(state_text)
        except json.JSONDecodeError:
            state_payload = state_text
        events.append({"type": "state_snapshot", "state": snapshot})
        events.append(
            {"type": "state", "state": state_payload, "state_text": state_text}
        )
        if state_image:
            meta = {
                "mime_type": state_image.get("mime_type"),
                "width": state_image.get("width"),
                "height": state_image.get("height"),
            }
            events.append({"type": "state_image", "meta": meta})

        result: ProviderResult = provider.next_tool_calls(
            state_text=state_text,
            tool_schemas=tools,
            instructions=instructions,
            state_image=state_image,
        )
        if result.usage:
            _accumulate_usage(usage_totals, result.usage)
        if result.cost is not None:
            cost_total += result.cost
            cost_seen = True

        provider_event = {"type": "provider_result", "error": result.error}
        if result.usage:
            provider_event["usage"] = result.usage
        if result.cost is not None:
            provider_event["cost"] = result.cost
        if record_provider_raw:
            provider_event["raw"] = result.raw
        events.append(provider_event)

        if result.error or not result.tool_calls:
            break

        # Only execute the first tool call per turn for consistency.
        call = result.tool_calls[0]
        tool_calls += 1
        if allowed_set is not None and call.name not in allowed_set:
            tool_result = {"ok": False, "error": f"tool not allowed: {call.name}"}
            tool_meta = {"state_mutating": False, "illegal_action": True}
            illegal_moves += 1
        else:
            execution = adapter.execute_tool(call.name, call.arguments)
            tool_result = execution.result
            tool_meta = execution.meta
            if tool_meta.get("illegal_action", False) or not tool_result.get(
                "ok", False
            ):
                illegal_moves += 1
        events.append(
            {"type": "tool_call", "name": call.name, "arguments": call.arguments}
        )
        events.append({"type": "tool_result", "result": tool_result, "meta": tool_meta})

    return EpisodeResult(
        solved=adapter.is_solved(),
        game_metrics=adapter.episode_metrics(),
        illegal_moves=illegal_moves,
        tool_calls=tool_calls,
        events=events,
        usage=usage_totals or None,
        cost=cost_total if cost_seen else None,
    )
