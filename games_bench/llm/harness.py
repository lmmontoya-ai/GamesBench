from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from games_bench.hanoi import HanoiToolbox, TowerOfHanoiEnv, tool_schemas

from .providers import ProviderResult, ToolCall


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    solved: bool
    n_disks: int
    move_count: int
    optimal_steps: int
    illegal_moves: int
    tool_calls: int
    history: list[tuple[int, int]]
    events: list[dict[str, Any]]
    usage: dict[str, float] | None
    cost: float | None


def default_instructions() -> str:
    return (
        "You are solving Tower of Hanoi.\n"
        "- Pegs are indexed 0, 1, 2.\n"
        "- Disks are integers; 1 is smallest.\n"
        "- Only move the top disk of a peg.\n"
        "- Never place a larger disk on a smaller disk.\n"
        "Call exactly one tool per turn."
    )


def _execute_tool(toolbox: HanoiToolbox, call: ToolCall) -> dict[str, Any]:
    name = call.name
    args = call.arguments
    if name == "hanoi_get_state":
        return toolbox.get_state()
    if name == "hanoi_move":
        return toolbox.move(**args)
    if name == "hanoi_reset":
        return toolbox.reset(**args)
    if name == "hanoi_is_solved":
        return toolbox.is_solved()
    if name == "hanoi_get_legal_moves":
        return toolbox.get_legal_moves()
    if name == "hanoi_step":
        return toolbox.step(args.get("action"))
    return {"ok": False, "error": f"unknown tool: {name}"}


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
    env: TowerOfHanoiEnv,
    provider: Any,
    *,
    max_turns: int = 200,
    tool_prefix: str = "hanoi",
    instructions: str | None = None,
    state_formatter: Callable[[TowerOfHanoiEnv], str] | None = None,
    allowed_tools: list[str] | None = None,
    record_provider_raw: bool = False,
) -> EpisodeResult:
    toolbox = HanoiToolbox(env)
    tools = tool_schemas(tool_prefix=tool_prefix)
    if allowed_tools is not None:
        if not allowed_tools:
            raise ValueError("allowed_tools must not be empty")
        allowed_set = set(allowed_tools)
        tools = [tool for tool in tools if tool["name"] in allowed_set]
    else:
        allowed_set = None

    instructions = instructions or default_instructions()
    state_formatter = state_formatter or (
        lambda e: e.format_prompt_state(
            include_legal_moves=False, include_action_space=False
        )
    )

    illegal_moves = 0
    tool_calls = 0
    events: list[dict[str, Any]] = []
    usage_totals: dict[str, float] = {}
    cost_total = 0.0
    cost_seen = False

    for _ in range(max_turns):
        if env.is_solved():
            break
        state_text = state_formatter(env)
        try:
            state_payload = json.loads(state_text)
        except json.JSONDecodeError:
            state_payload = state_text
        events.append({"type": "state", "state": state_payload})

        result: ProviderResult = provider.next_tool_calls(
            state_text=state_text, tool_schemas=tools, instructions=instructions
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
            illegal_moves += 1
        else:
            tool_result = _execute_tool(toolbox, call)
            if not tool_result.get("ok", False):
                illegal_moves += 1
        events.append(
            {"type": "tool_call", "name": call.name, "arguments": call.arguments}
        )
        events.append({"type": "tool_result", "result": tool_result})

    return EpisodeResult(
        solved=env.is_solved(),
        n_disks=env.n_disks,
        move_count=env.move_count,
        optimal_steps=env.optimal_steps(),
        illegal_moves=illegal_moves,
        tool_calls=tool_calls,
        history=list(env.history),
        events=events,
        usage=usage_totals or None,
        cost=cost_total if cost_seen else None,
    )
