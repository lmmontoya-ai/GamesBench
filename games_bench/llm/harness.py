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
    terminated_early: bool = False
    termination_reason: str | None = None

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
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, (int, float)):
        prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, (int, float)):
        completion_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if isinstance(prompt_tokens, (int, float)):
        total["prompt_tokens"] = total.get("prompt_tokens", 0.0) + float(prompt_tokens)
    if isinstance(completion_tokens, (int, float)):
        total["completion_tokens"] = total.get("completion_tokens", 0.0) + float(
            completion_tokens
        )
    if isinstance(total_tokens, (int, float)):
        total["total_tokens"] = total.get("total_tokens", 0.0) + float(total_tokens)


def _snapshot_key(snapshot: Any) -> str:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)


def _infer_deadlock_checker(
    adapter: GameAdapter,
) -> Callable[[GameAdapter], bool] | None:
    env = getattr(adapter, "env", None)
    if env is None:
        return None
    if not bool(getattr(env, "detect_deadlocks", False)):
        return None
    checker = getattr(env, "is_deadlocked", None)
    if not callable(checker):
        return None
    return lambda _adapter, _checker=checker: bool(_checker())


def _infer_deadlock_terminate_on_check(adapter: GameAdapter) -> bool:
    env = getattr(adapter, "env", None)
    if env is None:
        return False
    return bool(getattr(env, "detect_deadlocks", False)) and bool(
        getattr(env, "terminal_on_deadlock", False)
    )


def _state_message_content(
    *,
    state_text: str,
    state_image: dict[str, Any] | None,
) -> str | list[dict[str, Any]]:
    if not state_image:
        return state_text
    parts: list[dict[str, Any]] = []
    if state_text:
        parts.append({"type": "text", "text": state_text})
    data_url = state_image.get("data_url")
    if not data_url:
        mime = state_image.get("mime_type", "image/png")
        data = state_image.get("data_base64")
        if data:
            data_url = f"data:{mime};base64,{data}"
    if data_url:
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    return parts or state_text


def _state_image_meta(state_image: dict[str, Any] | None) -> dict[str, Any] | None:
    if not state_image:
        return None
    return {
        "mime_type": state_image.get("mime_type"),
        "width": state_image.get("width"),
        "height": state_image.get("height"),
    }


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def run_tool_calling_episode(
    adapter: GameAdapter,
    provider: Any,
    *,
    max_turns: int = 200,
    instructions: str | None = None,
    state_formatter: Callable[[GameAdapter], str] | None = None,
    trace_state_formatter: Callable[[GameAdapter], str] | None = None,
    state_image_renderer: Callable[[GameAdapter], dict[str, Any]] | None = None,
    allowed_tools: list[str] | None = None,
    record_provider_raw: bool = False,
    stagnation_patience: int | None = None,
    deadlock_patience: int | None = None,
    deadlock_checker: Callable[[GameAdapter], bool] | None = None,
    deadlock_terminate_on_check: bool | None = None,
    stateless: bool = False,
    max_tool_calls_per_turn: int = 1,
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
    trace_state_formatter = trace_state_formatter or state_formatter
    if state_image_renderer and not getattr(provider, "supports_images", False):
        raise ValueError("Provider does not support image inputs.")
    if stagnation_patience is not None and int(stagnation_patience) < 1:
        raise ValueError("stagnation_patience must be >= 1 when provided.")
    if deadlock_patience is not None and int(deadlock_patience) < 1:
        raise ValueError("deadlock_patience must be >= 1 when provided.")
    if int(max_tool_calls_per_turn) < 1:
        raise ValueError("max_tool_calls_per_turn must be >= 1.")
    deadlock_checker = deadlock_checker or _infer_deadlock_checker(adapter)
    terminate_on_deadlock_check = (
        bool(deadlock_terminate_on_check)
        if deadlock_terminate_on_check is not None
        else _infer_deadlock_terminate_on_check(adapter)
    )

    illegal_moves = 0
    tool_calls = 0
    events: list[dict[str, Any]] = []
    usage_totals: dict[str, float] = {}
    cost_total = 0.0
    cost_seen = False
    last_snapshot: str | None = None
    stagnation_turns = 0
    deadlock_turns = 0
    terminated_early = False
    termination_reason: str | None = None
    conversation: list[dict[str, Any]] | None = None
    if not stateless:
        conversation = [{"role": "system", "content": instructions}]

    for turn_index in range(max_turns):
        if adapter.is_solved():
            break
        state_text = state_formatter(adapter)
        trace_state_text = trace_state_formatter(adapter)
        state_image = state_image_renderer(adapter) if state_image_renderer else None
        snapshot = adapter.get_state_snapshot()
        snapshot_key = _snapshot_key(snapshot)
        if last_snapshot is not None and snapshot_key == last_snapshot:
            stagnation_turns += 1
        else:
            stagnation_turns = 0
        last_snapshot = snapshot_key

        is_deadlocked = False
        if deadlock_checker is not None:
            is_deadlocked = bool(deadlock_checker(adapter))
        if is_deadlocked:
            deadlock_turns += 1
        else:
            deadlock_turns = 0

        try:
            state_payload = json.loads(state_text)
        except json.JSONDecodeError:
            state_payload = state_text
        events.append(
            {
                "type": "state_snapshot",
                "state": snapshot,
                "turn_index": turn_index,
            }
        )
        events.append(
            {
                "type": "state",
                "state": state_payload,
                "state_text": state_text,
                "trace_state_text": trace_state_text,
                "turn_index": turn_index,
            }
        )
        state_image_meta = _state_image_meta(state_image)
        if state_image_meta:
            events.append(
                {
                    "type": "state_image",
                    "meta": state_image_meta,
                    "turn_index": turn_index,
                }
            )
        if terminate_on_deadlock_check and is_deadlocked:
            terminated_early = True
            termination_reason = "deadlock_terminal"
            events.append(
                {
                    "type": "early_stop",
                    "reason": termination_reason,
                    "stagnation_turns": stagnation_turns,
                    "deadlock_turns": deadlock_turns,
                    "turn_index": turn_index,
                }
            )
            break
        if stagnation_patience is not None and stagnation_turns >= int(
            stagnation_patience
        ):
            terminated_early = True
            termination_reason = f"stagnation:{stagnation_turns}"
            events.append(
                {
                    "type": "early_stop",
                    "reason": termination_reason,
                    "stagnation_turns": stagnation_turns,
                    "deadlock_turns": deadlock_turns,
                    "turn_index": turn_index,
                }
            )
            break
        if deadlock_patience is not None and deadlock_turns >= int(deadlock_patience):
            terminated_early = True
            termination_reason = f"deadlock:{deadlock_turns}"
            events.append(
                {
                    "type": "early_stop",
                    "reason": termination_reason,
                    "stagnation_turns": stagnation_turns,
                    "deadlock_turns": deadlock_turns,
                    "turn_index": turn_index,
                }
            )
            break

        if conversation is not None:
            conversation.append(
                {
                    "role": "user",
                    "content": _state_message_content(
                        state_text=state_text,
                        state_image=state_image,
                    ),
                }
            )
        result: ProviderResult = provider.next_tool_calls(
            state_text=state_text,
            tool_schemas=tools,
            instructions=instructions,
            state_image=state_image,
            conversation=list(conversation) if conversation is not None else None,
        )
        if result.usage:
            _accumulate_usage(usage_totals, result.usage)
        if result.cost is not None:
            cost_total += result.cost
            cost_seen = True

        provider_event = {
            "type": "provider_result",
            "error": result.error,
            "turn_index": turn_index,
        }
        if result.usage:
            provider_event["usage"] = result.usage
        if result.cost is not None:
            provider_event["cost"] = result.cost
        if record_provider_raw:
            provider_event["raw"] = result.raw
        events.append(provider_event)

        if conversation is not None and result.error:
            conversation.append(
                {"role": "assistant", "content": _compact_json({"error": result.error})}
            )
        if result.error or not result.tool_calls:
            break

        requested_calls = list(result.tool_calls)
        executed_calls = requested_calls[: int(max_tool_calls_per_turn)]
        if len(requested_calls) > len(executed_calls):
            events.append(
                {
                    "type": "tool_calls_truncated",
                    "requested": len(requested_calls),
                    "executed": len(executed_calls),
                    "max_tool_calls_per_turn": int(max_tool_calls_per_turn),
                    "turn_index": turn_index,
                }
            )

        stop_turn = False
        for action_index, call in enumerate(executed_calls):
            if conversation is not None:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": _compact_json(
                            {
                                "tool_call": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                }
                            }
                        ),
                    }
                )
            tool_calls += 1
            if allowed_set is not None and call.name not in allowed_set:
                tool_result = {"ok": False, "error": f"tool not allowed: {call.name}"}
                tool_meta = {
                    "state_mutating": False,
                    "illegal_action": True,
                    "action_kind": "query",
                    "counts_as_move": False,
                }
                illegal_moves += 1
            else:
                execution = adapter.execute_tool(call.name, call.arguments)
                tool_result = execution.result
                tool_meta = execution.meta
                if "illegal_action" in tool_meta:
                    if bool(tool_meta.get("illegal_action")):
                        illegal_moves += 1
                elif not tool_result.get("ok", False):
                    illegal_moves += 1
            events.append(
                {
                    "type": "tool_call",
                    "name": call.name,
                    "arguments": call.arguments,
                    "turn_index": turn_index,
                    "action_index": action_index,
                }
            )
            events.append(
                {
                    "type": "tool_result",
                    "result": tool_result,
                    "meta": tool_meta,
                    "turn_index": turn_index,
                    "action_index": action_index,
                }
            )

            # Capture a canonical post-action state trail for multi-action turns.
            action_state_snapshot = adapter.get_state_snapshot()
            action_state_text = trace_state_formatter(adapter)
            events.append(
                {
                    "type": "action_state",
                    "state": action_state_snapshot,
                    "state_text": action_state_text,
                    "tool_name": call.name,
                    "turn_index": turn_index,
                    "action_index": action_index,
                }
            )
            if state_image_renderer is not None:
                action_state_image = state_image_renderer(adapter)
                action_state_image_meta = _state_image_meta(action_state_image)
                if action_state_image_meta:
                    events.append(
                        {
                            "type": "action_state_image",
                            "meta": action_state_image_meta,
                            "tool_name": call.name,
                            "turn_index": turn_index,
                            "action_index": action_index,
                        }
                    )
            if conversation is not None:
                conversation.append(
                    {
                        "role": "user",
                        "content": _compact_json(
                            {"tool_result": tool_result, "tool_meta": tool_meta}
                        ),
                    }
                )
            if bool(tool_meta.get("terminate_episode")):
                terminated_early = True
                termination_reason = str(
                    tool_meta.get("termination_reason", "adapter_requested")
                )
                events.append(
                    {
                        "type": "early_stop",
                        "reason": termination_reason,
                        "stagnation_turns": stagnation_turns,
                        "deadlock_turns": deadlock_turns,
                        "turn_index": turn_index,
                        "action_index": action_index,
                    }
                )
                stop_turn = True
                break
            if adapter.is_solved():
                stop_turn = True
                break
        if stop_turn:
            break

    return EpisodeResult(
        solved=adapter.is_solved(),
        game_metrics=adapter.episode_metrics(),
        illegal_moves=illegal_moves,
        tool_calls=tool_calls,
        events=events,
        usage=usage_totals or None,
        cost=cost_total if cost_seen else None,
        terminated_early=terminated_early,
        termination_reason=termination_reason,
    )
