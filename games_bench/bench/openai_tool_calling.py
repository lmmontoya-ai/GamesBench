from __future__ import annotations

import json
import os
from typing import Any

from games_bench.games.hanoi.env import HanoiToolbox, TowerOfHanoiEnv, tool_schemas
from games_bench.games.hanoi.prompts import default_instructions


def _to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": s["name"],
            "description": s.get("description", ""),
            "parameters": s["parameters"],
        }
        for s in schemas
    ]


def _model_dump(item: Any) -> Any:
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump()
    return item


def main() -> int:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency. Install with: uv sync --group llm\n"
            f"ImportError: {exc}"
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    n_disks = int(os.environ.get("HANOI_N_DISKS", "3"))
    max_moves = int(os.environ.get("HANOI_MAX_MOVES", "200"))

    env = TowerOfHanoiEnv(
        n_disks=n_disks, record_history=True, illegal_action_behavior="penalize"
    )
    toolbox = HanoiToolbox(env)

    # Keep the toolset minimal to encourage "move-only" behavior.
    schemas = [
        s for s in tool_schemas(tool_prefix="hanoi") if s["name"] == "hanoi_move"
    ]
    tools = _to_openai_tools(schemas)

    instructions = (
        default_instructions()
        + "\nUse ONLY the tool `hanoi_move` to make moves. Call exactly one tool per turn."
    )

    # We keep an input transcript so the model sees tool results (updated state).
    transcript: list[Any] = [
        {
            "role": "user",
            "content": env.format_prompt_state(
                include_legal_moves=False, include_action_space=False, compact_json=True
            ),
        }
    ]

    client = OpenAI()

    illegal_moves = 0
    while not env.is_solved() and env.move_count < max_moves:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
            input=[_model_dump(x) for x in transcript],
        )

        output_items = [_model_dump(x) for x in getattr(response, "output", [])]
        transcript.extend(output_items)

        function_calls = [
            x
            for x in output_items
            if isinstance(x, dict) and x.get("type") == "function_call"
        ]
        if not function_calls:
            # Should be rare with tool_choice="required", but handle gracefully.
            break

        for call in function_calls:
            if call.get("name") != "hanoi_move":
                result = {
                    "ok": False,
                    "error": f"unexpected tool call: {call.get('name')}",
                }
            else:
                args = json.loads(call.get("arguments", "{}"))
                result = toolbox.move(
                    from_peg=args.get("from_peg"), to_peg=args.get("to_peg")
                )
                if not result.get("ok", False):
                    illegal_moves += 1

            transcript.append(
                {
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": json.dumps(result),
                }
            )

    print(
        json.dumps(
            {
                "solved": env.is_solved(),
                "n_disks": env.n_disks,
                "move_count": env.move_count,
                "optimal_steps": env.optimal_steps(),
                "illegal_moves": illegal_moves,
                "history": env.history,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
