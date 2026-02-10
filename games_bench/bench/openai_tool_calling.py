from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

from games_bench.bench.game_loader import build_env_and_adapter, parse_env_kwargs
from games_bench.llm import OpenAIResponsesProvider, run_tool_calling_episode


def _build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    if args.game == "hanoi":
        env_kwargs.setdefault("record_history", True)
        env_kwargs.setdefault("illegal_action_behavior", "penalize")
        if args.n_pegs is not None:
            env_kwargs["n_pegs"] = args.n_pegs
        if args.n_disks is not None:
            env_kwargs["n_disks"] = args.n_disks
    return env_kwargs


def _move_tool_names(adapter: Any) -> list[str]:
    names: list[str] = []
    for schema in adapter.tool_schemas():
        name = schema.get("name")
        if isinstance(name, str) and name.endswith("_move"):
            names.append(name)
    return names


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    parser = argparse.ArgumentParser(
        description="OpenAI tool-calling demo for a registered game."
    )
    parser.add_argument("--game", default="hanoi", help="Registered game name.")
    parser.add_argument(
        "--env-kwargs",
        default=None,
        help="JSON object of kwargs for the selected game's env factory.",
    )
    parser.add_argument(
        "--n-disks",
        type=int,
        default=3,
        help="Hanoi convenience flag (overrides env_kwargs.n_disks).",
    )
    parser.add_argument(
        "--n-pegs",
        type=int,
        default=3,
        help="Hanoi convenience flag (overrides env_kwargs.n_pegs).",
    )
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name (default: OPENAI_MODEL or gpt-4.1-mini).",
    )
    args = parser.parse_args()

    provider = OpenAIResponsesProvider(model=args.model)
    _env, adapter = build_env_and_adapter(args.game, env_kwargs=_build_env_kwargs(args))

    move_tools = _move_tool_names(adapter)
    if not move_tools:
        raise SystemExit(
            f"No *_move tool found for game '{args.game}'. Cannot run move-only demo."
        )

    instructions = (
        adapter.default_instructions()
        + f"\nUse ONLY the tool `{move_tools[0]}` to make moves. "
        "Use as few tool calls per turn as needed."
    )

    result = run_tool_calling_episode(
        adapter,
        provider,
        max_turns=args.max_turns,
        instructions=instructions,
        allowed_tools=[move_tools[0]],
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
