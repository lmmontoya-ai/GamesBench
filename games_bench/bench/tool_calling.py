from __future__ import annotations

import argparse
import json

from games_bench.bench.game_loader import build_env_and_adapter, parse_env_kwargs


def _build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    if args.game == "hanoi":
        env_kwargs.setdefault("illegal_action_behavior", "penalize")
        if args.n_disks is not None:
            env_kwargs["n_disks"] = args.n_disks
    return env_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tool-calling demo for a registered game."
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
        default=None,
        help="Hanoi convenience flag (overrides env_kwargs.n_disks).",
    )
    args = parser.parse_args()

    _env, adapter = build_env_and_adapter(args.game, env_kwargs=_build_env_kwargs(args))

    # Tool schemas you can register with an LLM tool-calling framework.
    schemas = adapter.tool_schemas()
    print("Tool schemas:\n", json.dumps(schemas, indent=2), "\n")

    # A prompt-friendly state snapshot.
    print("Prompt state:\n", adapter.format_state(), "\n")

    # Simulate "LLM tool calls".
    if args.game == "hanoi":
        move_1 = adapter.execute_tool("hanoi_move", {"from_peg": 0, "to_peg": 2}).result
        move_2 = adapter.execute_tool("hanoi_move", {"from_peg": 0, "to_peg": 2}).result
        legal = adapter.execute_tool("hanoi_get_legal_moves", {}).result
        step = adapter.execute_tool("hanoi_step", {"action": [0, 1]}).result
        print("hanoi_move(0 -> 2):", move_1)
        print("hanoi_move(0 -> 2) again (illegal):", move_2)
        print("hanoi_get_legal_moves():", legal)
        print("hanoi_step([0,1]):", step)
    else:
        print(
            f"Loaded adapter for '{args.game}'. Use manual-tool-loop for interactive calls."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
