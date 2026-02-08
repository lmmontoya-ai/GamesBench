from __future__ import annotations

import argparse
import json

from games_bench.bench.game_loader import build_env_and_adapter, parse_env_kwargs


def _print(obj: object) -> None:
    print(obj, flush=True)


def _read_json(prompt: str) -> dict:
    while True:
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            _print(f"Invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            _print("Expected a JSON object.")
            continue
        return data


def _build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    if args.game == "hanoi":
        env_kwargs.setdefault("record_history", True)
        env_kwargs.setdefault("illegal_action_behavior", "penalize")
        if args.n_pegs is not None:
            env_kwargs["n_pegs"] = args.n_pegs
        if args.n_disks is not None:
            env_kwargs["n_disks"] = args.n_disks
    return env_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive manual tool loop for a registered game."
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
    parser.add_argument(
        "--n-pegs",
        type=int,
        default=None,
        help="Hanoi convenience flag (overrides env_kwargs.n_pegs).",
    )
    parser.add_argument("--max-turns", type=int, default=200)
    args = parser.parse_args()

    _env, adapter = build_env_and_adapter(args.game, env_kwargs=_build_env_kwargs(args))
    max_turns = args.max_turns

    schemas = adapter.tool_schemas()
    _print("Tool schemas (paste into your tool-calling framework if needed):")
    _print(json.dumps(schemas, indent=2))
    _print("")

    _print("Manual tool loop:")
    _print("- Paste a single tool call as JSON.")
    _print('- Example: {"name":"hanoi_move","arguments":{"from_peg":0,"to_peg":2}}')
    _print("- Type 'q' to quit.")
    _print("")

    try:
        turns = 0
        while True:
            if adapter.is_solved():
                _print("Solved!")
                break
            if turns >= max_turns:
                _print("Stopped (max_turns reached).")
                break

            _print("State:")
            _print(adapter.format_state())
            _print("")

            call = _read_json("tool_call_json> ")
            turns += 1
            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(name, str):
                _print("Tool call must have a string field 'name'.")
                continue
            if not isinstance(arguments, dict):
                _print("Tool call must have an object field 'arguments'.")
                continue

            result = adapter.execute_tool(name, arguments).result

            _print("Tool result:")
            _print(json.dumps(result, indent=2, sort_keys=True))
            _print("")
    except KeyboardInterrupt:
        _print("\nExiting.")

    metrics = adapter.episode_metrics()
    _print(f"turns={turns} solved={adapter.is_solved()} metrics={json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
