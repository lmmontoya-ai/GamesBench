from __future__ import annotations

import json

from games_bench.games.hanoi.env import HanoiToolbox, TowerOfHanoiEnv, tool_schemas


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


def main() -> int:
    env = TowerOfHanoiEnv(
        n_disks=3,
        record_history=True,
        illegal_action_behavior="penalize",
    )
    tools = HanoiToolbox(env)
    max_turns = 200

    schemas = tool_schemas(tool_prefix="hanoi")
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
            if env.is_solved():
                _print("Solved!")
                break
            if turns >= max_turns:
                _print("Stopped (max_turns reached).")
                break

            _print("State:")
            _print(
                env.format_prompt_state(
                    include_legal_moves=False, include_action_space=False
                )
            )
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

            if name == "hanoi_get_state":
                result = tools.get_state()
            elif name == "hanoi_move":
                result = tools.move(**arguments)
            elif name == "hanoi_reset":
                result = tools.reset(**arguments)
            elif name == "hanoi_is_solved":
                result = tools.is_solved()
            elif name == "hanoi_get_legal_moves":
                result = tools.get_legal_moves()
            elif name == "hanoi_step":
                result = tools.step(arguments.get("action"))
            else:
                result = {"ok": False, "error": f"unknown tool: {name}"}

            _print("Tool result:")
            _print(json.dumps(result, indent=2, sort_keys=True))
            _print("")
    except KeyboardInterrupt:
        _print("\nExiting.")

    _print(
        f"turns={turns} move_count={env.move_count} optimal={env.optimal_steps()} solved={env.is_solved()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
