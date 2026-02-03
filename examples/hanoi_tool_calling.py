from __future__ import annotations

import json

from games_bench.hanoi import HanoiToolbox, TowerOfHanoiEnv, tool_schemas


def main() -> None:
    env = TowerOfHanoiEnv(n_disks=3, illegal_action_behavior="penalize")
    tools = HanoiToolbox(env)

    # Tool schemas you can register with an LLM tool-calling framework.
    schemas = tool_schemas(tool_prefix="hanoi")
    print("Tool schemas:\n", json.dumps(schemas, indent=2), "\n")

    # A prompt-friendly state snapshot.
    print("Prompt state:\n", env.format_prompt_state(), "\n")

    # Simulate "LLM tool calls".
    print("hanoi_move(0 -> 2):", tools.move(from_peg=0, to_peg=2))
    print("hanoi_move(0 -> 2) again (illegal):", tools.move(from_peg=0, to_peg=2))
    print("hanoi_get_legal_moves():", tools.get_legal_moves())

    # RL-style tool call:
    print("hanoi_step([0,1]):", tools.step([0, 1]))


if __name__ == "__main__":
    main()
