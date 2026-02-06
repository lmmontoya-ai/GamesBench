from __future__ import annotations

from typing import Any

from games_bench.games.hanoi.env import HanoiToolbox, TowerOfHanoiEnv, tool_schemas
from games_bench.games.hanoi.prompts import default_instructions
from games_bench.llm.game_adapter import ToolExecution


class HanoiGameAdapter:
    """Benchmark-side adapter for running Hanoi with the generic LLM harness."""

    def __init__(
        self,
        env: TowerOfHanoiEnv,
        *,
        tool_prefix: str = "hanoi",
        instructions: str | None = None,
    ) -> None:
        self.env = env
        self._toolbox = HanoiToolbox(env)
        self._tool_prefix = tool_prefix
        self._instructions = instructions

    def tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(tool_prefix=self._tool_prefix)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        mutating = False
        illegal_action = False

        if name == f"{self._tool_prefix}_get_state":
            result = self._toolbox.get_state()
        elif name == f"{self._tool_prefix}_move":
            result = self._toolbox.move(**arguments)
            mutating = bool(result.get("ok", False))
            illegal_action = not mutating
        elif name == f"{self._tool_prefix}_reset":
            result = self._toolbox.reset(**arguments)
            mutating = bool(result.get("ok", False))
        elif name == f"{self._tool_prefix}_is_solved":
            result = self._toolbox.is_solved()
        elif name == f"{self._tool_prefix}_get_legal_moves":
            result = self._toolbox.get_legal_moves()
        elif name == f"{self._tool_prefix}_step":
            result = self._toolbox.step(arguments.get("action"))
            info = result.get("info")
            illegal_action = (
                bool(info.get("illegal_action")) if isinstance(info, dict) else False
            )
            mutating = bool(result.get("ok", False)) and not illegal_action
        else:
            result = {"ok": False, "error": f"unknown tool: {name}"}
            illegal_action = True

        return ToolExecution(
            result=result,
            meta={
                "state_mutating": mutating,
                "illegal_action": illegal_action,
            },
        )

    def get_state_snapshot(self) -> dict[str, Any]:
        return self.env.get_state().to_dict()

    def is_solved(self) -> bool:
        return self.env.is_solved()

    def default_instructions(self) -> str:
        if self._instructions is not None:
            return self._instructions
        return default_instructions(
            start_peg=self.env.start_peg, goal_peg=self.env.goal_peg
        )

    def format_state(self) -> str:
        return self.env.format_prompt_state(
            include_legal_moves=False, include_action_space=False
        )

    def episode_metrics(self) -> dict[str, Any]:
        return {
            "n_disks": self.env.n_disks,
            "move_count": self.env.move_count,
            "optimal_steps": self.env.optimal_steps(),
            "history": list(self.env.history),
        }
