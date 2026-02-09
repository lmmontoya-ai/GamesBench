from __future__ import annotations

from typing import Any

from games_bench.games.adapter import ToolExecution
from games_bench.games.sokoban.env import SokobanEnv, SokobanToolbox, tool_schemas
from games_bench.games.sokoban.prompts import default_instructions


class SokobanGameAdapter:
    """Game-side adapter for running Sokoban with the generic LLM harness."""

    def __init__(
        self,
        env: SokobanEnv,
        *,
        tool_prefix: str = "sokoban",
        instructions: str | None = None,
    ) -> None:
        self.env = env
        self._toolbox = SokobanToolbox(env)
        self._tool_prefix = tool_prefix
        self._instructions = instructions

    def tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(tool_prefix=self._tool_prefix)

    def _state_payload(self) -> dict[str, Any]:
        state = self.env.get_state().to_dict()
        return {
            "state": state,
            "boxes_on_goals": self.env.boxes_on_goals,
            "total_goals": self.env.n_boxes,
        }

    def _error_result(self, message: str) -> dict[str, Any]:
        return {"ok": False, "error": message, **self._state_payload()}

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        if name == f"{self._tool_prefix}_move":
            direction = arguments.get("direction")
            result = self._toolbox.move(direction)
            ok = bool(result.get("ok", False))
            deadlocked = bool(result.get("deadlocked", False))
            terminate_on_deadlock = (
                ok
                and deadlocked
                and self.env.detect_deadlocks
                and self.env.terminal_on_deadlock
                and not self.env.is_solved()
            )
            return ToolExecution(
                result=result,
                meta={
                    "state_mutating": ok,
                    "illegal_action": not ok,
                    "action_kind": "move",
                    "counts_as_move": ok,
                    "terminate_episode": terminate_on_deadlock,
                    "termination_reason": (
                        "deadlock_terminal" if terminate_on_deadlock else None
                    ),
                },
            )

        if name == f"{self._tool_prefix}_undo":
            result = self._toolbox.undo()
            ok = bool(result.get("ok", False))
            return ToolExecution(
                result=result,
                meta={
                    "state_mutating": ok,
                    "illegal_action": False,
                    "action_kind": "undo",
                    "counts_as_move": False,
                },
            )

        if name == f"{self._tool_prefix}_get_state":
            result = self._toolbox.get_state()
            return ToolExecution(
                result=result,
                meta={
                    "state_mutating": False,
                    "illegal_action": False,
                    "action_kind": "query",
                    "counts_as_move": False,
                },
            )

        if name == f"{self._tool_prefix}_is_solved":
            result = self._toolbox.is_solved()
            return ToolExecution(
                result=result,
                meta={
                    "state_mutating": False,
                    "illegal_action": False,
                    "action_kind": "query",
                    "counts_as_move": False,
                },
            )

        if name == f"{self._tool_prefix}_get_legal_moves":
            result = self._toolbox.get_legal_moves()
            return ToolExecution(
                result=result,
                meta={
                    "state_mutating": False,
                    "illegal_action": False,
                    "action_kind": "query",
                    "counts_as_move": False,
                },
            )

        return ToolExecution(
            result=self._error_result(f"unknown tool: {name}"),
            meta={
                "state_mutating": False,
                "illegal_action": False,
                "action_kind": "query",
                "counts_as_move": False,
            },
        )

    def get_state_snapshot(self) -> dict[str, Any]:
        return self.env.get_state().to_dict()

    def is_solved(self) -> bool:
        return self.env.is_solved()

    def default_instructions(self) -> str:
        if self._instructions is not None:
            return self._instructions
        return default_instructions(tool_prefix=self._tool_prefix)

    def format_state(self) -> str:
        return self.env.format_prompt_state(
            include_legal_moves=False,
            include_deadlock_status=False,
        )

    def episode_metrics(self) -> dict[str, Any]:
        state = self.env.get_state()
        return {
            "level_id": self.env.level.level_id,
            "n_boxes": self.env.n_boxes,
            "grid_size": {"width": state.width, "height": state.height},
            "move_count": self.env.move_count,
            "push_count": self.env.push_count,
            "boxes_on_goals": self.env.boxes_on_goals,
            "deadlocked": self.env.is_deadlocked(),
            "optimal_moves": self.env.level.optimal_moves,
            "optimal_pushes": self.env.level.optimal_pushes,
            "known_optimal": self.env.level.known_optimal,
            "history": list(self.env.history),
        }
