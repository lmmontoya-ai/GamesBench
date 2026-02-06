from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolExecution:
    result: dict[str, Any]
    meta: dict[str, Any]


class GameAdapter(Protocol):
    def tool_schemas(self) -> list[dict[str, Any]]: ...

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecution: ...

    def get_state_snapshot(self) -> dict[str, Any]: ...

    def is_solved(self) -> bool: ...

    def default_instructions(self) -> str: ...

    def format_state(self) -> str: ...

    def episode_metrics(self) -> dict[str, Any]: ...
