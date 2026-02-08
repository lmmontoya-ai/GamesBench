"""Backward-compatible re-export for Tower of Hanoi."""

from __future__ import annotations

from games_bench.games.hanoi.env import (
    ACTION_SPACE,
    Action,
    Disk,
    HanoiError,
    HanoiState,
    HanoiToolbox,
    IllegalMoveError,
    InvalidActionError,
    InvalidPegError,
    MIN_PEGS,
    PegIndex,
    TowerOfHanoiEnv,
    action_space_for_n_pegs,
    format_state_for_prompt,
    state_schema,
    tool_schemas,
)

__all__ = [
    "ACTION_SPACE",
    "Action",
    "Disk",
    "HanoiError",
    "HanoiState",
    "HanoiToolbox",
    "IllegalMoveError",
    "InvalidActionError",
    "InvalidPegError",
    "MIN_PEGS",
    "PegIndex",
    "TowerOfHanoiEnv",
    "action_space_for_n_pegs",
    "format_state_for_prompt",
    "state_schema",
    "tool_schemas",
]
