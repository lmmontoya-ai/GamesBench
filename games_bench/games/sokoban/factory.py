from __future__ import annotations

from typing import Any

from .env import SokobanEnv
from .level_loader import load_level_by_id


def make_sokoban_env(
    *,
    level_id: str | None = None,
    level_set: str = "starter-authored-v1",
    level_index: int = 1,
    **env_kwargs: Any,
) -> SokobanEnv:
    """Create a Sokoban env from registry-friendly kwargs.

    The registry contract expects an env factory with optional kwargs. Sokoban
    requires a level, so we derive one from either `level_id` or
    (`level_set`, `level_index`).
    """

    resolved_level_id = level_id or f"{level_set}:{level_index}"
    level = load_level_by_id(resolved_level_id)
    return SokobanEnv(level, **env_kwargs)
