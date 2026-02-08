from __future__ import annotations

from typing import Any

from .env import SokobanEnv
from .level_loader import load_level_by_id
from .procgen import generate_procedural_level


def make_sokoban_env(
    *,
    level_id: str | None = None,
    level_set: str = "starter-authored-v1",
    level_index: int = 1,
    procedural: bool = False,
    width: int | None = None,
    height: int | None = None,
    n_boxes: int | None = None,
    procgen_seed: int | None = None,
    procgen_wall_density: float = 0.08,
    procgen_scramble_steps: int | None = None,
    **env_kwargs: Any,
) -> SokobanEnv:
    """Create a Sokoban env from registry-friendly kwargs.

    The registry contract expects an env factory with optional kwargs. Sokoban
    requires a level, so we derive one from either `level_id` or
    (`level_set`, `level_index`).
    """

    use_procedural = procedural or (
        width is not None and height is not None and n_boxes is not None
    )
    if use_procedural:
        if width is None or height is None or n_boxes is None:
            raise ValueError(
                "procedural Sokoban env requires width, height, and n_boxes"
            )
        level = generate_procedural_level(
            width=int(width),
            height=int(height),
            n_boxes=int(n_boxes),
            seed=procgen_seed,
            wall_density=float(procgen_wall_density),
            scramble_steps=procgen_scramble_steps,
        )
    else:
        resolved_level_id = level_id or f"{level_set}:{level_index}"
        level = load_level_by_id(resolved_level_id)
    return SokobanEnv(level, **env_kwargs)
