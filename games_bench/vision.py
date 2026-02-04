"""Backward-compatible re-export for Hanoi vision helpers."""

from __future__ import annotations

from games_bench.games.hanoi.vision import (
    StateImage,
    render_hanoi_env_image,
    render_hanoi_image,
    render_hanoi_state_image,
)

__all__ = [
    "StateImage",
    "render_hanoi_env_image",
    "render_hanoi_image",
    "render_hanoi_state_image",
]
