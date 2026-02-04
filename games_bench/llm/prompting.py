"""Backward-compatible prompt helpers for Hanoi."""

from __future__ import annotations

from games_bench.games.hanoi.prompts import (  # re-export
    IMAGE_INSTRUCTIONS_SUFFIX,
    default_instructions,
    format_instructions,
    with_image_instructions,
)

__all__ = [
    "IMAGE_INSTRUCTIONS_SUFFIX",
    "default_instructions",
    "format_instructions",
    "with_image_instructions",
]
