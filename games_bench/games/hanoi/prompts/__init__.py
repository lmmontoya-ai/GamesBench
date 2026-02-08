from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


DEFAULT_TEMPLATE = _read("default.txt")
IMAGE_INSTRUCTIONS_SUFFIX = _read("image_suffix.txt")


def format_instructions(
    instructions: str,
    *,
    n_pegs: int = 3,
    start_peg: int = 0,
    goal_peg: int | None = None,
) -> str:
    if n_pegs < 3:
        raise ValueError(f"n_pegs must be >= 3, got {n_pegs}")
    resolved_goal_peg = n_pegs - 1 if goal_peg is None else goal_peg
    if (
        "{start_peg}" in instructions
        or "{goal_peg}" in instructions
        or "{n_pegs}" in instructions
        or "{max_peg_index}" in instructions
    ):
        return instructions.format(
            n_pegs=n_pegs,
            max_peg_index=n_pegs - 1,
            start_peg=start_peg,
            goal_peg=resolved_goal_peg,
        )
    return instructions


def default_instructions(
    *, n_pegs: int = 3, start_peg: int = 0, goal_peg: int | None = None
) -> str:
    return format_instructions(
        DEFAULT_TEMPLATE,
        n_pegs=n_pegs,
        start_peg=start_peg,
        goal_peg=goal_peg,
    )


def with_image_instructions(instructions: str) -> str:
    if IMAGE_INSTRUCTIONS_SUFFIX in instructions:
        return instructions
    return f"{instructions}\n{IMAGE_INSTRUCTIONS_SUFFIX}"
