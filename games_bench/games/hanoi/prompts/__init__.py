from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


DEFAULT_TEMPLATE = _read("default.txt")
IMAGE_INSTRUCTIONS_SUFFIX = _read("image_suffix.txt")


def format_instructions(instructions: str, *, start_peg: int, goal_peg: int) -> str:
    if "{start_peg}" in instructions or "{goal_peg}" in instructions:
        return instructions.format(start_peg=start_peg, goal_peg=goal_peg)
    return instructions


def default_instructions(*, start_peg: int = 0, goal_peg: int = 2) -> str:
    return format_instructions(DEFAULT_TEMPLATE, start_peg=start_peg, goal_peg=goal_peg)


def with_image_instructions(instructions: str) -> str:
    if IMAGE_INSTRUCTIONS_SUFFIX in instructions:
        return instructions
    return f"{instructions}\n{IMAGE_INSTRUCTIONS_SUFFIX}"
