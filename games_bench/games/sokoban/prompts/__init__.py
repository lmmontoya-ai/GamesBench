from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


DEFAULT_TEMPLATE = _read("default.txt")
IMAGE_INSTRUCTIONS_SUFFIX = _read("image_suffix.txt")

_LEGAL_MOVES_SUFFIX = """
Planning aid:
- Use `sokoban_get_legal_moves` when selecting between alternatives.
- Avoid actions that are legal but likely to create deadlocks.
""".strip()

_DEADLOCK_WARNINGS_SUFFIX = """
Deadlock policy:
- A deadlocked state may terminate the episode.
- Treat corner and wall traps as high risk unless they are goals.
""".strip()

_PROMPT_VARIANT_SUFFIXES = {
    "minimal": "",
    "with_legal_moves": _LEGAL_MOVES_SUFFIX,
    "with_deadlock_warnings": _DEADLOCK_WARNINGS_SUFFIX,
    "full": f"{_LEGAL_MOVES_SUFFIX}\n\n{_DEADLOCK_WARNINGS_SUFFIX}",
}


def default_instructions() -> str:
    return DEFAULT_TEMPLATE


def instructions_for_variant(name: str = "minimal") -> str:
    suffix = _PROMPT_VARIANT_SUFFIXES.get(name)
    if suffix is None:
        available = ", ".join(sorted(_PROMPT_VARIANT_SUFFIXES))
        raise ValueError(
            f"unknown Sokoban prompt variant: {name!r} (available: {available})"
        )
    if not suffix:
        return DEFAULT_TEMPLATE
    return f"{DEFAULT_TEMPLATE}\n\n{suffix}"


def with_image_instructions(instructions: str) -> str:
    if IMAGE_INSTRUCTIONS_SUFFIX in instructions:
        return instructions
    return f"{instructions}\n{IMAGE_INSTRUCTIONS_SUFFIX}"
