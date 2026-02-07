from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


DEFAULT_TEMPLATE = _read("default.txt")
IMAGE_INSTRUCTIONS_SUFFIX = _read("image_suffix.txt")

_DEADLOCK_WARNINGS_SUFFIX = """
Deadlock policy:
- A deadlocked state may terminate the episode.
- Treat corner and wall traps as high risk unless they are goals.
""".strip()

_PROMPT_VARIANTS = (
    "minimal",
    "with_legal_moves",
    "with_deadlock_warnings",
    "full",
)


def _legal_moves_suffix(tool_prefix: str) -> str:
    return (
        "Planning aid:\n"
        f"- Use `{tool_prefix}_get_legal_moves` when selecting between alternatives.\n"
        "- Avoid actions that are legal but likely to create deadlocks."
    )


def default_instructions(*, tool_prefix: str = "sokoban") -> str:
    return instructions_for_variant("minimal", tool_prefix=tool_prefix)


def instructions_for_variant(
    name: str = "minimal", *, tool_prefix: str = "sokoban"
) -> str:
    if name not in _PROMPT_VARIANTS:
        available = ", ".join(sorted(_PROMPT_VARIANTS))
        raise ValueError(
            f"unknown Sokoban prompt variant: {name!r} (available: {available})"
        )
    legal_moves_suffix = _legal_moves_suffix(tool_prefix)
    if name == "minimal":
        suffix = ""
    elif name == "with_legal_moves":
        suffix = legal_moves_suffix
    elif name == "with_deadlock_warnings":
        suffix = _DEADLOCK_WARNINGS_SUFFIX
    else:
        suffix = f"{legal_moves_suffix}\n\n{_DEADLOCK_WARNINGS_SUFFIX}"

    if not suffix:
        return DEFAULT_TEMPLATE
    return f"{DEFAULT_TEMPLATE}\n\n{suffix}"


def with_image_instructions(instructions: str) -> str:
    if IMAGE_INSTRUCTIONS_SUFFIX in instructions:
        return instructions
    return f"{instructions}\n{IMAGE_INSTRUCTIONS_SUFFIX}"
