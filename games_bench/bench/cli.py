from __future__ import annotations

import sys
from typing import Callable

from games_bench.bench import (
    batch,
    manual_tool_loop,
    openai_tool_calling,
    provider,
    render,
    review,
    rl,
    tool_calling,
)


COMMANDS: dict[str, tuple[str, Callable[[], int]]] = {
    "run": ("Batch benchmark", batch.main),
    "provider": ("Single provider episode", provider.main),
    "render": ("Render recordings (html/video)", render.main),
    "review": ("Review run (prompt + images)", review.main),
    "rl": ("RL demo", rl.main),
    "tool-calling": ("Tool-calling demo", tool_calling.main),
    "openai-tool-calling": ("OpenAI tool-calling demo", openai_tool_calling.main),
    "manual-tool-loop": ("Manual tool loop demo", manual_tool_loop.main),
}


def _print_help() -> None:
    print("games-bench <command> [args]\n")
    print("Commands:")
    for name, (desc, _) in COMMANDS.items():
        print(f"  {name:20s} {desc}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command = args.pop(0)
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n")
        _print_help()
        return 2

    _, handler = COMMANDS[command]
    sys.argv = [f"games-bench {command}"] + args
    return handler()


if __name__ == "__main__":
    raise SystemExit(main())
