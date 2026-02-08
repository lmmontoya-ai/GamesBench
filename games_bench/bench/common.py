from __future__ import annotations

import argparse


PROVIDER_CHOICES = ["openrouter", "openai", "codex", "cli"]


def add_common_batch_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_game_filter: bool = False,
) -> None:
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        help="Which provider to use.",
    )
    parser.add_argument("--model", help="Model name for OpenAI/OpenRouter.")
    parser.add_argument(
        "--config", help="Path to JSON config (models + optional defaults)."
    )
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument(
        "--provider-retries",
        type=int,
        default=None,
        help="Retry count for provider 5xx errors (default from config or 2).",
    )
    parser.add_argument(
        "--provider-backoff",
        type=float,
        default=None,
        help="Base backoff (seconds) for provider retries (default from config or 1.0).",
    )
    parser.add_argument(
        "--stream-debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable provider streaming debug logs (OpenRouter only). "
            "Use --no-stream-debug to force disable."
        ),
    )
    parser.add_argument("--cli-cmd", help="Command to run for provider=cli.")
    parser.add_argument(
        "--no-stdin",
        action="store_true",
        help="Do not pass prompt via stdin for provider=cli.",
    )
    parser.add_argument("--codex-path", default="codex")
    parser.add_argument(
        "--codex-arg",
        action="append",
        dest="codex_args",
        default=[],
        help="Extra args to pass to codex exec (repeatable).",
    )
    parser.add_argument(
        "--record-provider-raw",
        action="store_true",
        help="Include raw provider responses in traces.",
    )
    parser.add_argument(
        "--no-record-provider-raw",
        action="store_true",
        help="Disable raw provider responses in traces.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write per-episode recordings (states/actions) into run directory.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable recordings even if config enables them.",
    )
    parser.add_argument(
        "--record-raw",
        action="store_true",
        help="Write raw generations (prompt + model output + tool result) to JSONL.",
    )
    parser.add_argument(
        "--no-record-raw",
        action="store_true",
        help="Disable raw generations logging even if config enables it.",
    )
    if include_game_filter:
        parser.add_argument(
            "--game",
            action="append",
            dest="games",
            default=None,
            help="Game(s) to run (default: all in config).",
        )
