from __future__ import annotations

import argparse
import sys
from typing import Any


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
    parser.add_argument(
        "--stateless",
        action="store_true",
        help=(
            "Disable conversation history across turns. "
            "By default runs are stateful and include prior turn context."
        ),
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help=(
            "Skip summary scoring during generation. "
            "You can score later with `games-bench score --run-dir ...`."
        ),
    )
    parser.add_argument(
        "--score-version",
        default=None,
        help="Score version label written into summary.json (default: score-v1).",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=None,
        help="Number of episode workers to run concurrently (default from config or 1).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Stable run identifier to reuse output directory naming. "
            "Required when using --resume."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from existing artifacts.",
    )
    parser.add_argument(
        "--strict-resume",
        action="store_true",
        help="Fail resume if checkpoint/job plan metadata do not match exactly.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help=(
            "Persist execution checkpoint every N committed episodes "
            "(default from config or 1)."
        ),
    )
    parser.add_argument(
        "--max-inflight-provider",
        type=int,
        default=None,
        help=(
            "Maximum concurrent provider calls across workers. "
            "Defaults to 4 for OpenRouter, otherwise parallelism."
        ),
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=None,
        help=(
            "Early-stop an episode after N consecutive turns with unchanged state. "
            "Disabled when unset."
        ),
    )
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
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Show benchmark episode progress with ETA and per-episode stats. "
            "Defaults to enabled on TTY stderr."
        ),
    )
    parser.add_argument(
        "--progress-refresh-s",
        type=float,
        default=None,
        help=(
            "Minimum seconds between progress refreshes "
            "(default from config or 0.25)."
        ),
    )
    parser.add_argument(
        "--suite",
        default=None,
        help=(
            "Named benchmark suite (for example: standard-v1). "
            "Suite config is merged before --config overrides."
        ),
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available benchmark suites and exit.",
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


def resolve_interaction_mode(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[bool, str]:
    stateless = bool(getattr(args, "stateless", False)) or bool(
        config.get("stateless", False)
    )
    interaction_mode = "stateless" if stateless else "stateful"
    return stateless, interaction_mode


def resolve_spec_name(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    interaction_mode: str,
) -> tuple[str, str]:
    raw_base = (
        getattr(args, "suite", None)
        or config.get("spec")
        or config.get("suite")
        or "custom"
    )
    spec_base = str(raw_base).strip() or "custom"
    for suffix in ("-stateful", "-stateless"):
        if spec_base.endswith(suffix):
            trimmed = spec_base[: -len(suffix)].strip("-_ ")
            spec_base = trimmed or "custom"
            break
    return spec_base, f"{spec_base}-{interaction_mode}"


def resolve_progress_settings(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[bool, float, bool]:
    progress_arg = getattr(args, "progress", None)
    progress_refresh_arg = getattr(args, "progress_refresh_s", None)
    has_config_progress = "progress" in config
    has_config_refresh = "progress_refresh_s" in config

    if progress_arg is not None:
        enabled = bool(progress_arg)
        explicit_request = True
    elif has_config_progress:
        enabled = bool(config.get("progress"))
        explicit_request = True
    else:
        enabled = bool(sys.stderr.isatty())
        explicit_request = False

    refresh_value = (
        progress_refresh_arg
        if progress_refresh_arg is not None
        else config.get("progress_refresh_s", 0.25)
    )
    if refresh_value is None and has_config_refresh:
        refresh_value = 0.25

    try:
        refresh_s = float(refresh_value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("progress_refresh_s must be a positive number.") from exc
    if refresh_s <= 0:
        raise SystemExit("progress_refresh_s must be > 0.")
    return enabled, refresh_s, explicit_request


def resolve_scoring_settings(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[bool, str]:
    scoring_enabled = not bool(getattr(args, "no_score", False))
    if scoring_enabled:
        scoring_enabled = bool(config.get("score", True))

    raw_version = getattr(args, "score_version", None) or config.get(
        "score_version", "score-v1"
    )
    score_version = str(raw_version).strip() if raw_version is not None else "score-v1"
    if not score_version:
        raise SystemExit("score_version must not be empty.")
    return scoring_enabled, score_version
