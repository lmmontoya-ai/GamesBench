from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

from games_bench.llm import (
    CLIProvider,
    CodexAppServerProvider,
    CodexCLIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
)


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def build_provider(
    args: argparse.Namespace,
    model: str | None,
    *,
    provider_retries: int | None = None,
    provider_backoff: float | None = None,
    stream_debug: bool | None = None,
    parallel_tool_calls: bool | None = None,
    max_tool_calls_per_turn: int | None = None,
) -> Any:
    retries = provider_retries
    if retries is None:
        retries = getattr(args, "provider_retries", 2)
    if retries is None:
        retries = 2

    backoff = provider_backoff
    if backoff is None:
        backoff = getattr(args, "provider_backoff", 1.0)
    if backoff is None:
        backoff = 1.0

    debug = stream_debug
    if debug is None:
        debug = getattr(args, "stream_debug", False)

    if args.provider == "openrouter":
        model = model or require_env("OPENROUTER_MODEL")
        return OpenRouterProvider(
            model=model,
            max_retries=int(retries),
            retry_backoff_s=float(backoff),
            stream_debug=bool(debug),
            timeout_s=int(getattr(args, "timeout_s", 300)),
            parallel_tool_calls=parallel_tool_calls,
        )
    if args.provider == "openai":
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        return OpenAIResponsesProvider(
            model=model,
            parallel_tool_calls=parallel_tool_calls,
            max_retries=int(retries),
            retry_backoff_s=float(backoff),
            timeout_s=int(getattr(args, "timeout_s", 300)),
        )
    if args.provider == "codex":
        return CodexAppServerProvider(
            model=model,
            codex_path=args.codex_path,
            app_args=getattr(args, "codex_app_args", []),
            timeout_s=int(getattr(args, "timeout_s", 300)),
            max_tool_calls_per_turn=int(max_tool_calls_per_turn or 1),
        )
    if args.provider == "codex-exec":
        return CodexCLIProvider(
            codex_path=args.codex_path,
            extra_args=getattr(args, "codex_args", []),
            timeout_s=int(getattr(args, "timeout_s", 300)),
        )
    if args.provider == "cli":
        if not args.cli_cmd:
            raise SystemExit("--cli-cmd is required for provider=cli")
        return CLIProvider(
            command=args.cli_cmd, use_stdin=not args.no_stdin, timeout_s=args.timeout_s
        )
    raise SystemExit(f"Unknown provider: {args.provider}")


def resolve_models(
    provider: str,
    config: dict[str, Any] | None,
    fallback: str | None,
) -> list[str]:
    if config and "models" in config:
        models = config["models"]
        if isinstance(models, list):
            return [str(m) for m in models]
        if isinstance(models, dict):
            if provider in models:
                provider_models = models[provider]
                if isinstance(provider_models, list):
                    return [str(m) for m in provider_models]
                return [str(provider_models)]
            if "default" in models:
                default_models = models["default"]
                if isinstance(default_models, list):
                    return [str(m) for m in default_models]
                return [str(default_models)]
    if provider in {"openrouter", "openai"}:
        return [fallback] if fallback else []
    if provider == "codex":
        return [fallback or "default"]
    return [fallback or "default"]


def parse_str_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            result.extend(parse_str_list(value))
            continue
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                result.append(chunk)
    return result


def parse_int_list(values: Iterable[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            result.append(int(chunk))
    return result


def resolve_out_dir_base(base: str | Path, game_name: str) -> Path:
    base_str = str(base)
    if "{game}" in base_str:
        base_str = base_str.replace("{game}", game_name)
    path = Path(base_str)
    if path.name == game_name:
        return path
    return path / game_name


def resolve_positive_int(
    arg_value: int | None,
    config: dict[str, Any],
    key: str,
    default: int,
) -> int:
    value = int(arg_value) if arg_value is not None else int(config.get(key, default))
    if value < 1:
        raise SystemExit(f"{key} must be >= 1, got {value}")
    return value


def resolve_optional_positive_int(
    arg_value: int | None,
    config: dict[str, Any],
    key: str,
) -> int | None:
    value = arg_value if arg_value is not None else config.get(key)
    if value is None:
        return None
    resolved = int(value)
    if resolved < 1:
        raise SystemExit(f"{key} must be >= 1, got {resolved}")
    return resolved


def resolve_max_tool_calls_per_turn(
    arg_value: int | None,
    config: dict[str, Any],
    *,
    default: int = 1,
) -> int:
    if arg_value is not None:
        value = int(arg_value)
    elif "max_actions_per_turn" in config:
        value = int(config["max_actions_per_turn"])
    else:
        value = int(config.get("max_tool_calls_per_turn", default))
    if value < 1:
        raise SystemExit(f"max_tool_calls_per_turn must be >= 1, got {value}")
    return value


def resolve_parallel_settings(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    provider_name: str,
) -> tuple[int, int]:
    parallelism = resolve_positive_int(
        getattr(args, "parallelism", None), config, "parallelism", 1
    )
    max_inflight_arg = getattr(args, "max_inflight_provider", None)
    max_inflight_cfg = config.get("max_inflight_provider")
    if max_inflight_arg is not None:
        max_inflight = int(max_inflight_arg)
    elif max_inflight_cfg is not None:
        max_inflight = int(max_inflight_cfg)
    elif provider_name == "openrouter":
        max_inflight = min(parallelism, 4)
    else:
        max_inflight = parallelism
    if max_inflight < 1:
        raise SystemExit(f"max_inflight_provider must be >= 1, got {max_inflight}")
    return parallelism, max_inflight


def resolve_checkpoint_interval(
    args: argparse.Namespace, config: dict[str, Any]
) -> int:
    raw = (
        getattr(args, "checkpoint_interval", None)
        if getattr(args, "checkpoint_interval", None) is not None
        else config.get("checkpoint_interval", 1)
    )
    value = int(raw)
    if value < 1:
        raise SystemExit(f"checkpoint_interval must be >= 1, got {value}")
    return value
