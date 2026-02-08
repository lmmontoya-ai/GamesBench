from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

from games_bench.bench.game_loader import build_env_and_adapter, parse_env_kwargs
from games_bench.llm import (
    CLIProvider,
    CodexCLIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
    run_tool_calling_episode,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _build_provider(args: argparse.Namespace) -> Any:
    if args.provider == "openrouter":
        model = args.model or _require_env("OPENROUTER_MODEL")
        return OpenRouterProvider(model=model, max_retries=2, retry_backoff_s=1.0)
    if args.provider == "openai":
        model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        return OpenAIResponsesProvider(model=model)
    if args.provider == "codex":
        return CodexCLIProvider(
            codex_path=args.codex_path,
            extra_args=args.codex_args,
            timeout_s=args.timeout_s,
        )
    if args.provider == "cli":
        if not args.cli_cmd:
            raise SystemExit("--cli-cmd is required for provider=cli")
        return CLIProvider(
            command=args.cli_cmd, use_stdin=not args.no_stdin, timeout_s=args.timeout_s
        )
    raise SystemExit(f"Unknown provider: {args.provider}")


def _build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    if args.game == "hanoi":
        env_kwargs.setdefault("record_history", True)
        env_kwargs.setdefault("illegal_action_behavior", "penalize")
        if args.n_pegs is not None:
            env_kwargs["n_pegs"] = args.n_pegs
        if args.n_disks is not None:
            env_kwargs["n_disks"] = args.n_disks
    return env_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-calling benchmark runner.")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "openai", "codex", "cli"],
        required=True,
        help="Which provider to use.",
    )
    parser.add_argument("--game", default="hanoi", help="Registered game name.")
    parser.add_argument(
        "--env-kwargs",
        default=None,
        help="JSON object of kwargs for the selected game's env factory.",
    )
    parser.add_argument("--model", help="Model name for OpenAI/OpenRouter.")
    parser.add_argument(
        "--n-disks",
        type=int,
        default=None,
        help="Hanoi convenience flag (overrides env_kwargs.n_disks).",
    )
    parser.add_argument(
        "--n-pegs",
        type=int,
        default=None,
        help="Hanoi convenience flag (overrides env_kwargs.n_pegs).",
    )
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--timeout-s", type=int, default=300)
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

    args = parser.parse_args()
    provider = _build_provider(args)
    _, adapter = build_env_and_adapter(
        args.game,
        env_kwargs=_build_env_kwargs(args),
    )
    result = run_tool_calling_episode(adapter, provider, max_turns=args.max_turns)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
