from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

from games_bench.bench.hanoi_adapter import HanoiGameAdapter
from games_bench.games.hanoi.env import TowerOfHanoiEnv
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tool-calling benchmark runner (Tower of Hanoi)."
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "openai", "codex", "cli"],
        required=True,
        help="Which provider to use.",
    )
    parser.add_argument("--model", help="Model name for OpenAI/OpenRouter.")
    parser.add_argument("--n-disks", type=int, default=3)
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

    env = TowerOfHanoiEnv(
        n_disks=args.n_disks, record_history=True, illegal_action_behavior="penalize"
    )
    adapter = HanoiGameAdapter(env)
    result = run_tool_calling_episode(adapter, provider, max_turns=args.max_turns)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
