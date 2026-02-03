from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from games_bench.hanoi import TowerOfHanoiEnv
from games_bench.llm import (
    CLIProvider,
    CodexCLIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
    build_recording,
    default_instructions,
    run_tool_calling_episode,
)


@dataclass(frozen=True, slots=True)
class PromptVariant:
    name: str
    instructions: str
    include_legal_moves: bool
    include_action_space: bool


@dataclass(frozen=True, slots=True)
class ToolVariant:
    name: str
    allowed_tools: list[str] | None


DEFAULT_PROMPT_VARIANTS = {
    "minimal": PromptVariant(
        name="minimal",
        instructions=default_instructions(),
        include_legal_moves=False,
        include_action_space=False,
    ),
    "legal_moves": PromptVariant(
        name="legal_moves",
        instructions=default_instructions(),
        include_legal_moves=True,
        include_action_space=False,
    ),
    "action_space": PromptVariant(
        name="action_space",
        instructions=default_instructions(),
        include_legal_moves=False,
        include_action_space=True,
    ),
    "full": PromptVariant(
        name="full",
        instructions=default_instructions(),
        include_legal_moves=True,
        include_action_space=True,
    ),
}

DEFAULT_TOOL_VARIANTS = {
    "move_only": ToolVariant(name="move_only", allowed_tools=["hanoi_move"]),
    "move_and_state": ToolVariant(
        name="move_and_state",
        allowed_tools=[
            "hanoi_move",
            "hanoi_get_state",
            "hanoi_get_legal_moves",
            "hanoi_is_solved",
        ],
    ),
    "all_tools": ToolVariant(name="all_tools", allowed_tools=None),
}


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _build_provider(args: argparse.Namespace, model: str | None) -> Any:
    if args.provider == "openrouter":
        model = model or _require_env("OPENROUTER_MODEL")
        return OpenRouterProvider(model=model)
    if args.provider == "openai":
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
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


def _parse_int_list(values: Iterable[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            result.append(int(chunk))
    return result


def _load_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _resolve_models(
    provider: str, config: dict[str, Any] | None, fallback: str | None
) -> list[str]:
    if config and "models" in config:
        models = config["models"]
        if isinstance(models, list):
            return [str(m) for m in models]
        if isinstance(models, dict):
            if provider in models:
                return [str(m) for m in models[provider]]
            if "default" in models:
                return [str(m) for m in models["default"]]
    if provider in {"openrouter", "openai"}:
        return [fallback] if fallback else []
    return [fallback or "default"]


def _load_prompt_variants(path: str) -> dict[str, PromptVariant]:
    data = json.loads(Path(path).read_text())
    variants: dict[str, PromptVariant] = {}
    for item in data:
        name = item["name"]
        variants[name] = PromptVariant(
            name=name,
            instructions=item.get("instructions", default_instructions()),
            include_legal_moves=bool(item.get("include_legal_moves", False)),
            include_action_space=bool(item.get("include_action_space", False)),
        )
    return variants


def _compute_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "episodes": 0,
            "solved": 0,
            "solve_rate": 0.0,
            "avg_moves": None,
            "avg_move_ratio": None,
            "avg_illegal_moves": None,
            "avg_tool_calls": None,
            "token_totals": None,
            "token_avgs": None,
            "cost_total": None,
            "cost_avg": None,
        }

    solved = [e for e in episodes if e["solved"]]
    move_ratios = [
        e["move_count"] / e["optimal_steps"] for e in solved if e["optimal_steps"] > 0
    ]

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    token_totals = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    token_count = 0
    cost_total = 0.0
    cost_count = 0

    for ep in episodes:
        usage = ep.get("usage") or {}
        if usage:
            token_count += 1
            token_totals["prompt_tokens"] += float(usage.get("prompt_tokens", 0.0))
            token_totals["completion_tokens"] += float(
                usage.get("completion_tokens", 0.0)
            )
            if "total_tokens" in usage:
                token_totals["total_tokens"] += float(usage.get("total_tokens", 0.0))
            else:
                token_totals["total_tokens"] += float(
                    usage.get("prompt_tokens", 0.0)
                ) + float(usage.get("completion_tokens", 0.0))

        if ep.get("cost") is not None:
            cost_count += 1
            cost_total += float(ep["cost"])

    token_avgs = None
    if token_count:
        token_avgs = {
            "prompt_tokens": token_totals["prompt_tokens"] / token_count,
            "completion_tokens": token_totals["completion_tokens"] / token_count,
            "total_tokens": token_totals["total_tokens"] / token_count,
        }

    return {
        "episodes": len(episodes),
        "solved": len(solved),
        "solve_rate": len(solved) / len(episodes),
        "avg_moves": _mean([e["move_count"] for e in episodes]),
        "avg_move_ratio": _mean(move_ratios),
        "avg_illegal_moves": _mean([e["illegal_moves"] for e in episodes]),
        "avg_tool_calls": _mean([e["tool_calls"] for e in episodes]),
        "token_totals": token_totals if token_count else None,
        "token_avgs": token_avgs,
        "cost_total": cost_total if cost_count else None,
        "cost_avg": (cost_total / cost_count) if cost_count else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch benchmark for tool-calling Hanoi."
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "openai", "codex", "cli"],
        required=True,
        help="Which provider to use.",
    )
    parser.add_argument("--model", help="Model name for OpenAI/OpenRouter.")
    parser.add_argument(
        "--config", help="Path to JSON config (models + optional defaults)."
    )
    parser.add_argument("--n-disks", action="append", default=None)
    parser.add_argument("--runs-per-variant", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
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
    parser.add_argument(
        "--prompt-variant",
        action="append",
        dest="prompt_variants",
        default=None,
        help="Prompt variants to run (default: minimal).",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to JSON list of prompt variants (overrides built-ins).",
    )
    parser.add_argument(
        "--tools-variant",
        action="append",
        dest="tool_variants",
        default=None,
        help="Tool variants to run (default: move_only).",
    )
    parser.add_argument(
        "--allowed-tools",
        help="Comma-separated list of tool names (override tools variant).",
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

    args = parser.parse_args()

    config = _load_config(args.config) if args.config else None
    models = _resolve_models(args.provider, config, args.model)
    if not models:
        raise SystemExit("No models provided. Use --model or config.json.")

    n_disks_list = (
        _parse_int_list(args.n_disks)
        if args.n_disks is not None
        else _parse_int_list([str(x) for x in (config or {}).get("n_disks", [3])])
    )
    runs_per_variant = (
        args.runs_per_variant
        if args.runs_per_variant is not None
        else int((config or {}).get("runs_per_variant", 3))
    )
    max_turns = (
        args.max_turns
        if args.max_turns is not None
        else int((config or {}).get("max_turns", 200))
    )
    out_dir_base = args.out_dir or (config or {}).get("out_dir", "runs/hanoi")
    prompt_variants = (
        _load_prompt_variants(args.prompt_file)
        if args.prompt_file
        else DEFAULT_PROMPT_VARIANTS
    )

    selected_prompt_names = args.prompt_variants or (config or {}).get(
        "prompt_variants", ["minimal"]
    )
    selected_prompt_variants = [prompt_variants[name] for name in selected_prompt_names]
    tool_variants = DEFAULT_TOOL_VARIANTS
    selected_tool_names = args.tool_variants or (config or {}).get(
        "tool_variants", ["move_only"]
    )
    selected_tool_variants = [tool_variants[name] for name in selected_tool_names]

    allowed_tools_override = args.allowed_tools or (config or {}).get("allowed_tools")
    if allowed_tools_override:
        selected_tool_variants = [
            ToolVariant(
                name="custom",
                allowed_tools=[
                    t.strip() for t in allowed_tools_override.split(",") if t.strip()
                ],
            )
        ]

    if args.record_provider_raw:
        record_provider_raw = True
    elif args.no_record_provider_raw:
        record_provider_raw = False
    else:
        record_provider_raw = bool((config or {}).get("record_provider_raw", False))

    if args.record:
        record = True
    elif args.no_record:
        record = False
    else:
        record = bool((config or {}).get("record", False))

    provider_name = args.provider
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in models:
        provider = _build_provider(args, model_name)
        model_slug = model_name.replace("/", "_").replace(":", "_")
        run_id = f"{timestamp}_{provider_name}_{model_slug}"

        out_dir = Path(out_dir_base) / provider_name / model_slug / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        run_config = {
            "run_id": run_id,
            "timestamp": timestamp,
            "provider": provider_name,
            "model": model_name,
            "n_disks": n_disks_list,
            "runs_per_variant": runs_per_variant,
            "max_turns": max_turns,
            "prompt_variants": [asdict(v) for v in selected_prompt_variants],
            "tool_variants": [asdict(v) for v in selected_tool_variants],
            "python": sys.version,
            "platform": platform.platform(),
        }
        (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

        episodes_path = out_dir / "episodes.jsonl"
        traces_path = out_dir / "traces.jsonl"
        recordings_dir = out_dir / "recordings"
        if record:
            recordings_dir.mkdir(parents=True, exist_ok=True)

        episodes: list[dict[str, Any]] = []
        episode_id = 0
        with episodes_path.open("w") as ep_file, traces_path.open("w") as trace_file:
            for n_disks in n_disks_list:
                for prompt_variant in selected_prompt_variants:
                    for tool_variant in selected_tool_variants:
                        variant_id = (
                            f"n{n_disks}__prompt={prompt_variant.name}"
                            f"__tools={tool_variant.name}"
                        )
                        for run_idx in range(runs_per_variant):
                            env = TowerOfHanoiEnv(
                                n_disks=n_disks,
                                record_history=True,
                                illegal_action_behavior="penalize",
                            )
                            state_formatter = (
                                lambda e, pv=prompt_variant: e.format_prompt_state(
                                    include_legal_moves=pv.include_legal_moves,
                                    include_action_space=pv.include_action_space,
                                )
                            )
                            result = run_tool_calling_episode(
                                env,
                                provider,
                                max_turns=max_turns,
                                instructions=prompt_variant.instructions,
                                state_formatter=state_formatter,
                                allowed_tools=tool_variant.allowed_tools,
                                record_provider_raw=record_provider_raw,
                            )
                            episode = {
                                "episode_id": episode_id,
                                "variant_id": variant_id,
                                "run_idx": run_idx,
                                "provider": provider_name,
                                "model": model_name,
                                "n_disks": n_disks,
                                "prompt_variant": prompt_variant.name,
                                "tools_variant": tool_variant.name,
                                "solved": result.solved,
                                "move_count": result.move_count,
                                "optimal_steps": result.optimal_steps,
                                "illegal_moves": result.illegal_moves,
                                "tool_calls": result.tool_calls,
                                "usage": result.usage,
                                "cost": result.cost,
                            }
                            if record:
                                recording = build_recording(
                                    events=result.events,
                                    metadata={
                                        "episode_id": episode_id,
                                        "variant_id": variant_id,
                                        "run_idx": run_idx,
                                        "provider": provider_name,
                                        "model": model_name,
                                        "n_disks": n_disks,
                                        "prompt_variant": prompt_variant.name,
                                        "tools_variant": tool_variant.name,
                                        "solved": result.solved,
                                    },
                                )
                                recording_path = (
                                    recordings_dir / f"episode_{episode_id:04d}.json"
                                )
                                recording_path.write_text(
                                    json.dumps(recording, indent=2)
                                )
                                episode["recording_file"] = str(recording_path)
                            episodes.append(episode)
                            ep_file.write(json.dumps(episode) + "\n")
                            trace_file.write(
                                json.dumps(
                                    {
                                        "episode_id": episode_id,
                                        "variant_id": variant_id,
                                        "events": result.events,
                                    }
                                )
                                + "\n"
                            )
                            episode_id += 1

        summary = {"overall": _compute_metrics(episodes), "variants": {}}
        for episode in episodes:
            variant_id = episode["variant_id"]
            summary["variants"].setdefault(variant_id, [])
            summary["variants"][variant_id].append(episode)
        summary["variants"] = {
            variant_id: _compute_metrics(items)
            for variant_id, items in summary["variants"].items()
        }

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"run_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
