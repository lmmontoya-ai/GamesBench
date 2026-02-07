from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from games_bench.bench.common import add_common_batch_arguments
from games_bench.games.hanoi.adapter import HanoiGameAdapter
from games_bench.games.hanoi.env import TowerOfHanoiEnv, tool_schemas
from games_bench.games.hanoi.prompts import (
    default_instructions,
    format_instructions,
    with_image_instructions,
)
from games_bench.games.hanoi.vision import render_hanoi_env_image
from games_bench.llm import (
    CLIProvider,
    CodexCLIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
    build_recording,
    run_tool_calling_episode,
)
from games_bench.config import load_config


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


def default_hanoi_config() -> dict[str, Any]:
    return {
        "n_disks": [3],
        "runs_per_variant": 3,
        "max_turns": 200,
        "start_peg": 0,
        "goal_peg": 2,
        "prompt_variants": ["minimal"],
        "tool_variants": ["move_only"],
        "state_format": "text",
        "image_size": "640x360",
        "image_labels": True,
        "image_background": "white",
        "record": False,
        "record_raw": False,
        "record_provider_raw": False,
        "provider_retries": 2,
        "provider_backoff": 1.0,
    }


def build_hanoi_adapter(env: TowerOfHanoiEnv, **kwargs: Any) -> HanoiGameAdapter:
    return HanoiGameAdapter(env, **kwargs)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _build_provider(
    args: argparse.Namespace,
    model: str | None,
    *,
    provider_retries: int | None = None,
    provider_backoff: float | None = None,
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

    if args.provider == "openrouter":
        model = model or _require_env("OPENROUTER_MODEL")
        return OpenRouterProvider(
            model=model,
            max_retries=int(retries),
            retry_backoff_s=float(backoff),
        )
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


def _parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid image size: {value}")
    return int(parts[0]), int(parts[1])


def _image_payload(image) -> dict[str, Any]:
    return {
        "mime_type": image.mime_type,
        "data_base64": image.data_base64,
        "data_url": image.data_url,
        "width": image.width,
        "height": image.height,
    }


def _write_raw_generations(
    events: list[dict[str, Any]],
    *,
    out_file,
    episode_id: int,
    variant_id: str,
    instructions: str,
    tool_schemas: list[dict[str, Any]],
    state_format: str,
    image_config: dict[str, Any],
) -> None:
    last_snapshot: dict[str, Any] | None = None
    last_image_meta: dict[str, Any] | None = None
    current: dict[str, Any] | None = None
    turn_index = 0

    for event in events:
        event_type = event.get("type")
        if event_type == "state_snapshot":
            last_snapshot = event.get("state")
            continue
        if event_type == "state_image":
            last_image_meta = event.get("meta")
            continue
        if event_type == "state":
            state_text = event.get("state_text", event.get("state"))
            prompt_text = (
                f"{instructions}\n\nSTATE:\n{state_text}\n\nTOOLS:\n"
                f"{json.dumps(tool_schemas, indent=2)}"
            )
            current = {
                "episode_id": episode_id,
                "variant_id": variant_id,
                "turn_index": turn_index,
                "prompt": {
                    "instructions": instructions,
                    "state_text": state_text,
                    "tool_schemas": tool_schemas,
                    "state_format": state_format,
                    "prompt_text": prompt_text,
                    "image": (
                        {
                            "meta": last_image_meta,
                            "config": image_config,
                        }
                        if state_format in {"image", "both"}
                        else None
                    ),
                },
                "state_snapshot": last_snapshot,
            }
            continue
        if event_type == "provider_result":
            if current is not None:
                current["provider_result"] = {
                    "error": event.get("error"),
                    "usage": event.get("usage"),
                    "cost": event.get("cost"),
                    "raw": event.get("raw"),
                }
            continue
        if event_type == "tool_call":
            if current is not None:
                current["tool_call"] = {
                    "name": event.get("name"),
                    "arguments": event.get("arguments"),
                }
            continue
        if event_type == "tool_result":
            if current is not None:
                current["tool_result"] = event.get("result")
                if event.get("meta") is not None:
                    current["tool_result_meta"] = event.get("meta")
                out_file.write(json.dumps(current) + "\n")
                current = None
                turn_index += 1

    if current is not None:
        current["incomplete"] = True
        out_file.write(json.dumps(current) + "\n")


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

    def _as_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    solved = [e for e in episodes if e.get("solved")]
    move_ratios: list[float] = []
    for episode in solved:
        move_count = _as_float(episode.get("move_count"))
        optimal_steps = _as_float(episode.get("optimal_steps"))
        if move_count is not None and optimal_steps is not None and optimal_steps > 0:
            move_ratios.append(move_count / optimal_steps)

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
        "avg_moves": _mean(
            [
                value
                for value in (_as_float(e.get("move_count")) for e in episodes)
                if value is not None
            ]
        ),
        "avg_move_ratio": _mean(move_ratios),
        "avg_illegal_moves": _mean(
            [
                value
                for value in (_as_float(e.get("illegal_moves")) for e in episodes)
                if value is not None
            ]
        ),
        "avg_tool_calls": _mean(
            [
                value
                for value in (_as_float(e.get("tool_calls")) for e in episodes)
                if value is not None
            ]
        ),
        "token_totals": token_totals if token_count else None,
        "token_avgs": token_avgs,
        "cost_total": cost_total if cost_count else None,
        "cost_avg": (cost_total / cost_count) if cost_count else None,
    }


def add_hanoi_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-disks", action="append", default=None)
    parser.add_argument("--start-peg", type=int, default=None)
    parser.add_argument("--goal-peg", type=int, default=None)
    parser.add_argument("--runs-per-variant", type=int, default=None)
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
        "--state-format",
        choices=["text", "image", "both"],
        help="State input format for the model (default: text).",
    )
    parser.add_argument(
        "--image-size",
        help="Image size as WIDTHxHEIGHT (default: 640x360).",
    )
    parser.add_argument(
        "--image-background",
        default=None,
        help="Background color for rendered images (default: white).",
    )
    parser.add_argument(
        "--image-labels",
        action="store_true",
        help="Label pegs in rendered images.",
    )
    parser.add_argument(
        "--no-image-labels",
        action="store_true",
        help="Do not label pegs in rendered images.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch benchmark for tool-calling Hanoi."
    )
    add_common_batch_arguments(parser)
    add_hanoi_arguments(parser)
    return parser


def _resolve_out_dir_base(base: str | Path, game_name: str) -> Path:
    base_str = str(base)
    if "{game}" in base_str:
        base_str = base_str.replace("{game}", game_name)
    path = Path(base_str)
    if path.name == game_name:
        return path
    return path / game_name


def run_batch(
    args: argparse.Namespace,
    config: dict[str, Any] | None,
    *,
    game_name: str = "hanoi",
) -> list[Path]:
    config = config or {}
    provider_name = getattr(args, "provider", None)
    if not provider_name:
        raise SystemExit("Missing required argument: --provider")
    model_arg = getattr(args, "model", None)
    models = _resolve_models(provider_name, config, model_arg)
    if not models:
        raise SystemExit("No models provided. Use --model or config.json.")

    n_disks_arg = getattr(args, "n_disks", None)
    n_disks_list = (
        _parse_int_list(n_disks_arg)
        if n_disks_arg is not None
        else _parse_int_list([str(x) for x in config.get("n_disks", [3])])
    )
    runs_per_variant_arg = getattr(args, "runs_per_variant", None)
    runs_per_variant = (
        runs_per_variant_arg
        if runs_per_variant_arg is not None
        else int(config.get("runs_per_variant", 3))
    )
    max_turns_arg = getattr(args, "max_turns", None)
    max_turns = (
        max_turns_arg
        if max_turns_arg is not None
        else int(config.get("max_turns", 200))
    )
    out_dir_base = getattr(args, "out_dir", None) or config.get(
        "out_dir", "artifacts/runs"
    )
    out_dir_base = _resolve_out_dir_base(out_dir_base, game_name)
    start_peg_arg = getattr(args, "start_peg", None)
    start_peg = (
        start_peg_arg if start_peg_arg is not None else int(config.get("start_peg", 0))
    )
    goal_peg_arg = getattr(args, "goal_peg", None)
    goal_peg = (
        goal_peg_arg if goal_peg_arg is not None else int(config.get("goal_peg", 2))
    )
    provider_retries_arg = getattr(args, "provider_retries", None)
    provider_retries = (
        provider_retries_arg
        if provider_retries_arg is not None
        else int(config.get("provider_retries", 2))
    )
    provider_backoff_arg = getattr(args, "provider_backoff", None)
    provider_backoff = (
        provider_backoff_arg
        if provider_backoff_arg is not None
        else float(config.get("provider_backoff", 1.0))
    )
    prompt_file_arg = getattr(args, "prompt_file", None)
    prompt_variants = (
        _load_prompt_variants(prompt_file_arg)
        if prompt_file_arg
        else DEFAULT_PROMPT_VARIANTS
    )

    selected_prompt_names = getattr(args, "prompt_variants", None) or config.get(
        "prompt_variants", ["minimal"]
    )
    selected_prompt_variants = [prompt_variants[name] for name in selected_prompt_names]
    tool_variants = DEFAULT_TOOL_VARIANTS
    selected_tool_names = getattr(args, "tool_variants", None) or config.get(
        "tool_variants", ["move_only"]
    )
    selected_tool_variants = [tool_variants[name] for name in selected_tool_names]

    allowed_tools_override = getattr(args, "allowed_tools", None) or config.get(
        "allowed_tools"
    )
    if allowed_tools_override:
        selected_tool_variants = [
            ToolVariant(
                name="custom",
                allowed_tools=[
                    t.strip() for t in allowed_tools_override.split(",") if t.strip()
                ],
            )
        ]

    if getattr(args, "record_provider_raw", False):
        record_provider_raw = True
    elif getattr(args, "no_record_provider_raw", False):
        record_provider_raw = False
    else:
        record_provider_raw = bool(config.get("record_provider_raw", False))

    if getattr(args, "record_raw", False):
        record_raw = True
    elif getattr(args, "no_record_raw", False):
        record_raw = False
    else:
        record_raw = bool(config.get("record_raw", False))

    if record_raw:
        record_provider_raw = True

    if getattr(args, "record", False):
        record = True
    elif getattr(args, "no_record", False):
        record = False
    else:
        record = bool(config.get("record", False))

    state_format = getattr(args, "state_format", None) or config.get(
        "state_format", "text"
    )
    image_size = _parse_size(
        getattr(args, "image_size", None) or config.get("image_size", "640x360")
    )
    if getattr(args, "image_labels", False):
        image_labels = True
    elif getattr(args, "no_image_labels", False):
        image_labels = False
    else:
        image_labels = bool(config.get("image_labels", True))
    image_background = getattr(args, "image_background", None) or config.get(
        "image_background", "white"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dirs: list[Path] = []
    for model_name in models:
        provider = _build_provider(
            args,
            model_name,
            provider_retries=provider_retries,
            provider_backoff=provider_backoff,
        )
        if state_format in {"image", "both"} and not getattr(
            provider, "supports_images", False
        ):
            raise SystemExit(
                f"Provider '{provider_name}' does not support state_format='{state_format}'. "
                "Use --state-format text or a provider with image support."
            )
        model_slug = model_name.replace("/", "_").replace(":", "_")
        run_id = f"{timestamp}_{provider_name}_{model_slug}"

        out_dir = Path(out_dir_base) / provider_name / model_slug / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        full_tool_schemas = tool_schemas()
        tool_schemas_by_variant = {
            variant.name: (
                [t for t in full_tool_schemas if t["name"] in variant.allowed_tools]
                if variant.allowed_tools is not None
                else full_tool_schemas
            )
            for variant in selected_tool_variants
        }

        run_config = {
            "run_id": run_id,
            "timestamp": timestamp,
            "game": game_name,
            "provider": provider_name,
            "model": model_name,
            "n_disks": n_disks_list,
            "start_peg": start_peg,
            "goal_peg": goal_peg,
            "runs_per_variant": runs_per_variant,
            "max_turns": max_turns,
            "prompt_variants": [asdict(v) for v in selected_prompt_variants],
            "tool_variants": [asdict(v) for v in selected_tool_variants],
            "tool_schemas": full_tool_schemas,
            "tool_schemas_by_variant": tool_schemas_by_variant,
            "state_format": state_format,
            "image_size": f"{image_size[0]}x{image_size[1]}",
            "image_labels": image_labels,
            "image_background": image_background,
            "record_raw": record_raw,
            "record_provider_raw": record_provider_raw,
            "provider_retries": provider_retries,
            "provider_backoff": provider_backoff,
            "python": platform.python_version(),
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
        raw_path = out_dir / "raw_generations.jsonl"
        with (
            episodes_path.open("w") as ep_file,
            traces_path.open("w") as trace_file,
            raw_path.open("w") if record_raw else open(os.devnull, "w") as raw_file,
        ):
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
                                start_peg=start_peg,
                                goal_peg=goal_peg,
                                record_history=True,
                                illegal_action_behavior="penalize",
                            )
                            adapter = HanoiGameAdapter(env)
                            instructions = format_instructions(
                                prompt_variant.instructions,
                                start_peg=env.start_peg,
                                goal_peg=env.goal_peg,
                            )
                            if state_format in {"image", "both"}:
                                instructions = with_image_instructions(instructions)
                            state_formatter = (
                                lambda a, pv=prompt_variant: a.env.format_prompt_state(
                                    include_legal_moves=pv.include_legal_moves,
                                    include_action_space=pv.include_action_space,
                                )
                            )
                            if state_format == "image":
                                state_formatter = lambda _a: "State image attached."
                            state_image_renderer = None
                            if state_format in {"image", "both"}:

                                def state_image_renderer_payload(
                                    a,
                                    size=image_size,
                                    labels=image_labels,
                                    background=image_background,
                                ):
                                    image = render_hanoi_env_image(
                                        a.env,
                                        size=size,
                                        label_pegs=labels,
                                        background=background,
                                    )
                                    return _image_payload(image)

                                state_image_renderer = state_image_renderer_payload
                            result = run_tool_calling_episode(
                                adapter,
                                provider,
                                max_turns=max_turns,
                                instructions=instructions,
                                state_formatter=state_formatter,
                                state_image_renderer=state_image_renderer,
                                allowed_tools=tool_variant.allowed_tools,
                                record_provider_raw=record_provider_raw,
                            )
                            if record_raw:
                                tools = (
                                    [
                                        t
                                        for t in tool_schemas()
                                        if t["name"] in tool_variant.allowed_tools
                                    ]
                                    if tool_variant.allowed_tools is not None
                                    else tool_schemas()
                                )
                                image_config = {
                                    "size": image_size,
                                    "labels": image_labels,
                                    "background": image_background,
                                }
                                _write_raw_generations(
                                    result.events,
                                    out_file=raw_file,
                                    episode_id=episode_id,
                                    variant_id=variant_id,
                                    instructions=instructions,
                                    tool_schemas=tools,
                                    state_format=state_format,
                                    image_config=image_config,
                                )
                            episode = {
                                "episode_id": episode_id,
                                "variant_id": variant_id,
                                "run_idx": run_idx,
                                "provider": provider_name,
                                "model": model_name,
                                "n_disks": result.game_metrics.get("n_disks", n_disks),
                                "prompt_variant": prompt_variant.name,
                                "tools_variant": tool_variant.name,
                                "solved": result.solved,
                                "move_count": result.game_metrics.get(
                                    "move_count", result.move_count
                                ),
                                "optimal_steps": result.game_metrics.get(
                                    "optimal_steps", result.optimal_steps
                                ),
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
        run_dirs.append(out_dir)
    return run_dirs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}
    run_dirs = run_batch(args, config, game_name="hanoi")
    print(json.dumps({"run_dirs": [str(p) for p in run_dirs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
