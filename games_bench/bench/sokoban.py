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
from games_bench.config import load_config
from games_bench.games.sokoban.adapter import SokobanGameAdapter
from games_bench.games.sokoban.env import SokobanEnv, SokobanLevel, tool_schemas
from games_bench.games.sokoban.level_loader import (
    list_bundled_level_sets,
    load_bundled_level_set,
    load_level_by_id,
)
from games_bench.games.sokoban.prompts import (
    instructions_for_variant,
    with_image_instructions,
)
from games_bench.games.sokoban.vision import render_sokoban_env_image
from games_bench.llm import (
    CLIProvider,
    CodexCLIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
    build_recording,
    run_tool_calling_episode,
)


@dataclass(frozen=True, slots=True)
class PromptVariant:
    name: str
    instructions: str
    include_legal_moves: bool
    include_deadlock_status: bool


@dataclass(frozen=True, slots=True)
class ToolVariant:
    name: str
    allowed_tools: list[str] | None
    terminal_on_deadlock_override: bool | None = None


DEFAULT_PROMPT_VARIANTS = {
    "minimal": PromptVariant(
        name="minimal",
        instructions=instructions_for_variant("minimal"),
        include_legal_moves=False,
        include_deadlock_status=False,
    ),
    "with_legal_moves": PromptVariant(
        name="with_legal_moves",
        instructions=instructions_for_variant("with_legal_moves"),
        include_legal_moves=True,
        include_deadlock_status=False,
    ),
    "with_deadlock_warnings": PromptVariant(
        name="with_deadlock_warnings",
        instructions=instructions_for_variant("with_deadlock_warnings"),
        include_legal_moves=False,
        include_deadlock_status=True,
    ),
    "full": PromptVariant(
        name="full",
        instructions=instructions_for_variant("full"),
        include_legal_moves=True,
        include_deadlock_status=True,
    ),
}

DEFAULT_TOOL_VARIANTS = {
    "move_only": ToolVariant(
        name="move_only",
        allowed_tools=["sokoban_move"],
        terminal_on_deadlock_override=True,
    ),
    "move_and_query": ToolVariant(
        name="move_and_query",
        allowed_tools=[
            "sokoban_move",
            "sokoban_get_state",
            "sokoban_is_solved",
            "sokoban_get_legal_moves",
        ],
        terminal_on_deadlock_override=True,
    ),
    "all_tools": ToolVariant(
        name="all_tools",
        allowed_tools=None,
        terminal_on_deadlock_override=False,
    ),
}


def default_sokoban_config() -> dict[str, Any]:
    return {
        "level_sets": ["starter-authored-v1"],
        "level_ids": None,
        "max_levels": 20,
        "max_optimal_moves": None,
        "runs_per_level": 1,
        "max_turns": 300,
        "prompt_variants": ["minimal"],
        "tool_variants": ["move_only"],
        "state_format": "text",
        "image_tile_size": 48,
        "image_labels": True,
        "image_background": "white",
        "detect_deadlocks": True,
        "terminal_on_deadlock": True,
        "record": False,
        "record_raw": False,
        "record_provider_raw": False,
        "provider_retries": 2,
        "provider_backoff": 1.0,
    }


def build_sokoban_adapter(env: SokobanEnv, **kwargs: Any) -> SokobanGameAdapter:
    return SokobanGameAdapter(env, **kwargs)


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


def _resolve_out_dir_base(base: str | Path, game_name: str) -> Path:
    base_str = str(base)
    if "{game}" in base_str:
        base_str = base_str.replace("{game}", game_name)
    path = Path(base_str)
    if path.name == game_name:
        return path
    return path / game_name


def _parse_str_list(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                result.append(chunk)
    return result


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    n = _as_float(numerator)
    d = _as_float(denominator)
    if n is None or d is None or d <= 0:
        return None
    return n / d


def _image_payload(image) -> dict[str, Any]:
    return {
        "mime_type": image.mime_type,
        "data_base64": image.data_base64,
        "data_url": image.data_url,
        "width": image.width,
        "height": image.height,
    }


def _dedupe_levels(levels: Iterable[SokobanLevel]) -> list[SokobanLevel]:
    seen: set[str] = set()
    deduped: list[SokobanLevel] = []
    for level in levels:
        if level.level_id in seen:
            continue
        seen.add(level.level_id)
        deduped.append(level)
    return deduped


def _select_levels(
    args: argparse.Namespace, config: dict[str, Any]
) -> list[SokobanLevel]:
    selected_level_ids = (
        _parse_str_list(getattr(args, "level_ids", None))
        if getattr(args, "level_ids", None) is not None
        else _parse_str_list(config.get("level_ids", []) or [])
    )
    if selected_level_ids:
        levels = [load_level_by_id(level_id) for level_id in selected_level_ids]
    else:
        selected_level_sets = (
            _parse_str_list(getattr(args, "level_sets", None))
            if getattr(args, "level_sets", None) is not None
            else _parse_str_list(config.get("level_sets", []) or [])
        )
        if not selected_level_sets:
            available_sets = list_bundled_level_sets()
            if not available_sets:
                raise SystemExit("No bundled Sokoban level sets available.")
            selected_level_sets = [available_sets[0]]

        levels = []
        for set_name in selected_level_sets:
            level_set = load_bundled_level_set(set_name)
            levels.extend(level_set.levels)

    levels = _dedupe_levels(levels)

    max_optimal_moves_arg = getattr(args, "max_optimal_moves", None)
    max_optimal_moves = (
        max_optimal_moves_arg
        if max_optimal_moves_arg is not None
        else config.get("max_optimal_moves")
    )
    if max_optimal_moves is not None:
        max_optimal_moves = int(max_optimal_moves)
        levels = [
            level
            for level in levels
            if level.known_optimal
            and level.optimal_moves is not None
            and level.optimal_moves <= max_optimal_moves
        ]

    max_levels_arg = getattr(args, "max_levels", None)
    max_levels = (
        max_levels_arg if max_levels_arg is not None else config.get("max_levels", 20)
    )
    if max_levels is not None:
        levels = levels[: int(max_levels)]

    if not levels:
        raise SystemExit("No Sokoban levels selected to run.")
    return levels


def _compute_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "episodes": 0,
            "solved": 0,
            "solve_rate": 0.0,
            "deadlocked": 0,
            "deadlock_rate": 0.0,
            "avg_moves": None,
            "avg_pushes": None,
            "avg_illegal_moves": None,
            "avg_tool_calls": None,
            "avg_boxes_on_goals_ratio": None,
            "avg_move_ratio": None,
            "n_with_optimal_moves": 0,
            "avg_push_ratio": None,
            "n_with_optimal_pushes": 0,
            "token_totals": None,
            "token_avgs": None,
            "cost_total": None,
            "cost_avg": None,
        }

    solved_count = sum(1 for ep in episodes if ep.get("solved"))
    deadlocked_count = sum(1 for ep in episodes if ep.get("deadlocked"))
    move_ratios = [
        ratio
        for ratio in (_as_float(ep.get("move_ratio")) for ep in episodes)
        if ratio is not None
    ]
    push_ratios = [
        ratio
        for ratio in (_as_float(ep.get("push_ratio")) for ep in episodes)
        if ratio is not None
    ]

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
        "solved": solved_count,
        "solve_rate": solved_count / len(episodes),
        "deadlocked": deadlocked_count,
        "deadlock_rate": deadlocked_count / len(episodes),
        "avg_moves": _mean(
            [
                value
                for value in (_as_float(ep.get("move_count")) for ep in episodes)
                if value is not None
            ]
        ),
        "avg_pushes": _mean(
            [
                value
                for value in (_as_float(ep.get("push_count")) for ep in episodes)
                if value is not None
            ]
        ),
        "avg_illegal_moves": _mean(
            [
                value
                for value in (_as_float(ep.get("illegal_moves")) for ep in episodes)
                if value is not None
            ]
        ),
        "avg_tool_calls": _mean(
            [
                value
                for value in (_as_float(ep.get("tool_calls")) for ep in episodes)
                if value is not None
            ]
        ),
        "avg_boxes_on_goals_ratio": _mean(
            [
                value
                for value in (
                    _as_float(ep.get("boxes_on_goals_ratio")) for ep in episodes
                )
                if value is not None
            ]
        ),
        "avg_move_ratio": _mean(move_ratios),
        "n_with_optimal_moves": len(move_ratios),
        "avg_push_ratio": _mean(push_ratios),
        "n_with_optimal_pushes": len(push_ratios),
        "token_totals": token_totals if token_count else None,
        "token_avgs": token_avgs,
        "cost_total": cost_total if cost_count else None,
        "cost_avg": (cost_total / cost_count) if cost_count else None,
    }


def _write_raw_generations(
    events: list[dict[str, Any]],
    *,
    out_file,
    episode_id: int,
    variant_id: str,
    instructions: str,
    tool_schemas_payload: list[dict[str, Any]],
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
                f"{json.dumps(tool_schemas_payload, indent=2)}"
            )
            current = {
                "episode_id": episode_id,
                "variant_id": variant_id,
                "turn_index": turn_index,
                "prompt": {
                    "instructions": instructions,
                    "state_text": state_text,
                    "tool_schemas": tool_schemas_payload,
                    "state_format": state_format,
                    "prompt_text": prompt_text,
                    "image": (
                        {"meta": last_image_meta, "config": image_config}
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


def add_sokoban_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level-set", action="append", dest="level_sets", default=None)
    parser.add_argument("--level-id", action="append", dest="level_ids", default=None)
    parser.add_argument("--max-levels", type=int, default=None)
    parser.add_argument("--max-optimal-moves", type=int, default=None)
    parser.add_argument("--runs-per-level", type=int, default=None)
    parser.add_argument(
        "--prompt-variant", action="append", dest="prompt_variants", default=None
    )
    parser.add_argument(
        "--tools-variant", action="append", dest="tool_variants", default=None
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
    parser.add_argument("--image-tile-size", type=int, default=None)
    parser.add_argument("--image-background", default=None)
    parser.add_argument("--image-labels", action="store_true")
    parser.add_argument("--no-image-labels", action="store_true")
    parser.add_argument(
        "--detect-deadlocks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable deadlock detection in the environment.",
    )
    parser.add_argument(
        "--terminal-on-deadlock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Terminate episode when deadlock is detected.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch benchmark for tool-calling Sokoban."
    )
    add_common_batch_arguments(parser)
    add_sokoban_arguments(parser)
    return parser


def run_batch(
    args: argparse.Namespace,
    config: dict[str, Any] | None,
    *,
    game_name: str = "sokoban",
) -> list[Path]:
    config = config or {}
    provider_name = getattr(args, "provider", None)
    if not provider_name:
        raise SystemExit("Missing required argument: --provider")

    model_arg = getattr(args, "model", None)
    models = _resolve_models(provider_name, config, model_arg)
    if not models:
        raise SystemExit("No models provided. Use --model or config.json.")

    levels = _select_levels(args, config)

    runs_per_level_arg = getattr(args, "runs_per_level", None)
    runs_per_level = (
        int(runs_per_level_arg)
        if runs_per_level_arg is not None
        else int(config.get("runs_per_level", 1))
    )
    max_turns_arg = getattr(args, "max_turns", None)
    max_turns = (
        int(max_turns_arg)
        if max_turns_arg is not None
        else int(config.get("max_turns", 300))
    )
    out_dir_base = getattr(args, "out_dir", None) or config.get(
        "out_dir", "artifacts/runs"
    )
    out_dir_base = _resolve_out_dir_base(out_dir_base, game_name)

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

    prompt_variants = DEFAULT_PROMPT_VARIANTS
    selected_prompt_names = getattr(args, "prompt_variants", None) or config.get(
        "prompt_variants", ["minimal"]
    )
    if not selected_prompt_names:
        raise SystemExit("No prompt variants selected.")
    unknown_prompt_variants = [
        name for name in selected_prompt_names if name not in prompt_variants
    ]
    if unknown_prompt_variants:
        raise SystemExit(
            "Unknown Sokoban prompt variant(s): "
            + ", ".join(sorted(set(unknown_prompt_variants)))
        )
    selected_prompt_variants = [prompt_variants[name] for name in selected_prompt_names]

    tool_variants = DEFAULT_TOOL_VARIANTS
    selected_tool_names = getattr(args, "tool_variants", None) or config.get(
        "tool_variants", ["move_only"]
    )
    if not selected_tool_names:
        raise SystemExit("No tool variants selected.")
    unknown_tool_variants = [
        name for name in selected_tool_names if name not in tool_variants
    ]
    if unknown_tool_variants:
        raise SystemExit(
            "Unknown Sokoban tool variant(s): "
            + ", ".join(sorted(set(unknown_tool_variants)))
        )
    selected_tool_variants = [tool_variants[name] for name in selected_tool_names]

    allowed_tools_override = getattr(args, "allowed_tools", None) or config.get(
        "allowed_tools"
    )
    if allowed_tools_override:
        selected_tool_variants = [
            ToolVariant(
                name="custom",
                allowed_tools=_parse_str_list([allowed_tools_override]),
                terminal_on_deadlock_override=None,
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
    image_tile_size_arg = getattr(args, "image_tile_size", None)
    image_tile_size = (
        int(image_tile_size_arg)
        if image_tile_size_arg is not None
        else int(config.get("image_tile_size", 48))
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

    detect_deadlocks_arg = getattr(args, "detect_deadlocks", None)
    detect_deadlocks = (
        bool(detect_deadlocks_arg)
        if detect_deadlocks_arg is not None
        else bool(config.get("detect_deadlocks", True))
    )
    terminal_on_deadlock_arg = getattr(args, "terminal_on_deadlock", None)
    terminal_on_deadlock = (
        bool(terminal_on_deadlock_arg)
        if terminal_on_deadlock_arg is not None
        else bool(config.get("terminal_on_deadlock", True))
    )

    for prompt_variant in selected_prompt_variants:
        for tool_variant in selected_tool_variants:
            if (
                prompt_variant.include_legal_moves
                and tool_variant.allowed_tools is not None
            ):
                if "sokoban_get_legal_moves" not in set(tool_variant.allowed_tools):
                    raise SystemExit(
                        f"Prompt variant '{prompt_variant.name}' requires tool "
                        "'sokoban_get_legal_moves', but it is unavailable in "
                        f"tools variant '{tool_variant.name}'."
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
            "level_ids": [level.level_id for level in levels],
            "runs_per_level": runs_per_level,
            "max_turns": max_turns,
            "prompt_variants": [asdict(v) for v in selected_prompt_variants],
            "tool_variants": [asdict(v) for v in selected_tool_variants],
            "tool_schemas": full_tool_schemas,
            "tool_schemas_by_variant": tool_schemas_by_variant,
            "state_format": state_format,
            "image_tile_size": image_tile_size,
            "image_labels": image_labels,
            "image_background": image_background,
            "detect_deadlocks": detect_deadlocks,
            "terminal_on_deadlock": terminal_on_deadlock,
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
            for level in levels:
                level_set_name = level.level_id.split(":", 1)[0]
                for prompt_variant in selected_prompt_variants:
                    for tool_variant in selected_tool_variants:
                        variant_id = (
                            f"level={level.level_id}__prompt={prompt_variant.name}"
                            f"__tools={tool_variant.name}"
                        )
                        effective_terminal_on_deadlock = terminal_on_deadlock
                        if tool_variant.terminal_on_deadlock_override is not None:
                            effective_terminal_on_deadlock = (
                                tool_variant.terminal_on_deadlock_override
                            )

                        for run_idx in range(runs_per_level):
                            env = SokobanEnv(
                                level,
                                record_history=True,
                                illegal_action_behavior="penalize",
                                detect_deadlocks=detect_deadlocks,
                                terminal_on_deadlock=effective_terminal_on_deadlock,
                            )
                            adapter = SokobanGameAdapter(env)
                            instructions = prompt_variant.instructions
                            if state_format in {"image", "both"}:
                                instructions = with_image_instructions(instructions)

                            state_formatter = (
                                lambda a, pv=prompt_variant: a.env.format_prompt_state(
                                    include_legal_moves=pv.include_legal_moves,
                                    include_deadlock_status=pv.include_deadlock_status,
                                )
                            )
                            if state_format == "image":
                                state_formatter = lambda _a: "State image attached."

                            state_image_renderer = None
                            if state_format in {"image", "both"}:

                                def state_image_renderer_payload(
                                    a,
                                    tile_size=image_tile_size,
                                    labels=image_labels,
                                    background=image_background,
                                ):
                                    image = render_sokoban_env_image(
                                        a.env,
                                        tile_size=tile_size,
                                        label_grid=labels,
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
                                variant_tool_schemas = (
                                    [
                                        t
                                        for t in tool_schemas()
                                        if t["name"] in tool_variant.allowed_tools
                                    ]
                                    if tool_variant.allowed_tools is not None
                                    else tool_schemas()
                                )
                                image_config = {
                                    "tile_size": image_tile_size,
                                    "labels": image_labels,
                                    "background": image_background,
                                }
                                _write_raw_generations(
                                    result.events,
                                    out_file=raw_file,
                                    episode_id=episode_id,
                                    variant_id=variant_id,
                                    instructions=instructions,
                                    tool_schemas_payload=variant_tool_schemas,
                                    state_format=state_format,
                                    image_config=image_config,
                                )

                            metrics = result.game_metrics
                            move_ratio = _ratio(
                                metrics.get("move_count"), metrics.get("optimal_moves")
                            )
                            push_ratio = _ratio(
                                metrics.get("push_count"), metrics.get("optimal_pushes")
                            )
                            boxes_on_goals_ratio = _ratio(
                                metrics.get("boxes_on_goals"), metrics.get("n_boxes")
                            )

                            episode = {
                                "episode_id": episode_id,
                                "variant_id": variant_id,
                                "run_idx": run_idx,
                                "provider": provider_name,
                                "model": model_name,
                                "level_id": metrics.get("level_id", level.level_id),
                                "level_set": level_set_name,
                                "prompt_variant": prompt_variant.name,
                                "tools_variant": tool_variant.name,
                                "n_boxes": metrics.get("n_boxes", level.n_boxes),
                                "grid_size": metrics.get("grid_size"),
                                "solved": result.solved,
                                "deadlocked": bool(metrics.get("deadlocked", False)),
                                "move_count": metrics.get(
                                    "move_count", result.move_count
                                ),
                                "push_count": metrics.get("push_count"),
                                "illegal_moves": result.illegal_moves,
                                "tool_calls": result.tool_calls,
                                "boxes_on_goals": metrics.get("boxes_on_goals"),
                                "boxes_on_goals_ratio": boxes_on_goals_ratio,
                                "optimal_moves": metrics.get("optimal_moves"),
                                "optimal_pushes": metrics.get("optimal_pushes"),
                                "known_optimal": bool(
                                    metrics.get("known_optimal", False)
                                ),
                                "move_ratio": move_ratio,
                                "push_ratio": push_ratio,
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
                                        "level_id": level.level_id,
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
    run_dirs = run_batch(args, config, game_name="sokoban")
    print(json.dumps({"run_dirs": [str(p) for p in run_dirs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
