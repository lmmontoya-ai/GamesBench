from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import random
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from games_bench.bench.common import (
    add_common_batch_arguments,
    resolve_interaction_mode,
    resolve_scoring_settings,
    resolve_spec_name,
)
from games_bench.bench.executor import run_episode_jobs
from games_bench.bench.lineage import ensure_run_manifest, make_lineage_event
from games_bench.bench.scoring import build_summary_document
from games_bench.bench.taxonomy import annotate_episode_with_taxonomy
from games_bench.config import load_config, merge_dicts, normalize_games_config
from games_bench.games.sokoban.adapter import SokobanGameAdapter
from games_bench.games.sokoban.env import SokobanEnv, SokobanLevel, tool_schemas
from games_bench.games.sokoban.level_loader import (
    list_bundled_level_sets,
    load_bundled_level_set,
    load_level_by_id,
)
from games_bench.games.sokoban.procgen import (
    generate_procedural_level,
    parse_grid_size,
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


@dataclass(frozen=True, slots=True)
class SokobanEpisodeJob:
    episode_id: int
    variant_id: str
    run_idx: int
    level: SokobanLevel
    level_set_name: str
    prompt_variant: PromptVariant
    tool_variant: ToolVariant
    effective_terminal_on_deadlock: bool


@dataclass(frozen=True, slots=True)
class SokobanEpisodeOutput:
    episode_id: int
    variant_id: str
    episode: dict[str, Any]
    events: list[dict[str, Any]]
    raw_lines: list[str]
    recording: dict[str, Any] | None


class _ThrottledProvider:
    def __init__(self, provider: Any, semaphore: threading.Semaphore | None) -> None:
        self._provider = provider
        self._semaphore = semaphore
        self.supports_images = bool(getattr(provider, "supports_images", False))

    def next_tool_calls(self, **kwargs: Any):
        if self._semaphore is None:
            return self._provider.next_tool_calls(**kwargs)
        with self._semaphore:
            return self._provider.next_tool_calls(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


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

_PROCGEN_PLUS_SCRAMBLE_SPAN = 60


def default_sokoban_config() -> dict[str, Any]:
    return {
        "spec": "sokoban-default",
        "level_sets": ["starter-authored-v1"],
        "level_ids": None,
        "max_levels": 20,
        "max_optimal_moves": None,
        "runs_per_level": 1,
        "max_turns": 300,
        "parallelism": 1,
        "max_inflight_provider": None,
        "stagnation_patience": None,
        "deadlock_patience": 8,
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
        "stream_debug": False,
        "procgen_cases": None,
        "procgen_grid_sizes": None,
        "procgen_box_counts": None,
        "procgen_levels_per_combo": 1,
        "procgen_seed": 0,
        "procgen_wall_density": 0.08,
        "procgen_scramble_steps": None,
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
    stream_debug: bool | None = None,
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
        model = model or _require_env("OPENROUTER_MODEL")
        return OpenRouterProvider(
            model=model,
            max_retries=int(retries),
            retry_backoff_s=float(backoff),
            stream_debug=bool(debug),
            timeout_s=int(getattr(args, "timeout_s", 300)),
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
    return [fallback or "default"]


def _resolve_out_dir_base(base: str | Path, game_name: str) -> Path:
    base_str = str(base)
    if "{game}" in base_str:
        base_str = base_str.replace("{game}", game_name)
    path = Path(base_str)
    if path.name == game_name:
        return path
    return path / game_name


def _parse_str_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            result.extend(_parse_str_list(value))
            continue
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                result.append(chunk)
    return result


def _parse_int_list(values: Iterable[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                result.append(int(chunk))
    return result


def _parse_scramble_steps_spec(value: Any) -> int | tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SystemExit("procgen scramble_steps must be an integer or range.")
    if isinstance(value, (int, float)):
        steps = int(value)
        if steps < 1:
            raise SystemExit("procgen scramble_steps must be >= 1.")
        return steps
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise SystemExit(
                "procgen scramble_steps ranges must contain exactly 2 values."
            )
        low = int(value[0])
        high = int(value[1])
        if low < 1 or high < low:
            raise SystemExit(
                "procgen scramble_steps range must satisfy 1 <= low <= high."
            )
        return (low, high)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text.endswith("+") and text[:-1].isdigit():
            low = int(text[:-1])
            if low < 1:
                raise SystemExit("procgen scramble_steps lower bound must be >= 1.")
            return (low, low + _PROCGEN_PLUS_SCRAMBLE_SPAN)
        if "-" in text:
            parts = text.split("-", 1)
            if (
                len(parts) == 2
                and parts[0].strip().isdigit()
                and parts[1].strip().isdigit()
            ):
                low = int(parts[0].strip())
                high = int(parts[1].strip())
                if low < 1 or high < low:
                    raise SystemExit(
                        "procgen scramble_steps range must satisfy 1 <= low <= high."
                    )
                return (low, high)
        if text.isdigit():
            steps = int(text)
            if steps < 1:
                raise SystemExit("procgen scramble_steps must be >= 1.")
            return steps
    raise SystemExit(
        "procgen scramble_steps must be an int, [low, high], 'low-high', or 'low+'."
    )


def _scramble_steps_to_json_value(value: int | tuple[int, int] | None) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return [int(value[0]), int(value[1])]
    return int(value)


def _sample_scramble_steps(
    value: int | tuple[int, int] | None, *, rng: random.Random
) -> int | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return int(rng.randint(int(value[0]), int(value[1])))
    return int(value)


def _validate_procgen_wall_density(value: float) -> float:
    if value < 0.0 or value > 0.35:
        raise SystemExit("procgen wall density must be within [0.0, 0.35].")
    return value


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


def _resolve_positive_int(
    arg_value: int | None,
    config: dict[str, Any],
    key: str,
    default: int,
) -> int:
    value = int(arg_value) if arg_value is not None else int(config.get(key, default))
    if value < 1:
        raise SystemExit(f"{key} must be >= 1, got {value}")
    return value


def _resolve_optional_positive_int(
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


def _resolve_parallel_settings(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    provider_name: str,
) -> tuple[int, int]:
    parallelism = _resolve_positive_int(
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


def _resolve_checkpoint_interval(
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


def _resolve_procgen_spec(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any] | None:
    procgen_cases_config = config.get("procgen_cases")
    procgen_grid_sizes_arg = getattr(args, "procgen_grid_sizes", None)
    procgen_box_counts_arg = getattr(args, "procgen_box_counts", None)

    procgen_grid_sizes = (
        _parse_str_list(procgen_grid_sizes_arg)
        if procgen_grid_sizes_arg is not None
        else _parse_str_list(config.get("procgen_grid_sizes", []) or [])
    )
    procgen_box_counts = (
        _parse_int_list(procgen_box_counts_arg)
        if procgen_box_counts_arg is not None
        else _parse_int_list(config.get("procgen_box_counts", []) or [])
    )

    if procgen_cases_config is not None and (
        procgen_grid_sizes_arg is not None
        or procgen_box_counts_arg is not None
        or procgen_grid_sizes
        or procgen_box_counts
    ):
        raise SystemExit(
            "Do not mix procgen_cases with procgen_grid_sizes/procgen_box_counts."
        )

    levels_per_combo_arg = getattr(args, "procgen_levels_per_combo", None)
    levels_per_combo_default = (
        int(levels_per_combo_arg)
        if levels_per_combo_arg is not None
        else int(config.get("procgen_levels_per_combo", 1))
    )
    if levels_per_combo_default < 1:
        raise SystemExit("procgen_levels_per_combo must be >= 1")

    procgen_seed_arg = getattr(args, "procgen_seed", None)
    procgen_seed = (
        int(procgen_seed_arg)
        if procgen_seed_arg is not None
        else config.get("procgen_seed", 0)
    )
    if procgen_seed is not None:
        procgen_seed = int(procgen_seed)

    procgen_wall_density_arg = getattr(args, "procgen_wall_density", None)
    procgen_wall_density_default = (
        float(procgen_wall_density_arg)
        if procgen_wall_density_arg is not None
        else float(config.get("procgen_wall_density", 0.08))
    )
    procgen_wall_density_default = _validate_procgen_wall_density(
        procgen_wall_density_default
    )

    procgen_scramble_steps_arg = getattr(args, "procgen_scramble_steps", None)
    procgen_scramble_steps_raw = (
        procgen_scramble_steps_arg
        if procgen_scramble_steps_arg is not None
        else config.get("procgen_scramble_steps")
    )
    procgen_scramble_steps_default = _parse_scramble_steps_spec(
        procgen_scramble_steps_raw
    )

    if procgen_cases_config is not None:
        if not isinstance(procgen_cases_config, list):
            raise SystemExit("procgen_cases must be a list of case objects.")
        if not procgen_cases_config:
            raise SystemExit("procgen_cases cannot be empty.")

        parsed_cases: list[dict[str, Any]] = []
        seen_cases: set[tuple[Any, ...]] = set()
        for case_index, raw_case in enumerate(procgen_cases_config):
            if not isinstance(raw_case, dict):
                raise SystemExit(
                    f"procgen_cases[{case_index}] must be an object with "
                    "'grid_size' and 'box_count'."
                )
            if "grid_size" not in raw_case or "box_count" not in raw_case:
                raise SystemExit(
                    f"procgen_cases[{case_index}] must include 'grid_size' and "
                    "'box_count'."
                )

            width, height = parse_grid_size(str(raw_case["grid_size"]))
            n_boxes = int(raw_case["box_count"])
            if n_boxes < 1:
                raise SystemExit("procgen case box_count must be >= 1")

            case_levels_per_combo = int(
                raw_case.get("levels_per_combo", levels_per_combo_default)
            )
            if case_levels_per_combo < 1:
                raise SystemExit("procgen case levels_per_combo must be >= 1")

            case_wall_density = _validate_procgen_wall_density(
                float(raw_case.get("wall_density", procgen_wall_density_default))
            )
            if "scramble_steps" in raw_case:
                case_scramble_steps = _parse_scramble_steps_spec(
                    raw_case.get("scramble_steps")
                )
            else:
                case_scramble_steps = procgen_scramble_steps_default

            case_key = (
                width,
                height,
                n_boxes,
                case_levels_per_combo,
                case_wall_density,
                case_scramble_steps,
            )
            if case_key in seen_cases:
                continue
            seen_cases.add(case_key)
            parsed_cases.append(
                {
                    "grid_size": (width, height),
                    "box_count": n_boxes,
                    "levels_per_combo": case_levels_per_combo,
                    "wall_density": case_wall_density,
                    "scramble_steps": case_scramble_steps,
                }
            )
        return {
            "mode": "cases",
            "cases": parsed_cases,
            "seed": procgen_seed,
        }

    if not procgen_grid_sizes and not procgen_box_counts:
        return None
    if not procgen_grid_sizes or not procgen_box_counts:
        raise SystemExit(
            "Procedural mode requires both procgen_grid_sizes and procgen_box_counts."
        )

    grid_sizes: list[tuple[int, int]] = []
    seen_grid_sizes: set[tuple[int, int]] = set()
    for raw in procgen_grid_sizes:
        parsed = parse_grid_size(raw)
        if parsed not in seen_grid_sizes:
            seen_grid_sizes.add(parsed)
            grid_sizes.append(parsed)

    box_counts: list[int] = []
    seen_box_counts: set[int] = set()
    for n_boxes in procgen_box_counts:
        if n_boxes < 1:
            raise SystemExit("procgen box counts must be >= 1")
        if n_boxes not in seen_box_counts:
            seen_box_counts.add(n_boxes)
            box_counts.append(n_boxes)

    cases: list[dict[str, Any]] = []
    for width, height in grid_sizes:
        for n_boxes in box_counts:
            cases.append(
                {
                    "grid_size": (width, height),
                    "box_count": n_boxes,
                    "levels_per_combo": levels_per_combo_default,
                    "wall_density": procgen_wall_density_default,
                    "scramble_steps": procgen_scramble_steps_default,
                }
            )

    return {
        "mode": "grid_box_product",
        "cases": cases,
        "seed": procgen_seed,
    }


def _select_procgen_levels(spec: dict[str, Any]) -> list[SokobanLevel]:
    levels: list[SokobanLevel] = []
    for combo_idx, case in enumerate(spec["cases"]):
        width, height = case["grid_size"]
        n_boxes = int(case["box_count"])
        levels_per_combo = int(case["levels_per_combo"])
        wall_density = float(case["wall_density"])
        scramble_steps_spec = case["scramble_steps"]

        combo_seed = spec["seed"]
        if combo_seed is not None:
            combo_seed = int(combo_seed) + (combo_idx * 10_000)

        scramble_label = _scramble_steps_to_json_value(scramble_steps_spec)
        if scramble_label is None:
            scramble_tag = "default"
        elif isinstance(scramble_label, list):
            scramble_tag = f"{scramble_label[0]}-{scramble_label[1]}"
        else:
            scramble_tag = str(scramble_label)
        level_id_prefix = (
            f"procgen:{width}x{height}:b{n_boxes}:sc{scramble_tag}:s"
            f"{combo_seed if combo_seed is not None else 'random'}"
        )
        combo_rng = random.Random(combo_seed)

        for level_idx in range(levels_per_combo):
            level_seed = None if combo_seed is None else int(combo_seed) + level_idx
            scramble_steps = _sample_scramble_steps(
                scramble_steps_spec,
                rng=combo_rng,
            )
            try:
                levels.append(
                    generate_procedural_level(
                        width=width,
                        height=height,
                        n_boxes=n_boxes,
                        seed=level_seed,
                        level_id=f"{level_id_prefix}:i{level_idx + 1}",
                        title=(
                            f"Procedural {width}x{height} ({n_boxes} boxes) "
                            f"#{level_idx + 1}"
                        ),
                        wall_density=wall_density,
                        scramble_steps=scramble_steps,
                    )
                )
            except (RuntimeError, ValueError) as exc:
                raise SystemExit(
                    "Failed to generate procedural Sokoban levels for "
                    f"{width}x{height}, n_boxes={n_boxes}, "
                    f"scramble={scramble_tag}: {exc}"
                ) from exc
    return _dedupe_levels(levels)


def _select_levels(
    args: argparse.Namespace, config: dict[str, Any]
) -> list[SokobanLevel]:
    procgen_spec = _resolve_procgen_spec(args, config)
    if procgen_spec is not None:
        explicit_level_ids = getattr(args, "level_ids", None)
        explicit_level_sets = getattr(args, "level_sets", None)
        if explicit_level_ids is not None or explicit_level_sets is not None:
            raise SystemExit(
                "Do not mix --level-id/--level-set with procedural generation flags."
            )
        levels = _select_procgen_levels(procgen_spec)
        max_levels_arg = getattr(args, "max_levels", None)
        max_levels = (
            max_levels_arg
            if max_levels_arg is not None
            else config.get("max_levels", 20)
        )
        if max_levels is not None:
            levels = levels[: int(max_levels)]
        if not levels:
            raise SystemExit("No procedural Sokoban levels selected to run.")
        return levels

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


def _merge_config_for_game(
    raw_config: dict[str, Any] | None,
    *,
    game_name: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    global_defaults, games_map = normalize_games_config(
        raw_config or {}, default_game=game_name
    )
    return merge_dicts(
        defaults, merge_dicts(global_defaults, games_map.get(game_name, {}))
    )


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
    solved_episodes = [ep for ep in episodes if ep.get("solved")]
    move_ratios = [
        ratio
        for ratio in (_as_float(ep.get("move_ratio")) for ep in solved_episodes)
        if ratio is not None
    ]
    push_ratios = [
        ratio
        for ratio in (_as_float(ep.get("push_ratio")) for ep in solved_episodes)
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


def score_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return _compute_metrics(episodes)


def compare_metrics(summary: dict[str, Any]) -> dict[str, float]:
    overall = summary.get("overall", summary)
    if not isinstance(overall, dict):
        return {}
    metrics: dict[str, float] = {}
    for name, value in overall.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            candidate = float(value)
            if math.isfinite(candidate):
                metrics[str(name)] = candidate
    return metrics


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


def _raw_lines_for_events(
    *,
    events: list[dict[str, Any]],
    episode_id: int,
    variant_id: str,
    instructions: str,
    tool_schemas_payload: list[dict[str, Any]],
    state_format: str,
    image_config: dict[str, Any],
) -> list[str]:
    buffer = io.StringIO()
    _write_raw_generations(
        events,
        out_file=buffer,
        episode_id=episode_id,
        variant_id=variant_id,
        instructions=instructions,
        tool_schemas_payload=tool_schemas_payload,
        state_format=state_format,
        image_config=image_config,
    )
    return [line for line in buffer.getvalue().splitlines() if line]


def _run_sokoban_episode_job(
    job: SokobanEpisodeJob,
    *,
    provider: Any,
    provider_name: str,
    model_name: str,
    spec_name: str,
    interaction_mode: str,
    stateless: bool,
    max_turns: int,
    state_format: str,
    image_tile_size: int,
    image_labels: bool,
    image_background: str,
    detect_deadlocks: bool,
    record_provider_raw: bool,
    record_raw: bool,
    record: bool,
    stagnation_patience: int | None,
    deadlock_patience: int | None,
) -> SokobanEpisodeOutput:
    env = SokobanEnv(
        job.level,
        record_history=True,
        illegal_action_behavior="penalize",
        detect_deadlocks=detect_deadlocks,
        terminal_on_deadlock=job.effective_terminal_on_deadlock,
    )
    adapter = SokobanGameAdapter(env)
    instructions = job.prompt_variant.instructions
    if state_format in {"image", "both"}:
        instructions = with_image_instructions(instructions)

    state_formatter = lambda a, pv=job.prompt_variant: a.env.format_prompt_state(
        include_legal_moves=pv.include_legal_moves,
        include_deadlock_status=pv.include_deadlock_status,
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

    deadlock_checker = None
    if detect_deadlocks:
        deadlock_checker = lambda a: bool(a.env.is_deadlocked())

    result = run_tool_calling_episode(
        adapter,
        provider,
        max_turns=max_turns,
        instructions=instructions,
        state_formatter=state_formatter,
        state_image_renderer=state_image_renderer,
        allowed_tools=job.tool_variant.allowed_tools,
        record_provider_raw=record_provider_raw,
        stagnation_patience=stagnation_patience,
        deadlock_patience=deadlock_patience,
        deadlock_checker=deadlock_checker,
        deadlock_terminate_on_check=job.effective_terminal_on_deadlock,
        stateless=stateless,
    )
    terminated_early = bool(getattr(result, "terminated_early", False))
    termination_reason = getattr(result, "termination_reason", None)
    turn_count = sum(
        1 for event in result.events if event.get("type") == "provider_result"
    )
    provider_error_count = sum(
        1
        for event in result.events
        if event.get("type") == "provider_result" and event.get("error")
    )

    metrics = result.game_metrics
    move_ratio = (
        _ratio(
            metrics.get("move_count"),
            metrics.get("optimal_moves"),
        )
        if result.solved
        else None
    )
    push_ratio = (
        _ratio(
            metrics.get("push_count"),
            metrics.get("optimal_pushes"),
        )
        if result.solved
        else None
    )
    boxes_on_goals_ratio = _ratio(metrics.get("boxes_on_goals"), metrics.get("n_boxes"))
    episode = annotate_episode_with_taxonomy(
        {
            "episode_id": job.episode_id,
            "game": "sokoban",
            "variant_id": job.variant_id,
            "run_idx": job.run_idx,
            "provider": provider_name,
            "model": model_name,
            "spec": spec_name,
            "interaction_mode": interaction_mode,
            "stateless": stateless,
            "level_id": metrics.get("level_id", job.level.level_id),
            "level_set": job.level_set_name,
            "prompt_variant": job.prompt_variant.name,
            "tools_variant": job.tool_variant.name,
            "n_boxes": metrics.get("n_boxes", job.level.n_boxes),
            "grid_size": metrics.get("grid_size"),
            "solved": result.solved,
            "deadlocked": bool(metrics.get("deadlocked", False)),
            "turn_count": turn_count,
            "move_count": metrics.get("move_count", result.move_count),
            "push_count": metrics.get("push_count"),
            "illegal_moves": result.illegal_moves,
            "tool_calls": result.tool_calls,
            "boxes_on_goals": metrics.get("boxes_on_goals"),
            "boxes_on_goals_ratio": boxes_on_goals_ratio,
            "optimal_moves": metrics.get("optimal_moves"),
            "optimal_pushes": metrics.get("optimal_pushes"),
            "known_optimal": bool(metrics.get("known_optimal", False)),
            "move_ratio": move_ratio,
            "push_ratio": push_ratio,
            "terminated_early": terminated_early,
            "termination_reason": termination_reason,
            "provider_error_count": provider_error_count,
            "usage": result.usage,
            "cost": result.cost,
        },
        game_name="sokoban",
    )

    recording = None
    if record:
        recording = build_recording(
            events=result.events,
            metadata={
                "episode_id": job.episode_id,
                "variant_id": job.variant_id,
                "run_idx": job.run_idx,
                "provider": provider_name,
                "model": model_name,
                "spec": spec_name,
                "interaction_mode": interaction_mode,
                "stateless": stateless,
                "level_id": job.level.level_id,
                "prompt_variant": job.prompt_variant.name,
                "tools_variant": job.tool_variant.name,
                "solved": result.solved,
                "terminated_early": terminated_early,
                "termination_reason": termination_reason,
            },
        )

    raw_lines: list[str] = []
    if record_raw:
        variant_tool_schemas = (
            [t for t in tool_schemas() if t["name"] in job.tool_variant.allowed_tools]
            if job.tool_variant.allowed_tools is not None
            else tool_schemas()
        )
        raw_lines = _raw_lines_for_events(
            events=result.events,
            episode_id=job.episode_id,
            variant_id=job.variant_id,
            instructions=instructions,
            tool_schemas_payload=variant_tool_schemas,
            state_format=state_format,
            image_config={
                "tile_size": image_tile_size,
                "labels": image_labels,
                "background": image_background,
            },
        )

    return SokobanEpisodeOutput(
        episode_id=job.episode_id,
        variant_id=job.variant_id,
        episode=episode,
        events=result.events,
        raw_lines=raw_lines,
        recording=recording,
    )


def add_sokoban_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level-set", action="append", dest="level_sets", default=None)
    parser.add_argument("--level-id", action="append", dest="level_ids", default=None)
    parser.add_argument(
        "--procgen-grid-size", action="append", dest="procgen_grid_sizes", default=None
    )
    parser.add_argument(
        "--procgen-box-count", action="append", dest="procgen_box_counts", default=None
    )
    parser.add_argument("--procgen-levels-per-combo", type=int, default=None)
    parser.add_argument("--procgen-seed", type=int, default=None)
    parser.add_argument("--procgen-wall-density", type=float, default=None)
    parser.add_argument("--procgen-scramble-steps", type=int, default=None)
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
    parser.add_argument(
        "--deadlock-patience",
        type=int,
        default=None,
        help=(
            "Early-stop after N consecutive turns while env.is_deadlocked() is true. "
            "Disabled when unset."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch benchmark for tool-calling Sokoban."
    )
    add_common_batch_arguments(parser)
    add_sokoban_arguments(parser)
    return parser


def estimate_episodes(
    args: argparse.Namespace,
    config: dict[str, Any] | None,
    *,
    game_name: str = "sokoban",  # noqa: ARG001
) -> int:
    config = config or {}
    provider_name = getattr(args, "provider", None)
    if not provider_name:
        raise SystemExit("Missing required argument: --provider")
    model_arg = getattr(args, "model", None)
    models = _resolve_models(provider_name, config, model_arg)
    if not models:
        raise SystemExit("No models provided. Use --model or config.json.")

    levels = _select_levels(args, config)
    runs_per_level = _resolve_positive_int(
        getattr(args, "runs_per_level", None), config, "runs_per_level", 1
    )

    prompt_variants = DEFAULT_PROMPT_VARIANTS
    selected_prompt_names = _parse_str_list(
        [
            getattr(args, "prompt_variants", None)
            or config.get("prompt_variants", ["minimal"])
        ]
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
    selected_tool_names = _parse_str_list(
        [
            getattr(args, "tool_variants", None)
            or config.get("tool_variants", ["move_only"])
        ]
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
        allowed_tools = _parse_str_list([allowed_tools_override])
        if not allowed_tools:
            raise SystemExit("allowed_tools override must include at least one tool.")
        selected_tool_variants = [
            ToolVariant(
                name="custom",
                allowed_tools=allowed_tools,
                terminal_on_deadlock_override=None,
            )
        ]

    for prompt_variant in selected_prompt_variants:
        for tool_variant in selected_tool_variants:
            if (
                prompt_variant.include_legal_moves
                and tool_variant.allowed_tools is not None
                and "sokoban_get_legal_moves" not in set(tool_variant.allowed_tools)
            ):
                raise SystemExit(
                    f"Prompt variant '{prompt_variant.name}' requires tool "
                    "'sokoban_get_legal_moves', but it is unavailable in "
                    f"tools variant '{tool_variant.name}'."
                )

    episodes_per_model = (
        len(levels)
        * len(selected_prompt_variants)
        * len(selected_tool_variants)
        * runs_per_level
    )
    return len(models) * episodes_per_model


def run_batch(
    args: argparse.Namespace,
    config: dict[str, Any] | None,
    *,
    game_name: str = "sokoban",
) -> list[Path]:
    config = config or {}
    progress_reporter = getattr(args, "_progress_reporter", None)
    provider_name = getattr(args, "provider", None)
    if not provider_name:
        raise SystemExit("Missing required argument: --provider")

    model_arg = getattr(args, "model", None)
    models = _resolve_models(provider_name, config, model_arg)
    if not models:
        raise SystemExit("No models provided. Use --model or config.json.")

    procgen_spec = _resolve_procgen_spec(args, config)
    levels = _select_levels(args, config)

    runs_per_level = _resolve_positive_int(
        getattr(args, "runs_per_level", None), config, "runs_per_level", 1
    )
    max_turns = _resolve_positive_int(
        getattr(args, "max_turns", None), config, "max_turns", 300
    )
    parallelism, max_inflight_provider = _resolve_parallel_settings(
        args=args,
        config=config,
        provider_name=provider_name,
    )
    stateless, interaction_mode = resolve_interaction_mode(args, config)
    spec_base, spec_name = resolve_spec_name(
        args,
        config,
        interaction_mode=interaction_mode,
    )
    score_enabled, score_version = resolve_scoring_settings(args, config)
    stagnation_patience = _resolve_optional_positive_int(
        getattr(args, "stagnation_patience", None),
        config,
        "stagnation_patience",
    )
    deadlock_patience = _resolve_optional_positive_int(
        getattr(args, "deadlock_patience", None),
        config,
        "deadlock_patience",
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
    stream_debug_arg = getattr(args, "stream_debug", None)
    stream_debug = (
        bool(stream_debug_arg)
        if stream_debug_arg is not None
        else bool(config.get("stream_debug", False))
    )

    prompt_variants = DEFAULT_PROMPT_VARIANTS
    selected_prompt_names = _parse_str_list(
        [
            getattr(args, "prompt_variants", None)
            or config.get("prompt_variants", ["minimal"])
        ]
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
    selected_tool_names = _parse_str_list(
        [
            getattr(args, "tool_variants", None)
            or config.get("tool_variants", ["move_only"])
        ]
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
        allowed_tools = _parse_str_list([allowed_tools_override])
        if not allowed_tools:
            raise SystemExit("allowed_tools override must include at least one tool.")
        selected_tool_variants = [
            ToolVariant(
                name="custom",
                allowed_tools=allowed_tools,
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
    run_id_override = getattr(args, "run_id", None) or config.get("run_id")
    resume = bool(getattr(args, "resume", False)) or bool(config.get("resume", False))
    strict_resume = bool(getattr(args, "strict_resume", False)) or bool(
        config.get("strict_resume", False)
    )
    checkpoint_interval = _resolve_checkpoint_interval(args, config)
    if resume and not run_id_override:
        raise SystemExit("--resume requires --run-id (or config run_id).")

    run_dirs: list[Path] = []
    for model_name in models:
        model_slug = model_name.replace("/", "_").replace(":", "_")
        default_run_id = f"{timestamp}_{provider_name}_{model_slug}"
        if run_id_override:
            run_id = (
                str(run_id_override)
                if len(models) == 1
                else f"{run_id_override}_{provider_name}_{model_slug}"
            )
        else:
            run_id = default_run_id

        out_dir = Path(out_dir_base) / provider_name / model_slug / run_id
        run_config_path = out_dir / "run_config.json"
        if resume:
            if not out_dir.exists():
                raise SystemExit(
                    "Resume requested but run directory does not exist: "
                    f"{out_dir}. Check --run-id."
                )
            if not out_dir.is_dir():
                raise SystemExit(
                    f"Resume requested but path is not a directory: {out_dir}"
                )
            if not run_config_path.exists():
                raise SystemExit(
                    "Resume requested but run_config.json is missing in "
                    f"{out_dir}. Check --run-id or rerun without --resume."
                )
        else:
            if out_dir.exists() and any(out_dir.iterdir()):
                raise SystemExit(
                    f"Run directory already exists: {out_dir}. "
                    "Use --resume to continue or choose a different --run-id."
                )
            out_dir.mkdir(parents=True, exist_ok=True)

        provider_semaphore = threading.BoundedSemaphore(max_inflight_provider)
        provider_local = threading.local()

        def get_provider() -> Any:
            provider = getattr(provider_local, "provider", None)
            if provider is None:
                base_provider = _build_provider(
                    args,
                    model_name,
                    provider_retries=provider_retries,
                    provider_backoff=provider_backoff,
                    stream_debug=stream_debug,
                )
                provider = _ThrottledProvider(base_provider, provider_semaphore)
                provider_local.provider = provider
            return provider

        preflight_provider = get_provider()
        if state_format in {"image", "both"} and not getattr(
            preflight_provider, "supports_images", False
        ):
            raise SystemExit(
                f"Provider '{provider_name}' does not support state_format='{state_format}'. "
                "Use --state-format text or a provider with image support."
            )

        full_tool_schemas = tool_schemas()
        tool_schemas_by_variant = {
            variant.name: (
                [t for t in full_tool_schemas if t["name"] in variant.allowed_tools]
                if variant.allowed_tools is not None
                else full_tool_schemas
            )
            for variant in selected_tool_variants
        }

        procgen_payload: dict[str, Any]
        if procgen_spec is None:
            procgen_payload = {"enabled": False}
        else:
            procgen_cases_payload = [
                {
                    "grid_size": f"{case['grid_size'][0]}x{case['grid_size'][1]}",
                    "box_count": int(case["box_count"]),
                    "levels_per_combo": int(case["levels_per_combo"]),
                    "wall_density": float(case["wall_density"]),
                    "scramble_steps": _scramble_steps_to_json_value(
                        case["scramble_steps"]
                    ),
                }
                for case in procgen_spec["cases"]
            ]
            grid_sizes = list(
                dict.fromkeys(case["grid_size"] for case in procgen_spec["cases"])
            )
            box_counts = list(
                dict.fromkeys(int(case["box_count"]) for case in procgen_spec["cases"])
            )
            procgen_payload = {
                "enabled": True,
                "mode": procgen_spec.get("mode", "grid_box_product"),
                "grid_sizes": [f"{width}x{height}" for width, height in grid_sizes],
                "box_counts": box_counts,
                "seed": procgen_spec["seed"],
                "cases": procgen_cases_payload,
            }

        if resume:
            run_config = json.loads(run_config_path.read_text())
            if strict_resume:
                for key in (
                    "run_id",
                    "game",
                    "provider",
                    "model",
                    "spec",
                    "interaction_mode",
                    "stateless",
                ):
                    current = {
                        "run_id": run_id,
                        "game": game_name,
                        "provider": provider_name,
                        "model": model_name,
                        "spec": spec_name,
                        "interaction_mode": interaction_mode,
                        "stateless": stateless,
                    }[key]
                    if run_config.get(key) != current:
                        raise SystemExit(
                            f"Strict resume mismatch for '{key}': "
                            f"existing={run_config.get(key)!r}, current={current!r}"
                        )
        else:
            run_config = {
                "run_id": run_id,
                "timestamp": timestamp,
                "spec_base": spec_base,
                "spec": spec_name,
                "interaction_mode": interaction_mode,
                "stateless": stateless,
                "game": game_name,
                "provider": provider_name,
                "model": model_name,
                "level_ids": [level.level_id for level in levels],
                "level_source": "procgen" if procgen_spec is not None else "bundled",
                "procgen": procgen_payload,
                "runs_per_level": runs_per_level,
                "max_turns": max_turns,
                "parallelism": parallelism,
                "max_inflight_provider": max_inflight_provider,
                "stagnation_patience": stagnation_patience,
                "deadlock_patience": deadlock_patience,
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
                "stream_debug": stream_debug,
                "score_enabled": score_enabled,
                "score_version": score_version,
                "resume": resume,
                "strict_resume": strict_resume,
                "checkpoint_interval": checkpoint_interval,
                "python": platform.python_version(),
                "platform": platform.platform(),
            }
            run_config_path.write_text(json.dumps(run_config, indent=2))

        ensure_run_manifest(
            out_dir,
            run_config=run_config,
            game_config=config,
            parent_run_id=(str(run_config.get("run_id") or run_id) if resume else None),
            lineage_event=(
                make_lineage_event(
                    "resume",
                    payload={"run_id": str(run_config.get("run_id") or run_id)},
                )
                if resume
                else None
            ),
        )

        jobs: list[SokobanEpisodeJob] = []
        episode_id = 0
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
                        jobs.append(
                            SokobanEpisodeJob(
                                episode_id=episode_id,
                                variant_id=variant_id,
                                run_idx=run_idx,
                                level=level,
                                level_set_name=level_set_name,
                                prompt_variant=prompt_variant,
                                tool_variant=tool_variant,
                                effective_terminal_on_deadlock=effective_terminal_on_deadlock,
                            )
                        )
                        episode_id += 1

        def run_job(job: SokobanEpisodeJob) -> SokobanEpisodeOutput:
            return _run_sokoban_episode_job(
                job,
                provider=get_provider(),
                provider_name=provider_name,
                model_name=model_name,
                spec_name=spec_name,
                interaction_mode=interaction_mode,
                stateless=stateless,
                max_turns=max_turns,
                state_format=state_format,
                image_tile_size=image_tile_size,
                image_labels=image_labels,
                image_background=image_background,
                detect_deadlocks=detect_deadlocks,
                record_provider_raw=record_provider_raw,
                record_raw=record_raw,
                record=record,
                stagnation_patience=stagnation_patience,
                deadlock_patience=deadlock_patience,
            )

        episodes = run_episode_jobs(
            out_dir=out_dir,
            run_id=str(run_config.get("run_id", run_id)),
            jobs=jobs,
            run_job=run_job,
            parallelism=parallelism,
            record=record,
            record_raw=record_raw,
            progress_reporter=progress_reporter,
            resume=resume,
            strict_resume=strict_resume,
            checkpoint_interval=checkpoint_interval,
        )

        if score_enabled:
            summary = build_summary_document(
                run_config=run_config,
                episodes=episodes,
                score_episodes=score_episodes,
                score_version=score_version,
                game_name=game_name,
                scoring_input={
                    "source": "inline",
                    "episodes_count": len(episodes),
                },
            )
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        run_dirs.append(out_dir)
    return run_dirs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    raw_config = load_config(args.config) if args.config else {}
    config = _merge_config_for_game(
        raw_config,
        game_name="sokoban",
        defaults=default_sokoban_config(),
    )
    run_dirs = run_batch(args, config, game_name="sokoban")
    print(json.dumps({"run_dirs": [str(p) for p in run_dirs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
