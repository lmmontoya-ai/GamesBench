from __future__ import annotations

import argparse
import io
import json
import math
import platform
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
from games_bench.bench.runner_shared import (
    build_provider as _shared_build_provider,
    parse_int_list as _shared_parse_int_list,
    parse_str_list as _shared_parse_str_list,
    resolve_checkpoint_interval as _shared_resolve_checkpoint_interval,
    resolve_models as _shared_resolve_models,
    resolve_optional_positive_int as _shared_resolve_optional_positive_int,
    resolve_out_dir_base as _shared_resolve_out_dir_base,
    resolve_parallel_settings as _shared_resolve_parallel_settings,
    resolve_positive_int as _shared_resolve_positive_int,
    require_env as _shared_require_env,
)
from games_bench.bench.scoring import build_summary_document
from games_bench.bench.taxonomy import annotate_episode_with_taxonomy
from games_bench.games.hanoi.adapter import HanoiGameAdapter
from games_bench.games.hanoi.env import TowerOfHanoiEnv, tool_schemas
from games_bench.games.hanoi.prompts import (
    DEFAULT_TEMPLATE,
    format_instructions,
    with_image_instructions,
)
from games_bench.games.hanoi.vision import render_hanoi_env_image
from games_bench.llm import (
    build_recording,
    run_tool_calling_episode,
)
from games_bench.config import load_config, merge_dicts, normalize_games_config


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


@dataclass(frozen=True, slots=True)
class HanoiCase:
    n_pegs: int
    n_disks: int
    start_peg: int
    goal_peg: int


@dataclass(frozen=True, slots=True)
class HanoiEpisodeJob:
    episode_id: int
    variant_id: str
    run_idx: int
    case: HanoiCase
    prompt_variant: PromptVariant
    tool_variant: ToolVariant


@dataclass(frozen=True, slots=True)
class HanoiEpisodeOutput:
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
        instructions=DEFAULT_TEMPLATE,
        include_legal_moves=False,
        include_action_space=False,
    ),
    "legal_moves": PromptVariant(
        name="legal_moves",
        instructions=DEFAULT_TEMPLATE,
        include_legal_moves=True,
        include_action_space=False,
    ),
    "action_space": PromptVariant(
        name="action_space",
        instructions=DEFAULT_TEMPLATE,
        include_legal_moves=False,
        include_action_space=True,
    ),
    "full": PromptVariant(
        name="full",
        instructions=DEFAULT_TEMPLATE,
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
        "spec": "hanoi-default",
        "cases": None,
        "n_pegs": [3],
        "n_disks": [3],
        "runs_per_variant": 3,
        "max_turns": 200,
        "parallelism": 1,
        "max_inflight_provider": None,
        "stagnation_patience": None,
        "optimal_turn_cap_multiplier": 4.0,
        "start_peg": 0,
        "goal_peg": None,
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
        "stream_debug": False,
    }


def build_hanoi_adapter(env: TowerOfHanoiEnv, **kwargs: Any) -> HanoiGameAdapter:
    return HanoiGameAdapter(env, **kwargs)


def _require_env(name: str) -> str:
    return _shared_require_env(name)


def _build_provider(
    args: argparse.Namespace,
    model: str | None,
    *,
    provider_retries: int | None = None,
    provider_backoff: float | None = None,
    stream_debug: bool | None = None,
) -> Any:
    return _shared_build_provider(
        args,
        model,
        provider_retries=provider_retries,
        provider_backoff=provider_backoff,
        stream_debug=stream_debug,
    )


def _parse_str_list(values: Iterable[Any]) -> list[str]:
    return _shared_parse_str_list(values)


def _parse_int_list(values: Iterable[str]) -> list[int]:
    return _shared_parse_int_list(values)


def _parse_case_token(value: str) -> tuple[int, int]:
    token = value.strip().lower()
    if "x" in token:
        parts = token.split("x")
    elif "," in token:
        parts = token.split(",")
    else:
        raise SystemExit("Invalid --case value. Expected NPEGSxNDISKS or NPEGS,NDISKS.")
    if len(parts) != 2:
        raise SystemExit("Invalid --case value. Expected NPEGSxNDISKS or NPEGS,NDISKS.")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise SystemExit(
            "Invalid --case value. Expected integer NPEGS and NDISKS."
        ) from exc


def _parse_cases_config(
    raw_cases: Any,
) -> list[tuple[int, int, int | None, int | None]]:
    if not isinstance(raw_cases, list):
        raise SystemExit("cases must be a list of case entries.")

    parsed: list[tuple[int, int, int | None, int | None]] = []
    for entry in raw_cases:
        if isinstance(entry, str):
            n_pegs, n_disks = _parse_case_token(entry)
            parsed.append((n_pegs, n_disks, None, None))
            continue
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            n_pegs = int(entry[0])
            n_disks = int(entry[1])
            parsed.append((n_pegs, n_disks, None, None))
            continue
        if isinstance(entry, dict):
            if "n_pegs" not in entry or "n_disks" not in entry:
                raise SystemExit(
                    "Each case object must include 'n_pegs' and 'n_disks'."
                )
            n_pegs = int(entry["n_pegs"])
            n_disks = int(entry["n_disks"])
            start_peg = (
                None if entry.get("start_peg") is None else int(entry["start_peg"])
            )
            goal_peg = None if entry.get("goal_peg") is None else int(entry["goal_peg"])
            parsed.append((n_pegs, n_disks, start_peg, goal_peg))
            continue
        raise SystemExit(
            "Each case must be a string, 2-item list/tuple, or object with "
            "'n_pegs' and 'n_disks'."
        )
    return parsed


def _resolve_start_goal_pegs(
    *,
    n_pegs: int,
    start_peg_value: int | None,
    goal_peg_value: int | None,
) -> tuple[int, int]:
    if n_pegs < 3:
        raise SystemExit(f"n_pegs must be >= 3, got {n_pegs}")

    start_peg = 0 if start_peg_value is None else int(start_peg_value)
    goal_peg = (n_pegs - 1) if goal_peg_value is None else int(goal_peg_value)
    max_idx = n_pegs - 1

    if start_peg < 0 or start_peg >= n_pegs:
        raise SystemExit(
            f"start_peg={start_peg} is out of range for n_pegs={n_pegs} "
            f"(valid: 0..{max_idx})."
        )
    if goal_peg < 0 or goal_peg >= n_pegs:
        raise SystemExit(
            f"goal_peg={goal_peg} is out of range for n_pegs={n_pegs} "
            f"(valid: 0..{max_idx})."
        )
    if start_peg == goal_peg:
        raise SystemExit("start_peg and goal_peg must be different.")
    return (start_peg, goal_peg)


def _resolve_models(
    provider: str, config: dict[str, Any] | None, fallback: str | None
) -> list[str]:
    return _shared_resolve_models(provider, config, fallback)


def _load_prompt_variants(path: str) -> dict[str, PromptVariant]:
    data = json.loads(Path(path).read_text())
    variants: dict[str, PromptVariant] = {}
    for item in data:
        name = item["name"]
        variants[name] = PromptVariant(
            name=name,
            instructions=item.get("instructions", DEFAULT_TEMPLATE),
            include_legal_moves=bool(item.get("include_legal_moves", False)),
            include_action_space=bool(item.get("include_action_space", False)),
        )
    return variants


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


def _parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid image size: {value}")
    return int(parts[0]), int(parts[1])


def _resolve_positive_int(
    arg_value: int | None,
    config: dict[str, Any],
    key: str,
    default: int,
) -> int:
    return _shared_resolve_positive_int(arg_value, config, key, default)


def _resolve_optional_positive_int(
    arg_value: int | None,
    config: dict[str, Any],
    key: str,
) -> int | None:
    return _shared_resolve_optional_positive_int(arg_value, config, key)


def _resolve_parallel_settings(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    provider_name: str,
) -> tuple[int, int]:
    return _shared_resolve_parallel_settings(
        args=args,
        config=config,
        provider_name=provider_name,
    )


def _resolve_optional_positive_float(
    arg_value: float | None,
    config: dict[str, Any],
    key: str,
) -> float | None:
    value = arg_value if arg_value is not None else config.get(key)
    if value is None:
        return None
    resolved = float(value)
    if resolved <= 0.0:
        raise SystemExit(f"{key} must be > 0, got {resolved}")
    return resolved


def _effective_hanoi_max_turns(
    *,
    max_turns: int,
    optimal_steps: int,
    optimal_turn_cap_multiplier: float | None,
) -> int:
    if optimal_turn_cap_multiplier is None:
        return max_turns
    optimal_cap = int(math.ceil(float(optimal_steps) * optimal_turn_cap_multiplier))
    optimal_cap = max(1, optimal_cap)
    return min(max_turns, optimal_cap)


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


def add_hanoi_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=None,
        help=(
            "Exact Hanoi case as NPEGSxNDISKS (or NPEGS,NDISKS). "
            "Repeatable. Avoids n_pegs x n_disks cartesian expansion."
        ),
    )
    parser.add_argument("--n-pegs", action="append", default=None)
    parser.add_argument("--n-disks", action="append", default=None)
    parser.add_argument("--start-peg", type=int, default=None)
    parser.add_argument("--goal-peg", type=int, default=None)
    parser.add_argument("--runs-per-variant", type=int, default=None)
    parser.add_argument(
        "--optimal-turn-cap-multiplier",
        type=float,
        default=None,
        help=(
            "Cap each episode turn budget to min(max_turns, ceil(optimal_steps * M)). "
            "Disabled when unset."
        ),
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


def estimate_episodes(
    args: argparse.Namespace,
    config: dict[str, Any] | None,
    *,
    game_name: str = "hanoi",  # noqa: ARG001
) -> int:
    config = config or {}
    provider_name = getattr(args, "provider", None)
    if not provider_name:
        raise SystemExit("Missing required argument: --provider")
    model_arg = getattr(args, "model", None)
    models = _resolve_models(provider_name, config, model_arg)
    if not models:
        raise SystemExit("No models provided. Use --model or config.json.")

    start_peg_arg = getattr(args, "start_peg", None)
    start_peg_config = config.get("start_peg", 0)
    start_peg_value = (
        int(start_peg_arg) if start_peg_arg is not None else start_peg_config
    )
    goal_peg_arg = getattr(args, "goal_peg", None)
    goal_peg_config = config.get("goal_peg")
    goal_peg_value = int(goal_peg_arg) if goal_peg_arg is not None else goal_peg_config

    cases_arg = getattr(args, "cases", None)
    n_pegs_arg = getattr(args, "n_pegs", None)
    n_disks_arg = getattr(args, "n_disks", None)
    if cases_arg is not None and (n_pegs_arg is not None or n_disks_arg is not None):
        raise SystemExit("Do not mix --case with --n-pegs/--n-disks.")

    raw_cases = cases_arg if cases_arg is not None else config.get("cases")
    scenarios: list[HanoiCase] = []
    if raw_cases is not None:
        parsed_cases = _parse_cases_config(raw_cases)
        for n_pegs, n_disks, case_start, case_goal in parsed_cases:
            if n_disks < 1:
                raise SystemExit(f"n_disks must be >= 1, got {n_disks}")
            effective_start = (
                int(start_peg_arg)
                if start_peg_arg is not None
                else (case_start if case_start is not None else start_peg_config)
            )
            effective_goal = (
                int(goal_peg_arg)
                if goal_peg_arg is not None
                else (case_goal if case_goal is not None else goal_peg_config)
            )
            resolved_start, resolved_goal = _resolve_start_goal_pegs(
                n_pegs=n_pegs,
                start_peg_value=effective_start,
                goal_peg_value=effective_goal,
            )
            scenarios.append(
                HanoiCase(
                    n_pegs=n_pegs,
                    n_disks=n_disks,
                    start_peg=resolved_start,
                    goal_peg=resolved_goal,
                )
            )
    else:
        n_pegs_list = (
            _parse_int_list(n_pegs_arg)
            if n_pegs_arg is not None
            else _parse_int_list([str(x) for x in config.get("n_pegs", [3])])
        )
        if not n_pegs_list:
            raise SystemExit("No peg counts provided. Use --n-pegs or config n_pegs.")
        if any(n_pegs < 3 for n_pegs in n_pegs_list):
            raise SystemExit("All n_pegs values must be >= 3.")

        n_disks_list = (
            _parse_int_list(n_disks_arg)
            if n_disks_arg is not None
            else _parse_int_list([str(x) for x in config.get("n_disks", [3])])
        )
        if not n_disks_list:
            raise SystemExit(
                "No disk counts provided. Use --n-disks or config n_disks."
            )
        if any(n_disks < 1 for n_disks in n_disks_list):
            raise SystemExit("All n_disks values must be >= 1.")

        for n_pegs in n_pegs_list:
            resolved_start, resolved_goal = _resolve_start_goal_pegs(
                n_pegs=n_pegs,
                start_peg_value=(
                    None if start_peg_value is None else int(start_peg_value)
                ),
                goal_peg_value=(
                    None if goal_peg_value is None else int(goal_peg_value)
                ),
            )
            for n_disks in n_disks_list:
                scenarios.append(
                    HanoiCase(
                        n_pegs=n_pegs,
                        n_disks=n_disks,
                        start_peg=resolved_start,
                        goal_peg=resolved_goal,
                    )
                )

    if not scenarios:
        raise SystemExit("No Hanoi cases selected to run.")

    deduped_scenarios: list[HanoiCase] = []
    seen_scenarios: set[tuple[int, int, int, int]] = set()
    for scenario in scenarios:
        key = (
            scenario.n_pegs,
            scenario.n_disks,
            scenario.start_peg,
            scenario.goal_peg,
        )
        if key in seen_scenarios:
            continue
        seen_scenarios.add(key)
        deduped_scenarios.append(scenario)
    scenarios = deduped_scenarios

    runs_per_variant = _resolve_positive_int(
        getattr(args, "runs_per_variant", None), config, "runs_per_variant", 3
    )
    prompt_file_arg = getattr(args, "prompt_file", None)
    prompt_variants = (
        _load_prompt_variants(prompt_file_arg)
        if prompt_file_arg
        else DEFAULT_PROMPT_VARIANTS
    )
    selected_prompt_names_raw = getattr(args, "prompt_variants", None) or config.get(
        "prompt_variants", ["minimal"]
    )
    selected_prompt_names = _parse_str_list([selected_prompt_names_raw])
    if not selected_prompt_names:
        raise SystemExit("No Hanoi prompt variants selected.")
    unknown_prompt_variants = [
        name for name in selected_prompt_names if name not in prompt_variants
    ]
    if unknown_prompt_variants:
        raise SystemExit(
            "Unknown Hanoi prompt variant(s): "
            + ", ".join(sorted(set(unknown_prompt_variants)))
        )
    selected_prompt_variants = [prompt_variants[name] for name in selected_prompt_names]

    tool_variants = DEFAULT_TOOL_VARIANTS
    selected_tool_names_raw = getattr(args, "tool_variants", None) or config.get(
        "tool_variants", ["move_only"]
    )
    selected_tool_names = _parse_str_list([selected_tool_names_raw])
    if not selected_tool_names:
        raise SystemExit("No Hanoi tool variants selected.")
    unknown_tool_variants = [
        name for name in selected_tool_names if name not in tool_variants
    ]
    if unknown_tool_variants:
        raise SystemExit(
            "Unknown Hanoi tool variant(s): "
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
            ToolVariant(name="custom", allowed_tools=allowed_tools)
        ]

    episodes_per_model = (
        len(scenarios)
        * len(selected_prompt_variants)
        * len(selected_tool_variants)
        * runs_per_variant
    )
    return len(models) * episodes_per_model


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
        tool_schemas=tool_schemas_payload,
        state_format=state_format,
        image_config=image_config,
    )
    return [line for line in buffer.getvalue().splitlines() if line]


def _run_hanoi_episode_job(
    job: HanoiEpisodeJob,
    *,
    provider: Any,
    provider_name: str,
    model_name: str,
    spec_name: str,
    interaction_mode: str,
    stateless: bool,
    max_turns: int,
    optimal_turn_cap_multiplier: float | None,
    state_format: str,
    image_size: tuple[int, int],
    image_labels: bool,
    image_background: str,
    record_provider_raw: bool,
    record_raw: bool,
    record: bool,
    stagnation_patience: int | None,
) -> HanoiEpisodeOutput:
    n_pegs = job.case.n_pegs
    n_disks = job.case.n_disks
    start_peg = job.case.start_peg
    goal_peg = job.case.goal_peg

    env = TowerOfHanoiEnv(
        n_disks=n_disks,
        n_pegs=n_pegs,
        start_peg=start_peg,
        goal_peg=goal_peg,
        record_history=True,
        illegal_action_behavior="penalize",
    )
    adapter = HanoiGameAdapter(env)
    instruction_template = job.prompt_variant.instructions
    if state_format in {"image", "both"}:
        instruction_template = with_image_instructions(instruction_template)
    instructions = format_instructions(
        instruction_template,
        n_pegs=env.n_pegs,
        start_peg=env.start_peg,
        goal_peg=env.goal_peg,
    )
    state_formatter = lambda a, pv=job.prompt_variant: a.env.format_prompt_state(
        include_legal_moves=pv.include_legal_moves,
        include_action_space=pv.include_action_space,
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

    effective_max_turns = _effective_hanoi_max_turns(
        max_turns=max_turns,
        optimal_steps=env.optimal_steps(),
        optimal_turn_cap_multiplier=optimal_turn_cap_multiplier,
    )
    result = run_tool_calling_episode(
        adapter,
        provider,
        max_turns=effective_max_turns,
        instructions=instructions,
        state_formatter=state_formatter,
        state_image_renderer=state_image_renderer,
        allowed_tools=job.tool_variant.allowed_tools,
        record_provider_raw=record_provider_raw,
        stagnation_patience=stagnation_patience,
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

    episode = annotate_episode_with_taxonomy(
        {
            "episode_id": job.episode_id,
            "game": "hanoi",
            "variant_id": job.variant_id,
            "run_idx": job.run_idx,
            "provider": provider_name,
            "model": model_name,
            "spec": spec_name,
            "interaction_mode": interaction_mode,
            "stateless": stateless,
            "n_pegs": result.game_metrics.get("n_pegs", n_pegs),
            "n_disks": result.game_metrics.get("n_disks", n_disks),
            "start_peg": env.start_peg,
            "goal_peg": env.goal_peg,
            "prompt_variant": job.prompt_variant.name,
            "tools_variant": job.tool_variant.name,
            "solved": result.solved,
            "turn_count": turn_count,
            "move_count": result.game_metrics.get("move_count", result.move_count),
            "optimal_steps": result.game_metrics.get(
                "optimal_steps", result.optimal_steps
            ),
            "illegal_moves": result.illegal_moves,
            "tool_calls": result.tool_calls,
            "max_turns_effective": effective_max_turns,
            "terminated_early": terminated_early,
            "termination_reason": termination_reason,
            "provider_error_count": provider_error_count,
            "usage": result.usage,
            "cost": result.cost,
        },
        game_name="hanoi",
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
                "n_pegs": n_pegs,
                "n_disks": n_disks,
                "start_peg": start_peg,
                "goal_peg": goal_peg,
                "prompt_variant": job.prompt_variant.name,
                "tools_variant": job.tool_variant.name,
                "solved": result.solved,
                "max_turns_effective": effective_max_turns,
                "terminated_early": terminated_early,
                "termination_reason": termination_reason,
            },
        )

    raw_lines: list[str] = []
    if record_raw:
        full_tools = tool_schemas(n_pegs=n_pegs)
        selected_tools = (
            [t for t in full_tools if t["name"] in job.tool_variant.allowed_tools]
            if job.tool_variant.allowed_tools is not None
            else full_tools
        )
        raw_lines = _raw_lines_for_events(
            events=result.events,
            episode_id=job.episode_id,
            variant_id=job.variant_id,
            instructions=instructions,
            tool_schemas_payload=selected_tools,
            state_format=state_format,
            image_config={
                "size": image_size,
                "labels": image_labels,
                "background": image_background,
            },
        )

    return HanoiEpisodeOutput(
        episode_id=job.episode_id,
        variant_id=job.variant_id,
        episode=episode,
        events=result.events,
        raw_lines=raw_lines,
        recording=recording,
    )


def _resolve_out_dir_base(base: str | Path, game_name: str) -> Path:
    return _shared_resolve_out_dir_base(base, game_name)


def _resolve_checkpoint_interval(
    args: argparse.Namespace, config: dict[str, Any]
) -> int:
    return _shared_resolve_checkpoint_interval(args, config)


def run_batch(
    args: argparse.Namespace,
    config: dict[str, Any] | None,
    *,
    game_name: str = "hanoi",
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

    start_peg_arg = getattr(args, "start_peg", None)
    start_peg_config = config.get("start_peg", 0)
    start_peg_value = (
        int(start_peg_arg) if start_peg_arg is not None else start_peg_config
    )
    goal_peg_arg = getattr(args, "goal_peg", None)
    goal_peg_config = config.get("goal_peg")
    goal_peg_value = int(goal_peg_arg) if goal_peg_arg is not None else goal_peg_config

    cases_arg = getattr(args, "cases", None)
    n_pegs_arg = getattr(args, "n_pegs", None)
    n_disks_arg = getattr(args, "n_disks", None)

    if cases_arg is not None and (n_pegs_arg is not None or n_disks_arg is not None):
        raise SystemExit("Do not mix --case with --n-pegs/--n-disks.")

    raw_cases = cases_arg if cases_arg is not None else config.get("cases")
    scenarios: list[HanoiCase] = []
    if raw_cases is not None:
        parsed_cases = _parse_cases_config(raw_cases)
        for n_pegs, n_disks, case_start, case_goal in parsed_cases:
            if n_disks < 1:
                raise SystemExit(f"n_disks must be >= 1, got {n_disks}")
            effective_start = (
                int(start_peg_arg)
                if start_peg_arg is not None
                else (case_start if case_start is not None else start_peg_config)
            )
            effective_goal = (
                int(goal_peg_arg)
                if goal_peg_arg is not None
                else (case_goal if case_goal is not None else goal_peg_config)
            )
            resolved_start, resolved_goal = _resolve_start_goal_pegs(
                n_pegs=n_pegs,
                start_peg_value=effective_start,
                goal_peg_value=effective_goal,
            )
            scenarios.append(
                HanoiCase(
                    n_pegs=n_pegs,
                    n_disks=n_disks,
                    start_peg=resolved_start,
                    goal_peg=resolved_goal,
                )
            )
    else:
        n_pegs_list = (
            _parse_int_list(n_pegs_arg)
            if n_pegs_arg is not None
            else _parse_int_list([str(x) for x in config.get("n_pegs", [3])])
        )
        if not n_pegs_list:
            raise SystemExit("No peg counts provided. Use --n-pegs or config n_pegs.")
        if any(n_pegs < 3 for n_pegs in n_pegs_list):
            raise SystemExit("All n_pegs values must be >= 3.")

        n_disks_list = (
            _parse_int_list(n_disks_arg)
            if n_disks_arg is not None
            else _parse_int_list([str(x) for x in config.get("n_disks", [3])])
        )
        if not n_disks_list:
            raise SystemExit(
                "No disk counts provided. Use --n-disks or config n_disks."
            )
        if any(n_disks < 1 for n_disks in n_disks_list):
            raise SystemExit("All n_disks values must be >= 1.")

        for n_pegs in n_pegs_list:
            resolved_start, resolved_goal = _resolve_start_goal_pegs(
                n_pegs=n_pegs,
                start_peg_value=(
                    None if start_peg_value is None else int(start_peg_value)
                ),
                goal_peg_value=(
                    None if goal_peg_value is None else int(goal_peg_value)
                ),
            )
            for n_disks in n_disks_list:
                scenarios.append(
                    HanoiCase(
                        n_pegs=n_pegs,
                        n_disks=n_disks,
                        start_peg=resolved_start,
                        goal_peg=resolved_goal,
                    )
                )

    if not scenarios:
        raise SystemExit("No Hanoi cases selected to run.")

    deduped_scenarios: list[HanoiCase] = []
    seen_scenarios: set[tuple[int, int, int, int]] = set()
    for scenario in scenarios:
        key = (
            scenario.n_pegs,
            scenario.n_disks,
            scenario.start_peg,
            scenario.goal_peg,
        )
        if key in seen_scenarios:
            continue
        seen_scenarios.add(key)
        deduped_scenarios.append(scenario)
    scenarios = deduped_scenarios

    n_pegs_list = list(dict.fromkeys(case.n_pegs for case in scenarios))
    n_disks_list = list(dict.fromkeys(case.n_disks for case in scenarios))

    start_goal_by_n_pegs_int: dict[int, tuple[int, int]] = {}
    for case in scenarios:
        pair = (case.start_peg, case.goal_peg)
        existing_pair = start_goal_by_n_pegs_int.get(case.n_pegs)
        if existing_pair is None:
            start_goal_by_n_pegs_int[case.n_pegs] = pair
            continue
        if existing_pair != pair:
            raise SystemExit(
                "Cases with the same n_pegs must share start/goal pegs in a single run."
            )
    resolved_start_goal_by_n_pegs = {
        str(n_pegs): pair for n_pegs, pair in start_goal_by_n_pegs_int.items()
    }

    runs_per_variant = _resolve_positive_int(
        getattr(args, "runs_per_variant", None), config, "runs_per_variant", 3
    )
    max_turns = _resolve_positive_int(
        getattr(args, "max_turns", None), config, "max_turns", 200
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
    optimal_turn_cap_multiplier = _resolve_optional_positive_float(
        getattr(args, "optimal_turn_cap_multiplier", None),
        config,
        "optimal_turn_cap_multiplier",
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
    prompt_file_arg = getattr(args, "prompt_file", None)
    prompt_variants = (
        _load_prompt_variants(prompt_file_arg)
        if prompt_file_arg
        else DEFAULT_PROMPT_VARIANTS
    )

    selected_prompt_names_raw = getattr(args, "prompt_variants", None) or config.get(
        "prompt_variants", ["minimal"]
    )
    selected_prompt_names = _parse_str_list([selected_prompt_names_raw])
    if not selected_prompt_names:
        raise SystemExit("No Hanoi prompt variants selected.")
    unknown_prompt_variants = [
        name for name in selected_prompt_names if name not in prompt_variants
    ]
    if unknown_prompt_variants:
        raise SystemExit(
            "Unknown Hanoi prompt variant(s): "
            + ", ".join(sorted(set(unknown_prompt_variants)))
        )
    selected_prompt_variants = [prompt_variants[name] for name in selected_prompt_names]
    tool_variants = DEFAULT_TOOL_VARIANTS
    selected_tool_names_raw = getattr(args, "tool_variants", None) or config.get(
        "tool_variants", ["move_only"]
    )
    selected_tool_names = _parse_str_list([selected_tool_names_raw])
    if not selected_tool_names:
        raise SystemExit("No Hanoi tool variants selected.")
    unknown_tool_variants = [
        name for name in selected_tool_names if name not in tool_variants
    ]
    if unknown_tool_variants:
        raise SystemExit(
            "Unknown Hanoi tool variant(s): "
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

        default_schema_n_pegs = scenarios[0].n_pegs
        full_tool_schemas = tool_schemas(n_pegs=default_schema_n_pegs)
        tool_schemas_by_variant = {
            variant.name: (
                [t for t in full_tool_schemas if t["name"] in variant.allowed_tools]
                if variant.allowed_tools is not None
                else full_tool_schemas
            )
            for variant in selected_tool_variants
        }
        tool_schemas_by_n_pegs = {
            str(n_pegs): tool_schemas(n_pegs=n_pegs)
            for n_pegs in sorted(set(case.n_pegs for case in scenarios))
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
                "n_pegs": n_pegs_list,
                "n_disks": n_disks_list,
                "start_peg": start_peg_value,
                "goal_peg": goal_peg_value,
                "cases": [
                    {
                        "n_pegs": case.n_pegs,
                        "n_disks": case.n_disks,
                        "start_peg": case.start_peg,
                        "goal_peg": case.goal_peg,
                    }
                    for case in scenarios
                ],
                "start_goal_by_n_pegs": {
                    key: {"start_peg": value[0], "goal_peg": value[1]}
                    for key, value in resolved_start_goal_by_n_pegs.items()
                },
                "runs_per_variant": runs_per_variant,
                "max_turns": max_turns,
                "parallelism": parallelism,
                "max_inflight_provider": max_inflight_provider,
                "stagnation_patience": stagnation_patience,
                "optimal_turn_cap_multiplier": optimal_turn_cap_multiplier,
                "prompt_variants": [asdict(v) for v in selected_prompt_variants],
                "tool_variants": [asdict(v) for v in selected_tool_variants],
                "tool_schemas": full_tool_schemas,
                "tool_schemas_by_variant": tool_schemas_by_variant,
                "tool_schemas_by_n_pegs": tool_schemas_by_n_pegs,
                "state_format": state_format,
                "image_size": f"{image_size[0]}x{image_size[1]}",
                "image_labels": image_labels,
                "image_background": image_background,
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

        jobs: list[HanoiEpisodeJob] = []
        episode_id = 0
        for case in scenarios:
            for prompt_variant in selected_prompt_variants:
                for tool_variant in selected_tool_variants:
                    variant_id = (
                        f"p{case.n_pegs}_n{case.n_disks}__prompt={prompt_variant.name}"
                        f"__tools={tool_variant.name}"
                    )
                    for run_idx in range(runs_per_variant):
                        jobs.append(
                            HanoiEpisodeJob(
                                episode_id=episode_id,
                                variant_id=variant_id,
                                run_idx=run_idx,
                                case=case,
                                prompt_variant=prompt_variant,
                                tool_variant=tool_variant,
                            )
                        )
                        episode_id += 1

        def run_job(job: HanoiEpisodeJob) -> HanoiEpisodeOutput:
            return _run_hanoi_episode_job(
                job,
                provider=get_provider(),
                provider_name=provider_name,
                model_name=model_name,
                spec_name=spec_name,
                interaction_mode=interaction_mode,
                stateless=stateless,
                max_turns=max_turns,
                optimal_turn_cap_multiplier=optimal_turn_cap_multiplier,
                state_format=state_format,
                image_size=image_size,
                image_labels=image_labels,
                image_background=image_background,
                record_provider_raw=record_provider_raw,
                record_raw=record_raw,
                record=record,
                stagnation_patience=stagnation_patience,
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
        game_name="hanoi",
        defaults=default_hanoi_config(),
    )
    run_dirs = run_batch(args, config, game_name="hanoi")
    print(json.dumps({"run_dirs": [str(p) for p in run_dirs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
