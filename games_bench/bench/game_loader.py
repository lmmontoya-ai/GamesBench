from __future__ import annotations

import json
from typing import Any

from games_bench.bench.registry import BenchSpec, get_benchmark, load_builtin_benchmarks
from games_bench.games.registry import GameSpec, get_game, load_builtin_games


def parse_env_kwargs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--env-kwargs must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--env-kwargs must decode to a JSON object.")
    return data


def get_specs(game_name: str) -> tuple[GameSpec, BenchSpec]:
    load_builtin_games()
    load_builtin_benchmarks()
    try:
        game_spec = get_game(game_name)
    except KeyError as exc:
        raise SystemExit(f"Unknown game: {game_name}") from exc
    try:
        bench_spec = get_benchmark(game_name)
    except KeyError as exc:
        raise SystemExit(f"No benchmark registered for game: {game_name}") from exc
    return game_spec, bench_spec


def build_env_and_adapter(
    game_name: str,
    *,
    env_kwargs: dict[str, Any] | None = None,
    adapter_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    game_spec, bench_spec = get_specs(game_name)
    env = game_spec.env_factory(**(env_kwargs or {}))
    if bench_spec.adapter_factory is None:
        raise SystemExit(f"Benchmark '{game_name}' does not define adapter_factory.")
    adapter = bench_spec.adapter_factory(env, **(adapter_kwargs or {}))
    return env, adapter
