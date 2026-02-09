from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from games_bench.bench.common import add_common_batch_arguments
from games_bench.bench.registry import (
    get_benchmark,
    list_benchmarks,
    load_builtin_benchmarks,
)
from games_bench.bench.suites import get_suite, iter_suites, load_builtin_suites
from games_bench.config import load_config, merge_dicts, normalize_games_config


def _select_games(
    games_map: dict[str, dict[str, Any]],
    requested: list[str] | None,
) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    return list(games_map.keys())


def _require_provider(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if not args.provider:
        parser.error("--provider is required")


def _print_suites() -> None:
    suites = [
        {"name": spec.name, "description": spec.description} for spec in iter_suites()
    ]
    print(json.dumps({"suites": suites}, indent=2))


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    suite_name = getattr(args, "suite", None)
    suite_config: dict[str, Any] = {}
    if suite_name:
        try:
            suite = get_suite(suite_name)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
        suite_config = suite.config_factory()
    file_config = load_config(args.config) if args.config else {}
    return merge_dicts(suite_config, file_config)


def _run_single_game(
    game_name: str,
    argv: list[str],
) -> list[str]:
    benchmark = get_benchmark(game_name)
    parser = argparse.ArgumentParser(
        description=f"Batch benchmark for {benchmark.description}."
    )
    add_common_batch_arguments(parser)
    if benchmark.add_arguments:
        benchmark.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.list_suites:
        _print_suites()
        return []
    _require_provider(args, parser)

    config = _resolve_config(args)
    global_defaults, games_map = normalize_games_config(config, default_game=game_name)
    bench_defaults = benchmark.default_config() if benchmark.default_config else {}
    game_config = merge_dicts(
        bench_defaults,
        merge_dicts(global_defaults, games_map.get(game_name, {})),
    )
    game_run_dirs = benchmark.batch_runner(args, game_config)
    return [str(p) for p in game_run_dirs]


def _run_config_mode(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(
        description=(
            "Batch benchmark runner (config-driven). "
            "Use `games-bench run <game>` for game-specific flags."
        )
    )
    add_common_batch_arguments(parser, include_game_filter=True)
    args = parser.parse_args(argv)
    if args.list_suites:
        _print_suites()
        return []
    _require_provider(args, parser)

    config = _resolve_config(args)
    global_defaults, games_map = normalize_games_config(config, default_game="hanoi")
    selected_games = _select_games(games_map, args.games)
    if not selected_games:
        selected_games = ["hanoi"]
    known_games = set(list_benchmarks())
    unknown_games = [name for name in selected_games if name not in known_games]
    if unknown_games:
        parser.error(f"Unknown game(s): {', '.join(unknown_games)}")

    run_dirs: list[str] = []
    for game_name in selected_games:
        benchmark = get_benchmark(game_name)
        bench_defaults = benchmark.default_config() if benchmark.default_config else {}
        game_config = merge_dicts(
            bench_defaults,
            merge_dicts(global_defaults, games_map.get(game_name, {})),
        )
        game_run_dirs = benchmark.batch_runner(args, game_config)
        run_dirs.extend([str(p) for p in game_run_dirs])
    return run_dirs


def main(argv: list[str] | None = None) -> int:
    load_builtin_benchmarks()
    load_builtin_suites()
    args = list(sys.argv[1:] if argv is None else argv)
    if "--list-suites" in args:
        _print_suites()
        return 0
    benchmark_names = set(list_benchmarks())
    if args and args[0] in benchmark_names:
        game_name = args.pop(0)
        run_dirs = _run_single_game(game_name, args)
    else:
        run_dirs = _run_config_mode(args)

    print(json.dumps({"run_dirs": run_dirs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
