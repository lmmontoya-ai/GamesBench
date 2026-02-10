from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from games_bench.bench.common import (
    add_common_batch_arguments,
    resolve_progress_settings,
)
from games_bench.bench.progress import build_episode_progress_reporter
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


def _normalize_games_config_or_error(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    default_game: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        return normalize_games_config(config, default_game=default_game)
    except ValueError as exc:
        parser.error(str(exc))
        raise


def _estimate_episode_total(
    *,
    args: argparse.Namespace,
    game_configs: list[tuple[str, Any, dict[str, Any]]],
    progress_enabled: bool,
    progress_explicit: bool,
) -> tuple[int, bool]:
    if not progress_enabled:
        return (0, False)

    total_episodes = 0
    can_estimate_all = True
    for game_name, benchmark, game_config in game_configs:
        estimator = benchmark.estimate_episodes
        if estimator is None:
            can_estimate_all = False
            continue
        try:
            raw_estimate = estimator(args, game_config)
        except NotImplementedError:
            can_estimate_all = False
            continue
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(
                f"Failed to estimate episodes for benchmark '{game_name}': {exc}"
            ) from exc
        if raw_estimate is None:
            can_estimate_all = False
            continue
        try:
            estimate = int(raw_estimate)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "Benchmark episode estimator must return an integer or None. "
                f"Got {raw_estimate!r} for benchmark '{game_name}'."
            ) from exc
        total_episodes += max(0, estimate)

    if can_estimate_all and total_episodes > 0:
        return (total_episodes, True)

    if progress_explicit and progress_enabled:
        print(
            "Progress requested but episode totals could not be estimated for all "
            "selected benchmarks. Progress disabled.",
            file=sys.stderr,
            flush=True,
        )
    return (0, False)


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
    global_defaults, games_map = _normalize_games_config_or_error(
        parser,
        config,
        default_game=game_name,
    )
    bench_defaults = benchmark.default_config() if benchmark.default_config else {}
    game_config = merge_dicts(
        bench_defaults,
        merge_dicts(global_defaults, games_map.get(game_name, {})),
    )
    progress_enabled, progress_refresh_s, progress_explicit = resolve_progress_settings(
        args, config
    )
    game_configs = [(game_name, benchmark, game_config)]
    total_episodes, progress_ready = _estimate_episode_total(
        args=args,
        game_configs=game_configs,
        progress_enabled=progress_enabled,
        progress_explicit=progress_explicit,
    )
    reporter = build_episode_progress_reporter(
        enabled=progress_ready,
        total_episodes=total_episodes,
        refresh_s=progress_refresh_s,
        explicit_request=progress_explicit,
    )
    setattr(args, "_progress_reporter", reporter)
    try:
        game_run_dirs = benchmark.batch_runner(args, game_config)
    finally:
        if hasattr(args, "_progress_reporter"):
            delattr(args, "_progress_reporter")
        reporter.close()
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
    global_defaults, games_map = _normalize_games_config_or_error(
        parser,
        config,
        default_game="hanoi",
    )
    selected_games = _select_games(games_map, args.games)
    if not selected_games:
        selected_games = ["hanoi"]
    known_games = set(list_benchmarks())
    unknown_games = [name for name in selected_games if name not in known_games]
    if unknown_games:
        parser.error(f"Unknown game(s): {', '.join(unknown_games)}")

    prepared_runs: list[tuple[str, Any, dict[str, Any]]] = []
    for game_name in selected_games:
        benchmark = get_benchmark(game_name)
        bench_defaults = benchmark.default_config() if benchmark.default_config else {}
        game_config = merge_dicts(
            bench_defaults,
            merge_dicts(global_defaults, games_map.get(game_name, {})),
        )
        prepared_runs.append((game_name, benchmark, game_config))

    progress_enabled, progress_refresh_s, progress_explicit = resolve_progress_settings(
        args, config
    )
    total_episodes, progress_ready = _estimate_episode_total(
        args=args,
        game_configs=prepared_runs,
        progress_enabled=progress_enabled,
        progress_explicit=progress_explicit,
    )
    reporter = build_episode_progress_reporter(
        enabled=progress_ready,
        total_episodes=total_episodes,
        refresh_s=progress_refresh_s,
        explicit_request=progress_explicit,
    )
    setattr(args, "_progress_reporter", reporter)

    run_dirs: list[str] = []
    try:
        for _game_name, benchmark, game_config in prepared_runs:
            game_run_dirs = benchmark.batch_runner(args, game_config)
            run_dirs.extend([str(p) for p in game_run_dirs])
    finally:
        if hasattr(args, "_progress_reporter"):
            delattr(args, "_progress_reporter")
        reporter.close()
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
