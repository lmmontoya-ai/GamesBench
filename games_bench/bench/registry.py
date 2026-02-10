from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class BenchSpec:
    name: str
    description: str
    batch_runner: Callable[[argparse.Namespace, dict[str, Any]], list[Path]]
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None
    default_config: Callable[[], dict[str, Any]] | None = None
    estimate_episodes: Callable[[argparse.Namespace, dict[str, Any]], int] | None = None
    episode_scorer: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    episode_taxonomy: (
        Callable[
            [dict[str, Any], dict[str, Any]],
            dict[str, Any] | tuple[str, list[str]] | list[str],
        ]
        | None
    ) = None
    compare_metrics: Callable[[dict[str, Any]], dict[str, float]] | None = None
    score_episodes: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    adapter_factory: Callable[..., Any] | None = None
    render_main: Callable[[], int] | None = None
    review_main: Callable[[], int] | None = None


_REGISTRY: dict[str, BenchSpec] = {}


def register_benchmark(spec: BenchSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_benchmark(name: str) -> BenchSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown benchmark: {name}")
    return _REGISTRY[name]


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_builtin_benchmarks() -> None:
    if _REGISTRY:
        return
    from games_bench.bench import hanoi as hanoi_bench
    from games_bench.bench import sokoban as sokoban_bench
    from games_bench.games.hanoi import render as hanoi_render
    from games_bench.games.hanoi import review as hanoi_review
    from games_bench.games.sokoban import render as sokoban_render
    from games_bench.games.sokoban import review as sokoban_review

    register_benchmark(
        BenchSpec(
            name="hanoi",
            description="Tower of Hanoi",
            batch_runner=hanoi_bench.run_batch,
            add_arguments=hanoi_bench.add_hanoi_arguments,
            default_config=hanoi_bench.default_hanoi_config,
            estimate_episodes=hanoi_bench.estimate_episodes,
            episode_scorer=hanoi_bench.score_episodes,
            compare_metrics=hanoi_bench.compare_metrics,
            score_episodes=hanoi_bench.score_episodes,
            adapter_factory=hanoi_bench.build_hanoi_adapter,
            render_main=hanoi_render.main,
            review_main=hanoi_review.main,
        )
    )
    register_benchmark(
        BenchSpec(
            name="sokoban",
            description="Sokoban",
            batch_runner=sokoban_bench.run_batch,
            add_arguments=sokoban_bench.add_sokoban_arguments,
            default_config=sokoban_bench.default_sokoban_config,
            estimate_episodes=sokoban_bench.estimate_episodes,
            episode_scorer=sokoban_bench.score_episodes,
            compare_metrics=sokoban_bench.compare_metrics,
            score_episodes=sokoban_bench.score_episodes,
            adapter_factory=sokoban_bench.build_sokoban_adapter,
            render_main=sokoban_render.main,
            review_main=sokoban_review.main,
        )
    )
