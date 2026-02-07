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
    from games_bench.games.hanoi import render as hanoi_render
    from games_bench.games.hanoi import review as hanoi_review

    register_benchmark(
        BenchSpec(
            name="hanoi",
            description="Tower of Hanoi",
            batch_runner=hanoi_bench.run_batch,
            add_arguments=hanoi_bench.add_hanoi_arguments,
            default_config=hanoi_bench.default_hanoi_config,
            adapter_factory=hanoi_bench.build_hanoi_adapter,
            render_main=hanoi_render.main,
            review_main=hanoi_review.main,
        )
    )
