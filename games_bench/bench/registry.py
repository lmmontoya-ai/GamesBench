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

    register_benchmark(
        BenchSpec(
            name="hanoi",
            description="Tower of Hanoi",
            batch_runner=hanoi_bench.run_batch,
        )
    )
