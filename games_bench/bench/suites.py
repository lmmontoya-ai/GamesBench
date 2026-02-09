from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """Named benchmark suite with a config payload factory."""

    name: str
    description: str
    config_factory: Callable[[], dict[str, Any]]


_REGISTRY: dict[str, SuiteSpec] = {}


def register_suite(spec: SuiteSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_suite(name: str) -> SuiteSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown suite: {name}")
    return _REGISTRY[name]


def list_suites() -> list[str]:
    return sorted(_REGISTRY.keys())


def iter_suites() -> list[SuiteSpec]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY.keys())]


def _standard_v1_config() -> dict[str, Any]:
    """Canonical cross-game planning benchmark for reproducible model evals."""

    return {
        "out_dir": "artifacts/runs",
        "record": True,
        "record_raw": False,
        "record_provider_raw": False,
        "provider_retries": 2,
        "provider_backoff": 1.0,
        "stream_debug": False,
        "games": {
            "hanoi": {
                # Exact difficulty ladder (no cartesian product):
                # (3,3), (3,4), (3,5), (3,10), (3,20), (4,4), (4,5), (4,10), (4,20)
                "cases": [
                    {"n_pegs": 3, "n_disks": 3},
                    {"n_pegs": 3, "n_disks": 4},
                    {"n_pegs": 3, "n_disks": 5},
                    {"n_pegs": 4, "n_disks": 4},
                    {"n_pegs": 3, "n_disks": 10},
                    {"n_pegs": 3, "n_disks": 20},
                    {"n_pegs": 4, "n_disks": 5},
                    {"n_pegs": 4, "n_disks": 10},
                    {"n_pegs": 4, "n_disks": 20},
                ],
                "runs_per_variant": 5,
                "max_turns": 400,
                "prompt_variants": ["full"],
                "tool_variants": ["move_only"],
                "state_format": "text",
            },
            "sokoban": {
                # Deterministic procedural long-horizon ladder.
                "procgen_cases": [
                    {
                        "grid_size": "8x8",
                        "box_count": 6,
                        "scramble_steps": [140, 180],
                        "levels_per_combo": 3,
                    },
                    {
                        "grid_size": "10x10",
                        "box_count": 6,
                        "scramble_steps": [220, 260],
                        "levels_per_combo": 3,
                    },
                    {
                        "grid_size": "10x10",
                        "box_count": 7,
                        "scramble_steps": [220, 260],
                        "levels_per_combo": 3,
                    },
                    {
                        "grid_size": "12x12",
                        "box_count": 8,
                        "scramble_steps": "300+",
                        "levels_per_combo": 3,
                    },
                ],
                "procgen_seed": 2026,
                "procgen_wall_density": 0.08,
                "max_levels": None,
                "runs_per_level": 3,
                "max_turns": 400,
                "prompt_variants": ["full"],
                "tool_variants": ["move_and_query"],
                "state_format": "text",
                "detect_deadlocks": True,
                "terminal_on_deadlock": True,
            },
        },
    }


def load_builtin_suites() -> None:
    if _REGISTRY:
        return
    register_suite(
        SuiteSpec(
            name="standard-v1",
            description=(
                "Canonical planning suite across Hanoi and Sokoban with "
                "fixed difficulty ladders and repeated runs."
            ),
            config_factory=_standard_v1_config,
        )
    )
