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


def _easy_v1_config() -> dict[str, Any]:
    """Accessible cross-game suite for smaller-capability models."""

    return {
        "spec": "easy-v1",
        "out_dir": "artifacts/runs",
        "record": True,
        "record_raw": False,
        "record_provider_raw": False,
        "provider_retries": 2,
        "provider_backoff": 1.0,
        "stream_debug": False,
        "parallelism": 2,
        "max_inflight_provider": 2,
        "games": {
            "hanoi": {
                # Easier fixed ladder suitable for smaller models.
                "cases": [
                    {"n_pegs": 3, "n_disks": 2},
                    {"n_pegs": 3, "n_disks": 3},
                    {"n_pegs": 3, "n_disks": 4},
                    {"n_pegs": 3, "n_disks": 5},
                    {"n_pegs": 4, "n_disks": 4},
                    {"n_pegs": 4, "n_disks": 5},
                ],
                "runs_per_variant": 3,
                "max_turns": 200,
                "stagnation_patience": 20,
                "optimal_turn_cap_multiplier": 4.0,
                "prompt_variants": ["full"],
                "tool_variants": ["move_and_state"],
                "state_format": "text",
            },
            "sokoban": {
                # Deterministic procedural ladder with lower spatial complexity.
                "procgen_cases": [
                    {
                        "grid_size": "8x8",
                        "box_count": 2,
                        "scramble_steps": [24, 32],
                        "levels_per_combo": 4,
                    },
                    {
                        "grid_size": "8x8",
                        "box_count": 3,
                        "scramble_steps": [36, 48],
                        "levels_per_combo": 4,
                    },
                    {
                        "grid_size": "10x10",
                        "box_count": 3,
                        "scramble_steps": [50, 70],
                        "levels_per_combo": 3,
                    },
                    {
                        "grid_size": "10x10",
                        "box_count": 4,
                        "scramble_steps": [70, 90],
                        "levels_per_combo": 3,
                    },
                ],
                "procgen_seed": 2026,
                "procgen_wall_density": 0.08,
                "max_levels": None,
                "runs_per_level": 2,
                "max_turns": 300,
                "stagnation_patience": 40,
                "deadlock_patience": 6,
                "prompt_variants": ["full"],
                "tool_variants": ["move_and_query"],
                "state_format": "text",
                "detect_deadlocks": True,
                "terminal_on_deadlock": True,
            },
        },
    }


def _standard_v1_config() -> dict[str, Any]:
    """Canonical cross-game planning benchmark for reproducible model evals."""

    return {
        "spec": "standard-v1",
        "out_dir": "artifacts/runs",
        "record": True,
        "record_raw": False,
        "record_provider_raw": False,
        "provider_retries": 2,
        "provider_backoff": 1.0,
        "stream_debug": False,
        "parallelism": 4,
        "max_inflight_provider": 4,
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
                "stagnation_patience": 40,
                "optimal_turn_cap_multiplier": 4.0,
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
                "stagnation_patience": 80,
                "deadlock_patience": 8,
                "prompt_variants": ["full"],
                "tool_variants": ["move_and_query"],
                "state_format": "text",
                "detect_deadlocks": True,
                "terminal_on_deadlock": True,
            },
        },
    }


def load_builtin_suites() -> None:
    if "easy-v1" not in _REGISTRY:
        register_suite(
            SuiteSpec(
                name="easy-v1",
                description=(
                    "Accessible planning suite across Hanoi and Sokoban tuned "
                    "for smaller-capability models."
                ),
                config_factory=_easy_v1_config,
            )
        )
    if "standard-v1" not in _REGISTRY:
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
