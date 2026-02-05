from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class GameSpec:
    name: str
    description: str
    env_factory: Callable[..., Any]


_REGISTRY: dict[str, GameSpec] = {}


def register_game(spec: GameSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_game(name: str) -> GameSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown game: {name}")
    return _REGISTRY[name]


def list_games() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_builtin_games() -> None:
    if _REGISTRY:
        return
    from games_bench.games.hanoi.env import TowerOfHanoiEnv

    register_game(
        GameSpec(
            name="hanoi",
            description="Tower of Hanoi",
            env_factory=TowerOfHanoiEnv,
        )
    )
