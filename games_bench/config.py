from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    return _expand_env_vars(data)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _expand_env_vars(v) for k, v in value.items()}
    return value


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def normalize_games_config(
    config: dict[str, Any], *, default_game: str = "hanoi"
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if "games" not in config:
        return config, {default_game: {}}

    global_defaults = {k: v for k, v in config.items() if k != "games"}
    games_section = config["games"]
    games_map: dict[str, dict[str, Any]] = {}

    if isinstance(games_section, dict):
        for name, game_config in games_section.items():
            games_map[str(name)] = game_config or {}
        return global_defaults, games_map

    if isinstance(games_section, list):
        for entry in games_section:
            if isinstance(entry, str):
                games_map[entry] = {}
                continue
            if not isinstance(entry, dict):
                raise ValueError("games list entries must be strings or objects")
            name = entry.get("name") or entry.get("game")
            if not name:
                raise ValueError("games list entries must include a name")
            game_config = entry.get("config") or entry.get("overrides")
            if game_config is None:
                game_config = {
                    k: v for k, v in entry.items() if k not in {"name", "game"}
                }
            games_map[str(name)] = game_config
        return global_defaults, games_map

    raise ValueError("games must be an object or list")
