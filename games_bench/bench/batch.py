from __future__ import annotations

import json
from typing import Any

from games_bench.config import load_config, merge_dicts, normalize_games_config
from games_bench.bench.registry import get_benchmark, load_builtin_benchmarks
from games_bench.bench import hanoi as hanoi_bench


def _select_games(
    games_map: dict[str, dict[str, Any]],
    requested: list[str] | None,
) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    return list(games_map.keys())


def main() -> int:
    parser = hanoi_bench.build_parser()
    parser.add_argument(
        "--game",
        action="append",
        dest="games",
        default=None,
        help="Game(s) to run (default: all in config).",
    )

    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}

    load_builtin_benchmarks()
    global_defaults, games_map = normalize_games_config(config, default_game="hanoi")
    selected_games = _select_games(games_map, args.games)
    if not selected_games:
        selected_games = ["hanoi"]

    run_dirs: list[str] = []
    for game_name in selected_games:
        game_config = merge_dicts(global_defaults, games_map.get(game_name, {}))
        benchmark = get_benchmark(game_name)
        game_run_dirs = benchmark.batch_runner(args, game_config)
        run_dirs.extend([str(p) for p in game_run_dirs])

    print(json.dumps({"run_dirs": run_dirs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
