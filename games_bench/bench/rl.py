from __future__ import annotations

import argparse
from functools import lru_cache

from games_bench.bench.game_loader import get_specs, parse_env_kwargs


@lru_cache(maxsize=None)
def _frame_stewart_steps(n_disks: int, n_pegs: int) -> int:
    if n_disks == 0:
        return 0
    if n_disks == 1:
        return 1
    if n_pegs == 3:
        return (1 << n_disks) - 1
    return min(
        2 * _frame_stewart_steps(split, n_pegs)
        + _frame_stewart_steps(n_disks - split, n_pegs - 1)
        for split in range(1, n_disks)
    )


def _best_split(n_disks: int, n_pegs: int) -> int:
    best_split = 1
    best_cost = _frame_stewart_steps(n_disks, n_pegs)
    for split in range(1, n_disks):
        candidate = 2 * _frame_stewart_steps(split, n_pegs) + _frame_stewart_steps(
            n_disks - split, n_pegs - 1
        )
        if candidate < best_cost:
            best_cost = candidate
            best_split = split
    return best_split


def solve_moves(
    n_disks: int, start: int, goal: int, peg_ids: list[int]
) -> list[tuple[int, int]]:
    if n_disks == 0:
        return []
    if n_disks == 1:
        return [(start, goal)]
    if len(peg_ids) < 3:
        raise ValueError("Tower of Hanoi requires at least 3 pegs.")
    if len(peg_ids) == 3:
        aux = next(peg for peg in peg_ids if peg not in {start, goal})
        return (
            solve_moves(n_disks - 1, start, aux, peg_ids)
            + [(start, goal)]
            + solve_moves(n_disks - 1, aux, goal, peg_ids)
        )

    split = _best_split(n_disks, len(peg_ids))
    aux_target = next(peg for peg in peg_ids if peg not in {start, goal})
    reduced_pegs = [peg for peg in peg_ids if peg != aux_target]
    return (
        solve_moves(split, start, aux_target, peg_ids)
        + solve_moves(n_disks - split, start, goal, reduced_pegs)
        + solve_moves(split, aux_target, goal, peg_ids)
    )


def _build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    if args.game == "hanoi":
        env_kwargs.setdefault("step_penalty", -0.01)
        env_kwargs.setdefault("illegal_move_penalty", -1.0)
        env_kwargs.setdefault("solve_reward", 1.0)
        if args.n_pegs is not None:
            env_kwargs["n_pegs"] = args.n_pegs
        if args.n_disks is not None:
            env_kwargs["n_disks"] = args.n_disks
    return env_kwargs


def main() -> int:
    parser = argparse.ArgumentParser(description="RL demo for a registered game.")
    parser.add_argument("--game", default="hanoi", help="Registered game name.")
    parser.add_argument(
        "--env-kwargs",
        default=None,
        help="JSON object of kwargs for the selected game's env factory.",
    )
    parser.add_argument(
        "--n-disks",
        type=int,
        default=3,
        help="Hanoi convenience flag (overrides env_kwargs.n_disks).",
    )
    parser.add_argument(
        "--n-pegs",
        type=int,
        default=3,
        help="Hanoi convenience flag (overrides env_kwargs.n_pegs).",
    )
    args = parser.parse_args()

    game_spec, _bench_spec = get_specs(args.game)
    env = game_spec.env_factory(**_build_env_kwargs(args))
    if args.game != "hanoi":
        print(
            f"Created environment for game '{args.game}'. RL scripted demo is Hanoi-only."
        )
        return 0

    env.reset(env.n_disks)

    episode_reward = 0.0
    peg_ids = list(range(env.n_pegs))
    for move in solve_moves(env.n_disks, env.start_peg, env.goal_peg, peg_ids):
        _state, reward, done, _info = env.step(move)
        episode_reward += reward
        if done:
            break

    print("Solved:", env.is_solved())
    print("Moves:", env.move_count, "(optimal:", env.optimal_steps(), ")")
    print("Episode reward:", episode_reward)
    print("\nPrompt state:\n", env.format_prompt_state())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
