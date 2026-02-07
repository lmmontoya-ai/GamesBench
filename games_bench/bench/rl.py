from __future__ import annotations

import argparse

from games_bench.bench.game_loader import get_specs, parse_env_kwargs


def solve_moves(n: int, start: int, goal: int, aux: int) -> list[tuple[int, int]]:
    if n == 0:
        return []
    return (
        solve_moves(n - 1, start, aux, goal)
        + [(start, goal)]
        + solve_moves(n - 1, aux, goal, start)
    )


def _build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    if args.game == "hanoi":
        env_kwargs.setdefault("step_penalty", -0.01)
        env_kwargs.setdefault("illegal_move_penalty", -1.0)
        env_kwargs.setdefault("solve_reward", 1.0)
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
    for move in solve_moves(env.n_disks, env.start_peg, env.goal_peg, aux=1):
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
