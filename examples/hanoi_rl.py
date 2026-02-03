from __future__ import annotations

from games_bench.hanoi import TowerOfHanoiEnv


def solve_moves(n: int, start: int, goal: int, aux: int) -> list[tuple[int, int]]:
    if n == 0:
        return []
    return (
        solve_moves(n - 1, start, aux, goal)
        + [(start, goal)]
        + solve_moves(n - 1, aux, goal, start)
    )


def main() -> None:
    env = TowerOfHanoiEnv(
        n_disks=3,
        step_penalty=-0.01,
        illegal_move_penalty=-1.0,
        solve_reward=1.0,
    )
    env.reset(3)

    episode_reward = 0.0
    for move in solve_moves(env.n_disks, env.start_peg, env.goal_peg, aux=1):
        state, reward, done, info = env.step(move)
        episode_reward += reward
        if done:
            break

    print("Solved:", env.is_solved())
    print("Moves:", env.move_count, "(optimal:", env.optimal_steps(), ")")
    print("Episode reward:", episode_reward)
    print("\nPrompt state:\n", env.format_prompt_state())


if __name__ == "__main__":
    main()
