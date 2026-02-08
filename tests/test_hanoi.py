from __future__ import annotations

import unittest

from games_bench.games.hanoi.env import (
    IllegalMoveError,
    InvalidPegError,
    TowerOfHanoiEnv,
    tool_schemas,
)


class TestTowerOfHanoiEnv(unittest.TestCase):
    def test_reset_and_legal_moves(self) -> None:
        env = TowerOfHanoiEnv(n_disks=3)
        state = env.reset(3)
        self.assertEqual(state.pegs[0], (3, 2, 1))
        self.assertEqual(state.pegs[1], ())
        self.assertEqual(state.pegs[2], ())

        legal = set(env.get_legal_moves())
        self.assertEqual(legal, {(0, 1), (0, 2)})

    def test_move_and_solve(self) -> None:
        env = TowerOfHanoiEnv(n_disks=1)
        self.assertFalse(env.is_solved())
        env.move(0, 2)
        self.assertTrue(env.is_solved())

    def test_illegal_move_raises(self) -> None:
        env = TowerOfHanoiEnv(n_disks=3)
        with self.assertRaises(IllegalMoveError):
            env.move(1, 2)  # from empty peg

    def test_step_penalizes_illegal_by_default(self) -> None:
        env = TowerOfHanoiEnv(
            n_disks=3, illegal_action_behavior="penalize", illegal_move_penalty=-2.0
        )
        state0 = env.get_state()
        state1, reward, done, info = env.step((1, 2))  # from empty peg
        self.assertEqual(state1, state0)
        self.assertEqual(reward, -2.0)
        self.assertFalse(done)
        self.assertTrue(info["illegal_action"])

    def test_step_accepts_int_action(self) -> None:
        env = TowerOfHanoiEnv(n_disks=3)
        state1, reward, done, info = env.step(0)  # action_space[0] == (0,1)
        self.assertEqual(info["action"], (0, 1))
        self.assertEqual(state1.pegs[0], (3, 2))
        self.assertEqual(state1.pegs[1], (1,))
        self.assertFalse(done)
        self.assertEqual(reward, 0.0)

    def test_variable_peg_count_is_supported(self) -> None:
        env = TowerOfHanoiEnv(n_disks=2, n_pegs=4, goal_peg=3)
        state = env.reset(2)
        self.assertEqual(state.n_pegs, 4)
        self.assertEqual(len(state.pegs), 4)
        self.assertEqual(set(env.get_legal_moves()), {(0, 1), (0, 2), (0, 3)})

        encoded = env.encode_action((0, 3))
        state1, _reward, _done, info = env.step(encoded)
        self.assertEqual(info["action"], (0, 3))
        self.assertEqual(state1.pegs[3], (1,))

    def test_invalid_goal_for_peg_count_raises(self) -> None:
        with self.assertRaises(InvalidPegError):
            TowerOfHanoiEnv(n_disks=2, n_pegs=4, goal_peg=4)

    def test_optimal_steps_uses_multi_peg_reference(self) -> None:
        self.assertEqual(TowerOfHanoiEnv(n_disks=3, n_pegs=4).optimal_steps(), 5)
        self.assertEqual(TowerOfHanoiEnv(n_disks=4, n_pegs=4).optimal_steps(), 9)

    def test_tool_schema_uses_dynamic_peg_bounds(self) -> None:
        move_schema = next(
            schema
            for schema in tool_schemas(n_pegs=4)
            if schema["name"] == "hanoi_move"
        )
        from_peg_schema = move_schema["parameters"]["properties"]["from_peg"]
        to_peg_schema = move_schema["parameters"]["properties"]["to_peg"]
        self.assertEqual(from_peg_schema["maximum"], 3)
        self.assertEqual(to_peg_schema["maximum"], 3)


if __name__ == "__main__":
    unittest.main()
