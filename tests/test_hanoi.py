from __future__ import annotations

import unittest

from games_bench.hanoi import IllegalMoveError, TowerOfHanoiEnv


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


if __name__ == "__main__":
    unittest.main()
