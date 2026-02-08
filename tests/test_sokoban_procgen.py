from __future__ import annotations

import unittest

from games_bench.games.sokoban import SokobanEnv
from games_bench.games.sokoban.procgen import (
    generate_procedural_level,
    generate_procedural_level_with_solution,
    generate_procedural_levels,
    parse_grid_size,
)


class TestSokobanProcgen(unittest.TestCase):
    def test_parse_grid_size(self) -> None:
        self.assertEqual(parse_grid_size("8x10"), (8, 10))
        self.assertEqual(parse_grid_size(" 12X7 "), (12, 7))
        with self.assertRaises(ValueError):
            parse_grid_size("8")
        with self.assertRaises(ValueError):
            parse_grid_size("axb")

    def test_generate_procedural_level_is_deterministic_with_seed(self) -> None:
        level_a = generate_procedural_level(
            width=8,
            height=8,
            n_boxes=2,
            seed=123,
            wall_density=0.0,
            scramble_steps=18,
            level_id="procgen:deterministic",
        )
        level_b = generate_procedural_level(
            width=8,
            height=8,
            n_boxes=2,
            seed=123,
            wall_density=0.0,
            scramble_steps=18,
            level_id="procgen:deterministic",
        )
        self.assertEqual(level_a.xsb, level_b.xsb)
        self.assertEqual(level_a.boxes_start, level_b.boxes_start)
        self.assertEqual(level_a.player_start, level_b.player_start)

    def test_generated_level_matches_requested_shape_and_boxes(self) -> None:
        level = generate_procedural_level(
            width=9,
            height=8,
            n_boxes=3,
            seed=7,
            wall_density=0.1,
            scramble_steps=24,
        )
        self.assertEqual(level.width, 9)
        self.assertEqual(level.height, 8)
        self.assertEqual(level.n_boxes, 3)
        self.assertEqual(len(level.goals), 3)
        self.assertEqual(len(level.boxes_start), 3)
        self.assertFalse(level.known_optimal)

    def test_generated_solution_replays_to_solved_state(self) -> None:
        level, solution = generate_procedural_level_with_solution(
            width=8,
            height=8,
            n_boxes=2,
            seed=42,
            wall_density=0.05,
            scramble_steps=20,
        )
        env = SokobanEnv(
            level,
            illegal_action_behavior="raise",
            detect_deadlocks=False,
            terminal_on_deadlock=False,
        )
        self.assertFalse(env.is_solved())
        for direction in solution:
            env.step(direction)
        self.assertTrue(env.is_solved())

    def test_generate_procedural_levels_returns_unique_level_ids(self) -> None:
        levels = generate_procedural_levels(
            width=8,
            height=8,
            n_boxes=2,
            count=3,
            seed=10,
            wall_density=0.0,
            scramble_steps=16,
        )
        self.assertEqual(len(levels), 3)
        self.assertEqual(len({level.level_id for level in levels}), 3)

    def test_invalid_generation_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            generate_procedural_level(width=5, height=8, n_boxes=1)
        with self.assertRaises(ValueError):
            generate_procedural_level(width=8, height=8, n_boxes=0)
        with self.assertRaises(ValueError):
            generate_procedural_level(width=8, height=8, n_boxes=2, wall_density=0.9)


if __name__ == "__main__":
    unittest.main()
