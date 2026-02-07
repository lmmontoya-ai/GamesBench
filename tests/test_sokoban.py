from __future__ import annotations

import unittest

from games_bench.games.sokoban import (
    ACTION_INDEX,
    IllegalMoveError,
    InvalidActionError,
    InvalidLevelError,
    SokobanEnv,
    SokobanToolbox,
    load_bundled_level_set,
    parse_xsb_levels,
)


def _level_from_xsb(xsb: str, *, set_name: str = "unit"):
    return parse_xsb_levels(xsb, set_name=set_name)[0]


class TestSokobanLevelLoading(unittest.TestCase):
    def test_parse_xsb_levels_with_comments_and_multiple_levels(self) -> None:
        text = """; unit:1
; Single push
#####
#@$.#
#####

; unit:2
; Walk then push
######
#@ $.#
######
"""
        levels = parse_xsb_levels(text, set_name="unit")
        self.assertEqual(len(levels), 2)
        self.assertEqual(levels[0].level_id, "unit:1")
        self.assertEqual(levels[0].title, "Single push")
        self.assertEqual(levels[0].n_boxes, 1)
        self.assertEqual(levels[1].level_id, "unit:2")
        self.assertEqual(levels[1].title, "Walk then push")

    def test_parse_xsb_levels_rejects_invalid_symbol(self) -> None:
        text = """#####
#@x.#
#####
"""
        with self.assertRaises(InvalidLevelError):
            parse_xsb_levels(text, set_name="bad")

    def test_parse_xsb_levels_requires_box_goal_balance(self) -> None:
        text = """#####
#@$.#
# . #
#####
"""
        with self.assertRaises(InvalidLevelError):
            parse_xsb_levels(text, set_name="bad")

    def test_load_bundled_level_set_starter(self) -> None:
        level_set = load_bundled_level_set("starter-authored-v1")
        self.assertEqual(level_set.name, "starter-authored-v1")
        self.assertEqual(len(level_set.levels), 3)
        self.assertTrue(level_set.levels[0].known_optimal)
        self.assertFalse(level_set.levels[2].known_optimal)


class TestSokobanEnv(unittest.TestCase):
    def test_reset_and_legal_moves(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level)
        state = env.reset()
        self.assertEqual(state.player, (1, 1))
        self.assertEqual(state.boxes, frozenset({(1, 2)}))
        self.assertEqual(env.get_legal_moves(), ["right"])

    def test_move_and_solve(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level)
        self.assertFalse(env.is_solved())
        state = env.move("right")
        self.assertTrue(env.is_solved())
        self.assertEqual(env.move_count, 1)
        self.assertEqual(env.push_count, 1)
        self.assertEqual(state.boxes, frozenset({(1, 3)}))

    def test_illegal_move_raises(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level)
        with self.assertRaises(IllegalMoveError):
            env.move("left")

    def test_chain_push_rejected(self) -> None:
        level = _level_from_xsb(
            """########
#@$$ ..#
########
"""
        )
        env = SokobanEnv(level)
        with self.assertRaises(IllegalMoveError):
            env.move("right")

    def test_step_penalizes_illegal_by_default(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level, illegal_move_penalty=-2.0)
        state_before = env.get_state()
        state_after, reward, done, info = env.step("left")
        self.assertEqual(state_after, state_before)
        self.assertEqual(reward, -2.0)
        self.assertFalse(done)
        self.assertTrue(info["illegal_action"])

    def test_step_accepts_int_action_and_rewards(self) -> None:
        level = _level_from_xsb(
            """######
#@ $.#
######
"""
        )
        env = SokobanEnv(level, step_penalty=-0.1, push_reward=0.5, solve_reward=1.0)

        state1, reward1, done1, info1 = env.step(ACTION_INDEX["right"])
        self.assertFalse(done1)
        self.assertEqual(info1["action_type"], "walk")
        self.assertAlmostEqual(reward1, -0.1)
        self.assertEqual(state1.player, (1, 2))

        state2, reward2, done2, info2 = env.step("right")
        self.assertTrue(done2)
        self.assertEqual(info2["action_type"], "push")
        self.assertAlmostEqual(reward2, 1.4)
        self.assertEqual(state2.boxes, frozenset({(1, 4)}))

    def test_step_raise_behavior_raises_on_bad_action(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level, illegal_action_behavior="raise")
        with self.assertRaises(InvalidActionError):
            env.step("diagonal")

    def test_undo_reverts_state_and_counts(self) -> None:
        level = _level_from_xsb(
            """######
#@ $.#
######
"""
        )
        env = SokobanEnv(level, record_history=True)

        env.move("right")
        env.move("right")
        self.assertEqual(env.move_count, 2)
        self.assertEqual(env.push_count, 1)
        self.assertEqual(len(env.history), 2)

        env.undo()
        self.assertEqual(env.move_count, 1)
        self.assertEqual(env.push_count, 0)
        self.assertEqual(len(env.history), 1)

        env.undo()
        self.assertEqual(env.move_count, 0)
        self.assertEqual(env.push_count, 0)
        self.assertEqual(len(env.history), 0)

        with self.assertRaises(IllegalMoveError):
            env.undo()

    def test_format_prompt_state(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level)
        text = env.format_prompt_state(include_legal_moves=True)
        self.assertIn("Board", text)
        self.assertIn("Boxes on goals", text)
        self.assertIn("Legal moves", text)


class TestSokobanToolbox(unittest.TestCase):
    def test_toolbox_methods(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level)
        toolbox = SokobanToolbox(env)

        state_result = toolbox.get_state()
        self.assertTrue(state_result["ok"])
        self.assertIn("xsb", state_result["state"])

        legal_result = toolbox.get_legal_moves()
        self.assertTrue(legal_result["ok"])
        self.assertEqual(legal_result["legal_moves"], ["right"])

        move_result = toolbox.move("right")
        self.assertTrue(move_result["ok"])
        self.assertEqual(move_result["direction"], "right")
        self.assertEqual(move_result["boxes_on_goals"], 1)

        undo_result = toolbox.undo()
        self.assertTrue(undo_result["ok"])
        self.assertEqual(undo_result["boxes_on_goals"], 0)

        undo_fail = toolbox.undo()
        self.assertFalse(undo_fail["ok"])
        self.assertIn("cannot undo", undo_fail["error"])


if __name__ == "__main__":
    unittest.main()
