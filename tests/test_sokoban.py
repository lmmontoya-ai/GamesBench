from __future__ import annotations

import unittest

from games_bench.games.sokoban import (
    ACTION_INDEX,
    compute_dead_squares,
    has_dead_square_deadlock,
    has_freeze_deadlock,
    IllegalMoveError,
    InvalidActionError,
    InvalidLevelError,
    SokobanEnv,
    SokobanError,
    SokobanToolbox,
    load_bundled_level_set,
    parse_xsb_levels,
    tool_schemas,
)
from games_bench.games.sokoban.factory import make_sokoban_env


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

    def test_parse_xsb_levels_supports_composite_symbols(self) -> None:
        level = _level_from_xsb(
            """######
#+*$ #
######
"""
        )
        self.assertEqual(level.player_start, (1, 1))
        self.assertEqual(level.boxes_start, frozenset({(1, 2), (1, 3)}))
        self.assertEqual(level.goals, frozenset({(1, 1), (1, 2)}))

    def test_parse_xsb_levels_rejects_multiple_players(self) -> None:
        text = """######
#@@$.#
######
"""
        with self.assertRaises(InvalidLevelError):
            parse_xsb_levels(text, set_name="bad")

    def test_parse_xsb_levels_rejects_missing_player(self) -> None:
        text = """######
#  $.#
######
"""
        with self.assertRaises(InvalidLevelError):
            parse_xsb_levels(text, set_name="bad")

    def test_parse_xsb_levels_rejects_missing_boxes(self) -> None:
        text = """######
# @ .#
######
"""
        with self.assertRaises(InvalidLevelError):
            parse_xsb_levels(text, set_name="bad")

    def test_parse_xsb_levels_rejects_empty_input(self) -> None:
        with self.assertRaises(InvalidLevelError):
            parse_xsb_levels("", set_name="bad")

    def test_parse_xsb_levels_title_does_not_prefix_match_other_ids(self) -> None:
        text = """; unit:10
#####
#@$.#
#####
"""
        levels = parse_xsb_levels(text, set_name="unit")
        self.assertEqual(levels[0].title, "unit:10")

    def test_load_bundled_level_set_starter(self) -> None:
        level_set = load_bundled_level_set("starter-authored-v1")
        self.assertEqual(level_set.name, "starter-authored-v1")
        self.assertEqual(len(level_set.levels), 3)
        self.assertTrue(level_set.levels[0].known_optimal)
        self.assertFalse(level_set.levels[2].known_optimal)


class TestSokobanEnv(unittest.TestCase):
    def test_factory_supports_procedural_generation(self) -> None:
        env = make_sokoban_env(
            procedural=True,
            width=8,
            height=8,
            n_boxes=2,
            procgen_seed=11,
            procgen_wall_density=0.0,
            procgen_scramble_steps=16,
        )
        self.assertEqual(env.level.width, 8)
        self.assertEqual(env.level.height, 8)
        self.assertEqual(env.level.n_boxes, 2)
        self.assertFalse(env.is_solved())

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

    def test_legal_moves_cover_all_directions(self) -> None:
        level = _level_from_xsb(
            """#######
#     #
#  @  #
#  $ .#
#     #
#######
"""
        )
        env = SokobanEnv(level)
        self.assertEqual(env.get_legal_moves(), ["up", "down", "left", "right"])

        env.move("up")
        self.assertEqual(env.get_state().player, (1, 3))
        env.reset()

        env.move("down")
        self.assertEqual(env.get_state().player, (3, 3))
        env.reset()

        env.move("left")
        self.assertEqual(env.get_state().player, (2, 2))
        env.reset()

        env.move("right")
        self.assertEqual(env.get_state().player, (2, 4))

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

    def test_push_into_wall_rejected(self) -> None:
        level = _level_from_xsb(
            """######
#@$#.#
######
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

    def test_step_terminate_behavior_ends_on_illegal(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level, illegal_action_behavior="terminate")
        _state, reward, done, info = env.step("left")
        self.assertTrue(done)
        self.assertEqual(reward, env.illegal_move_penalty)
        self.assertTrue(info["illegal_action"])
        self.assertFalse(info["truncated"])

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

    def test_step_push_off_goal_penalty(self) -> None:
        level = _level_from_xsb(
            """#######
# @*  #
#######
"""
        )
        env = SokobanEnv(level, step_penalty=-0.1, push_off_penalty=-0.4)
        _state, reward, done, info = env.step("right")
        self.assertFalse(done)
        self.assertEqual(info["action_type"], "push")
        self.assertAlmostEqual(reward, -0.5)
        self.assertFalse(info["solved"])

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

    def test_step_raise_behavior_raises_on_illegal_move(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level, illegal_action_behavior="raise")
        with self.assertRaises(IllegalMoveError):
            env.step("left")

    def test_step_max_steps_truncates_legal_action(self) -> None:
        level = _level_from_xsb(
            """######
#@ $.#
######
"""
        )
        env = SokobanEnv(level, max_steps=1)
        _state, _reward, done, info = env.step("right")
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["illegal_action"])

    def test_step_max_steps_truncates_illegal_action(self) -> None:
        level = _level_from_xsb(
            """######
#@ $.#
######
"""
        )
        env = SokobanEnv(level, max_steps=1, illegal_action_behavior="penalize")
        _state, _reward, done, info = env.step("left")
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        self.assertTrue(info["illegal_action"])

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

        with self.assertRaises(SokobanError):
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

    def test_format_prompt_state_includes_deadlock_status(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level, detect_deadlocks=True)
        text = env.format_prompt_state(include_deadlock_status=True)
        self.assertIn("Deadlocked: False", text)


class TestSokobanDeadlocks(unittest.TestCase):
    def test_compute_dead_squares_marks_corner_and_not_goal(self) -> None:
        level = _level_from_xsb(
            """#####
#@ .#
#$  #
#####
"""
        )
        dead_squares = compute_dead_squares(
            width=level.width,
            height=level.height,
            walls=level.walls,
            goals=level.goals,
        )
        self.assertIn((2, 1), dead_squares)
        self.assertNotIn((1, 3), dead_squares)
        self.assertNotIn((1, 2), dead_squares)

    def test_env_detects_dead_square_deadlock(self) -> None:
        level = _level_from_xsb(
            """#####
#@ .#
#$  #
#####
"""
        )
        env = SokobanEnv(level)
        self.assertTrue(env.is_deadlocked())

    def test_has_dead_square_deadlock_helper(self) -> None:
        level = _level_from_xsb(
            """#####
#@ .#
#$  #
#####
"""
        )
        dead_squares = compute_dead_squares(
            width=level.width,
            height=level.height,
            walls=level.walls,
            goals=level.goals,
        )
        self.assertTrue(
            has_dead_square_deadlock(
                boxes=level.boxes_start,
                goals=level.goals,
                dead_squares=dead_squares,
            )
        )

    def test_goals_in_corners_are_not_dead_squares(self) -> None:
        level = _level_from_xsb(
            """#####
#@  #
#.$ #
#####
"""
        )
        dead_squares = compute_dead_squares(
            width=level.width,
            height=level.height,
            walls=level.walls,
            goals=level.goals,
        )
        self.assertNotIn((2, 1), dead_squares)

    def test_compute_dead_squares_marks_dead_edge_cells(self) -> None:
        level = _level_from_xsb(
            """#######
#@$   #
#   . #
#######
"""
        )
        dead_squares = compute_dead_squares(
            width=level.width,
            height=level.height,
            walls=level.walls,
            goals=level.goals,
        )
        self.assertIn((1, 2), dead_squares)

    def test_compute_dead_squares_marks_right_corner_orientations(self) -> None:
        cases = [
            (
                "top-right",
                """#####
# @$#
# . #
#####
""",
                (1, 3),
            ),
            (
                "bottom-right",
                """#####
# @ #
# .$#
#####
""",
                (2, 3),
            ),
        ]
        for case_name, xsb, corner_pos in cases:
            with self.subTest(case=case_name):
                level = _level_from_xsb(xsb)
                dead_squares = compute_dead_squares(
                    width=level.width,
                    height=level.height,
                    walls=level.walls,
                    goals=level.goals,
                )
                self.assertIn(corner_pos, dead_squares)

    def test_solvable_nontrivial_state_not_deadlocked(self) -> None:
        level = _level_from_xsb(
            """#######
#     #
# $@ ##
#     #
# .   #
#######
"""
        )
        env = SokobanEnv(level)
        self.assertFalse(env.is_deadlocked())

    def test_env_ignores_frozen_box_on_goal(self) -> None:
        level = _level_from_xsb(
            """#####
# @ #
#*  #
#####
"""
        )
        env = SokobanEnv(level)
        self.assertTrue(env.is_solved())
        self.assertFalse(env.is_deadlocked())

    def test_step_deadlock_terminates_when_enabled(self) -> None:
        level = _level_from_xsb(
            """######
#@   #
#$ . #
######
"""
        )
        env = SokobanEnv(
            level,
            detect_deadlocks=True,
            terminal_on_deadlock=True,
            deadlock_penalty=-3.0,
        )
        _state, reward, done, info = env.step("right")
        self.assertTrue(done)
        self.assertEqual(reward, -3.0)
        self.assertTrue(info["deadlocked"])
        self.assertTrue(info["deadlock_terminated"])

    def test_step_deadlock_does_not_terminate_when_disabled(self) -> None:
        level = _level_from_xsb(
            """######
#@   #
#$ . #
######
"""
        )
        env = SokobanEnv(
            level,
            detect_deadlocks=True,
            terminal_on_deadlock=False,
            deadlock_penalty=-3.0,
        )
        _state, reward, done, info = env.step("right")
        self.assertFalse(done)
        self.assertEqual(reward, -3.0)
        self.assertTrue(info["deadlocked"])
        self.assertNotIn("deadlock_terminated", info)

    def test_push_creates_deadlock(self) -> None:
        level = _level_from_xsb(
            """#######
#     #
# $@ ##
#     #
# .   #
#######
"""
        )
        env = SokobanEnv(
            level,
            detect_deadlocks=True,
            terminal_on_deadlock=True,
            deadlock_penalty=-2.0,
        )
        self.assertFalse(env.is_deadlocked())
        _state, reward, done, info = env.step("left")
        self.assertTrue(done)
        self.assertEqual(reward, -2.0)
        self.assertTrue(info["deadlocked"])
        self.assertTrue(info["deadlock_terminated"])

    def test_detect_deadlocks_false_disables_deadlock_handling(self) -> None:
        level = _level_from_xsb(
            """#######
#     #
# $@ ##
#     #
# .   #
#######
"""
        )
        env = SokobanEnv(
            level,
            detect_deadlocks=False,
            terminal_on_deadlock=True,
            deadlock_penalty=-2.0,
        )
        _state, reward, done, info = env.step("left")
        self.assertFalse(done)
        self.assertEqual(reward, 0.0)
        self.assertFalse(info["deadlocked"])
        self.assertNotIn("deadlock_terminated", info)

    def test_undo_clears_deadlock_from_push(self) -> None:
        level = _level_from_xsb(
            """#######
#     #
# $@ ##
#     #
# .   #
#######
"""
        )
        env = SokobanEnv(
            level,
            detect_deadlocks=True,
            terminal_on_deadlock=False,
        )
        self.assertFalse(env.is_deadlocked())
        _state, _reward, done, info = env.step("left")
        self.assertFalse(done)
        self.assertTrue(info["deadlocked"])
        self.assertTrue(env.is_deadlocked())
        env.undo()
        self.assertFalse(env.is_deadlocked())

    def test_illegal_parse_action_terminates_when_state_deadlocked(self) -> None:
        level = _level_from_xsb(
            """#####
#@ .#
#$  #
#####
"""
        )
        env = SokobanEnv(
            level,
            detect_deadlocks=True,
            terminal_on_deadlock=True,
            illegal_action_behavior="penalize",
        )
        _state, _reward, done, info = env.step("diagonal")
        self.assertTrue(info["illegal_action"])
        self.assertTrue(info["deadlocked"])
        self.assertTrue(done)
        self.assertTrue(info["deadlock_terminated"])

    def test_illegal_move_terminates_when_state_deadlocked(self) -> None:
        level = _level_from_xsb(
            """#####
#@ .#
#$  #
#####
"""
        )
        env = SokobanEnv(
            level,
            detect_deadlocks=True,
            terminal_on_deadlock=True,
            illegal_action_behavior="penalize",
        )
        _state, _reward, done, info = env.step("left")
        self.assertTrue(info["illegal_action"])
        self.assertTrue(info["deadlocked"])
        self.assertTrue(done)
        self.assertTrue(info["deadlock_terminated"])

    def test_freeze_deadlock_detected(self) -> None:
        level = _level_from_xsb(
            """#######
#.@   #
#  #  #
##$$# #
#  #  #
#    .#
#######
"""
        )
        self.assertTrue(
            has_freeze_deadlock(
                width=level.width,
                height=level.height,
                walls=level.walls,
                boxes=level.boxes_start,
                goals=level.goals,
            )
        )

    def test_freeze_detection_ignores_goal_box(self) -> None:
        level = _level_from_xsb(
            """#######
#.@   #
#  #  #
##$*# #
#  #  #
#     #
#######
"""
        )
        self.assertFalse(
            has_freeze_deadlock(
                width=level.width,
                height=level.height,
                walls=level.walls,
                boxes=level.boxes_start,
                goals=level.goals,
            )
        )


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
        self.assertEqual(move_result["action_type"], "push")
        self.assertFalse(move_result["deadlocked"])
        self.assertEqual(move_result["boxes_on_goals"], 1)
        self.assertTrue(toolbox.is_solved()["solved"])

        undo_result = toolbox.undo()
        self.assertTrue(undo_result["ok"])
        self.assertEqual(undo_result["boxes_on_goals"], 0)

        undo_fail = toolbox.undo()
        self.assertFalse(undo_fail["ok"])
        self.assertIn("cannot undo", undo_fail["error"])

    def test_toolbox_move_invalid_direction(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        env = SokobanEnv(level)
        toolbox = SokobanToolbox(env)
        result = toolbox.move("diagonal")
        self.assertFalse(result["ok"])
        self.assertIn("direction string", result["error"])


class TestSokobanSchemas(unittest.TestCase):
    def test_all_tools_expose_response_schemas(self) -> None:
        schemas = tool_schemas()
        self.assertEqual(len(schemas), 5)
        for schema in schemas:
            self.assertIn("response_schema", schema)


if __name__ == "__main__":
    unittest.main()
