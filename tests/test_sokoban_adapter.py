from __future__ import annotations

import unittest
from typing import Any

from games_bench.games.sokoban import (
    SokobanEnv,
    SokobanGameAdapter,
    default_instructions,
    instructions_for_variant,
    parse_xsb_levels,
    with_image_instructions,
)
from games_bench.llm.harness import run_tool_calling_episode
from games_bench.llm.providers import ProviderResult, ToolCall


def _level_from_xsb(xsb: str, *, set_name: str = "unit"):
    return parse_xsb_levels(xsb, set_name=set_name)[0]


class _ScriptedProvider:
    supports_images = False

    def __init__(self, results: list[ProviderResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def next_tool_calls(
        self,
        *,
        state_text: str,
        tool_schemas: list[dict[str, Any]],
        instructions: str,
        state_image: dict[str, Any] | None = None,
        conversation: list[dict[str, Any]] | None = None,
    ) -> ProviderResult:
        del state_text, tool_schemas, instructions, state_image, conversation
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return ProviderResult([], raw=None, error="No scripted response")


class TestSokobanPrompts(unittest.TestCase):
    def test_prompt_variants(self) -> None:
        minimal = instructions_for_variant("minimal")
        self.assertEqual(minimal, default_instructions())
        self.assertIn("number of tool calls needed", minimal)
        self.assertIn("all boxes are on goals at the same time", minimal)
        self.assertIn("If state query tools are available", minimal)
        for symbol in ("`#`", "` `", "`@`", "`+`", "`$`", "`*`", "`.`"):
            self.assertIn(symbol, minimal)

        legal = instructions_for_variant("with_legal_moves")
        self.assertIn("sokoban_get_legal_moves", legal)

        deadlock = instructions_for_variant("with_deadlock_warnings")
        self.assertIn("deadlocked state", deadlock)

        full = instructions_for_variant("full")
        self.assertIn("sokoban_get_legal_moves", full)
        self.assertIn("deadlocked state", full)

    def test_unknown_prompt_variant_raises(self) -> None:
        with self.assertRaises(ValueError):
            instructions_for_variant("unknown")

    def test_with_image_instructions_is_idempotent(self) -> None:
        base = default_instructions()
        with_image = with_image_instructions(base)
        self.assertNotEqual(base, with_image)
        self.assertEqual(with_image, with_image_instructions(with_image))
        self.assertIn("dark gray tiles = walls", with_image)
        self.assertIn("blue token = player on goal", with_image)
        self.assertIn("If both text state and image are provided", with_image)

    def test_prompt_variant_respects_custom_tool_prefix(self) -> None:
        legal = instructions_for_variant("with_legal_moves", tool_prefix="custom")
        self.assertIn("custom_get_legal_moves", legal)
        self.assertNotIn("sokoban_get_legal_moves", legal)


class TestSokobanGameAdapter(unittest.TestCase):
    def test_move_tool_meta_legal_and_illegal(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))

        legal = adapter.execute_tool("sokoban_move", {"direction": "right"})
        self.assertTrue(legal.result["ok"])
        self.assertEqual(legal.meta["action_kind"], "move")
        self.assertTrue(legal.meta["state_mutating"])
        self.assertFalse(legal.meta["illegal_action"])
        self.assertTrue(legal.meta["counts_as_move"])

        illegal = adapter.execute_tool("sokoban_move", {"direction": "up"})
        self.assertFalse(illegal.result["ok"])
        self.assertEqual(illegal.meta["action_kind"], "move")
        self.assertFalse(illegal.meta["state_mutating"])
        self.assertTrue(illegal.meta["illegal_action"])
        self.assertFalse(illegal.meta["counts_as_move"])

    def test_move_tool_requests_termination_on_terminal_deadlock(self) -> None:
        level = _level_from_xsb(
            """#####
#@$ #
#  .#
#####
"""
        )
        env = SokobanEnv(level, detect_deadlocks=True, terminal_on_deadlock=True)
        adapter = SokobanGameAdapter(env)
        execution = adapter.execute_tool("sokoban_move", {"direction": "right"})
        self.assertTrue(execution.result["ok"])
        self.assertTrue(execution.result["deadlocked"])
        self.assertTrue(execution.meta["terminate_episode"])
        self.assertEqual(execution.meta["termination_reason"], "deadlock_terminal")

    def test_undo_tool_meta_success_and_failure(self) -> None:
        level = _level_from_xsb(
            """######
#@ $.#
######
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))

        empty = adapter.execute_tool("sokoban_undo", {})
        self.assertFalse(empty.result["ok"])
        self.assertEqual(empty.meta["action_kind"], "undo")
        self.assertFalse(empty.meta["state_mutating"])
        self.assertFalse(empty.meta["illegal_action"])
        self.assertFalse(empty.meta["counts_as_move"])

        adapter.execute_tool("sokoban_move", {"direction": "right"})
        successful = adapter.execute_tool("sokoban_undo", {})
        self.assertTrue(successful.result["ok"])
        self.assertEqual(successful.meta["action_kind"], "undo")
        self.assertTrue(successful.meta["state_mutating"])
        self.assertFalse(successful.meta["illegal_action"])
        self.assertFalse(successful.meta["counts_as_move"])

    def test_query_tools_meta(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))
        for tool_name in (
            "sokoban_get_state",
            "sokoban_is_solved",
            "sokoban_get_legal_moves",
        ):
            with self.subTest(tool=tool_name):
                execution = adapter.execute_tool(tool_name, {})
                self.assertTrue(execution.result["ok"])
                self.assertEqual(execution.meta["action_kind"], "query")
                self.assertFalse(execution.meta["state_mutating"])
                self.assertFalse(execution.meta["illegal_action"])
                self.assertFalse(execution.meta["counts_as_move"])

    def test_unknown_tool_is_flagged_illegal(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))
        execution = adapter.execute_tool("sokoban_missing", {})
        self.assertFalse(execution.result["ok"])
        self.assertIn("unknown tool", execution.result["error"])
        self.assertEqual(execution.meta["action_kind"], "query")
        self.assertFalse(execution.meta["illegal_action"])
        self.assertFalse(execution.meta["state_mutating"])
        self.assertFalse(execution.meta["counts_as_move"])

    def test_episode_metrics_contract(self) -> None:
        level = _level_from_xsb(
            """######
#@ $.#
######
"""
        )
        env = SokobanEnv(level, record_history=True)
        adapter = SokobanGameAdapter(env)
        adapter.execute_tool("sokoban_move", {"direction": "right"})
        metrics = adapter.episode_metrics()
        for key in (
            "level_id",
            "n_boxes",
            "grid_size",
            "move_count",
            "push_count",
            "boxes_on_goals",
            "deadlocked",
            "optimal_moves",
            "optimal_pushes",
            "known_optimal",
            "history",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["level_id"], level.level_id)
        self.assertEqual(metrics["n_boxes"], 1)
        self.assertEqual(
            metrics["grid_size"], {"width": level.width, "height": level.height}
        )
        self.assertEqual(metrics["move_count"], 1)
        self.assertEqual(metrics["push_count"], 0)
        self.assertEqual(metrics["boxes_on_goals"], 0)
        self.assertFalse(metrics["deadlocked"])
        self.assertFalse(metrics["known_optimal"])
        self.assertEqual(len(metrics["history"]), 1)

    def test_default_instructions_override(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level), instructions="custom")
        self.assertEqual(adapter.default_instructions(), "custom")

    def test_format_state_uses_env_representation(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))
        text = adapter.format_state()
        self.assertIn("Board", text)
        self.assertIn("Boxes on goals", text)

    def test_get_state_snapshot(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))
        snapshot = adapter.get_state_snapshot()
        self.assertEqual(snapshot["player"], [1, 1])
        self.assertEqual(snapshot["n_boxes"], 1)

    def test_custom_tool_prefix_routing(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level), tool_prefix="custom")
        execution = adapter.execute_tool("custom_move", {"direction": "right"})
        self.assertTrue(execution.result["ok"])
        self.assertEqual(execution.meta["action_kind"], "move")

    def test_move_tool_missing_direction_is_illegal(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))
        execution = adapter.execute_tool("sokoban_move", {})
        self.assertFalse(execution.result["ok"])
        self.assertTrue(execution.meta["illegal_action"])
        self.assertFalse(execution.meta["counts_as_move"])

    def test_harness_does_not_count_undo_failure_as_illegal(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))
        provider = _ScriptedProvider(
            [
                ProviderResult(tool_calls=[ToolCall("sokoban_undo", {})], raw={}),
                ProviderResult(
                    tool_calls=[ToolCall("sokoban_move", {"direction": "right"})],
                    raw={},
                ),
            ]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=4)
        self.assertTrue(result.solved)
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.illegal_moves, 0)

    def test_harness_stops_immediately_on_terminal_deadlock_move(self) -> None:
        level = _level_from_xsb(
            """#####
#   #
# $ #
# @.#
#####
"""
        )
        env = SokobanEnv(level, detect_deadlocks=True, terminal_on_deadlock=True)
        self.assertFalse(env.is_deadlocked())
        adapter = SokobanGameAdapter(env)
        provider = _ScriptedProvider(
            [
                ProviderResult(
                    tool_calls=[ToolCall("sokoban_move", {"direction": "up"})],
                    raw={},
                ),
                ProviderResult(
                    tool_calls=[ToolCall("sokoban_get_state", {})],
                    raw={},
                ),
            ]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=5)
        self.assertFalse(result.solved)
        self.assertTrue(result.terminated_early)
        self.assertEqual(result.termination_reason, "deadlock_terminal")
        self.assertEqual(result.tool_calls, 1)

    def test_harness_stops_on_initial_deadlock_when_terminal_enabled(self) -> None:
        level = _level_from_xsb(
            """#####
#@  #
#$ .#
#####
"""
        )
        env = SokobanEnv(level, detect_deadlocks=True, terminal_on_deadlock=True)
        self.assertTrue(env.is_deadlocked())
        adapter = SokobanGameAdapter(env)
        provider = _ScriptedProvider(
            [ProviderResult(tool_calls=[ToolCall("sokoban_get_state", {})], raw={})]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=5)
        self.assertFalse(result.solved)
        self.assertTrue(result.terminated_early)
        self.assertEqual(result.termination_reason, "deadlock_terminal")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(provider.calls, 0)

    def test_harness_emits_action_state_sequence_for_multi_action_turn(self) -> None:
        level = _level_from_xsb(
            """#####
#@$.#
#####
"""
        )
        adapter = SokobanGameAdapter(SokobanEnv(level))
        provider = _ScriptedProvider(
            [
                ProviderResult(
                    tool_calls=[
                        ToolCall("sokoban_get_state", {}),
                        ToolCall("sokoban_get_state", {}),
                    ],
                    raw={},
                )
            ]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=1,
            max_tool_calls_per_turn=2,
        )
        action_states = [
            event for event in result.events if event.get("type") == "action_state"
        ]
        self.assertEqual(len(action_states), 2)
        self.assertEqual([event.get("action_index") for event in action_states], [0, 1])
        self.assertEqual([event.get("turn_index") for event in action_states], [0, 0])
        self.assertTrue(all("state" in event for event in action_states))
        self.assertTrue(all("state_text" in event for event in action_states))
        state_events = [
            event for event in result.events if event.get("type") == "state"
        ]
        self.assertEqual(len(state_events), 1)
        self.assertIn("trace_state_text", state_events[0])


if __name__ == "__main__":
    unittest.main()
