from __future__ import annotations

import json
import unittest

from games_bench.bench import hanoi as hanoi_bench
from games_bench.bench import sokoban as sokoban_bench


class TestRawGenerations(unittest.TestCase):
    def test_hanoi_raw_generations_keep_multi_action_turn_sequence(self) -> None:
        events = [
            {
                "type": "state_snapshot",
                "state": {"pegs": [[2, 1], [], []]},
                "turn_index": 0,
            },
            {"type": "state", "state_text": "state-0", "turn_index": 0},
            {
                "type": "state_image",
                "meta": {"mime_type": "image/png", "width": 64, "height": 64},
                "turn_index": 0,
            },
            {
                "type": "provider_result",
                "error": None,
                "usage": {"total_tokens": 3},
                "turn_index": 0,
            },
            {
                "type": "tool_calls_truncated",
                "requested": 3,
                "executed": 2,
                "max_tool_calls_per_turn": 2,
                "turn_index": 0,
            },
            {
                "type": "tool_call",
                "name": "hanoi_move",
                "arguments": {"from_peg": 0, "to_peg": 1},
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "tool_result",
                "result": {"ok": True, "state": {"pegs": [[2], [1], []]}},
                "meta": {"state_mutating": True},
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "action_state",
                "state": {"pegs": [[2], [1], []]},
                "state_text": "state-1",
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "action_state_image",
                "meta": {"mime_type": "image/png", "width": 64, "height": 64},
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "tool_call",
                "name": "hanoi_move",
                "arguments": {"from_peg": 0, "to_peg": 2},
                "turn_index": 0,
                "action_index": 1,
            },
            {
                "type": "tool_result",
                "result": {"ok": True, "state": {"pegs": [[], [1], [2]]}},
                "meta": {"state_mutating": True},
                "turn_index": 0,
                "action_index": 1,
            },
            {
                "type": "action_state",
                "state": {"pegs": [[], [1], [2]]},
                "state_text": "state-2",
                "turn_index": 0,
                "action_index": 1,
            },
        ]
        lines = hanoi_bench._raw_lines_for_events(
            events=events,
            episode_id=7,
            variant_id="case",
            instructions="Solve",
            tool_schemas_payload=[{"name": "hanoi_move"}],
            state_format="both",
            image_config={"image_size": "64x64"},
        )
        self.assertEqual(len(lines), 1)
        row = json.loads(lines[0])
        self.assertEqual(len(row["actions"]), 2)
        self.assertEqual(row["actions"][0]["tool_call"]["name"], "hanoi_move")
        self.assertEqual(row["actions"][1]["tool_call"]["name"], "hanoi_move")
        self.assertEqual(row["actions"][0]["state_after_text"], "state-1")
        self.assertEqual(row["actions"][1]["state_after_text"], "state-2")
        self.assertIn("state_after_image", row["actions"][0])
        self.assertNotIn("tool_call", row)
        self.assertIn("tool_calls_truncated", row)
        self.assertEqual(row["tool_calls_truncated"]["requested"], 3)

    def test_sokoban_raw_generations_keep_multi_action_turn_sequence(self) -> None:
        events = [
            {"type": "state_snapshot", "state": {"board": "a"}, "turn_index": 0},
            {"type": "state", "state_text": "board-0", "turn_index": 0},
            {"type": "provider_result", "error": None, "turn_index": 0},
            {
                "type": "tool_call",
                "name": "sokoban_get_state",
                "arguments": {},
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "tool_result",
                "result": {"ok": True, "state": {"board": "a"}},
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "action_state",
                "state": {"board": "a"},
                "state_text": "board-0",
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "tool_call",
                "name": "sokoban_move",
                "arguments": {"direction": "right"},
                "turn_index": 0,
                "action_index": 1,
            },
            {
                "type": "tool_result",
                "result": {"ok": True, "state": {"board": "b"}},
                "meta": {"state_mutating": True},
                "turn_index": 0,
                "action_index": 1,
            },
            {
                "type": "action_state",
                "state": {"board": "b"},
                "state_text": "board-1",
                "turn_index": 0,
                "action_index": 1,
            },
        ]
        lines = sokoban_bench._raw_lines_for_events(
            events=events,
            episode_id=3,
            variant_id="level",
            instructions="Solve",
            tool_schemas_payload=[{"name": "sokoban_move"}],
            state_format="text",
            image_config={"tile": 24},
        )
        self.assertEqual(len(lines), 1)
        row = json.loads(lines[0])
        self.assertEqual(len(row["actions"]), 2)
        self.assertEqual(row["actions"][0]["tool_call"]["name"], "sokoban_get_state")
        self.assertEqual(row["actions"][1]["tool_call"]["name"], "sokoban_move")
        self.assertEqual(row["actions"][1]["state_after_text"], "board-1")
        self.assertNotIn("tool_call", row)

    def test_single_action_row_keeps_legacy_tool_fields(self) -> None:
        events = [
            {
                "type": "state_snapshot",
                "state": {"pegs": [[1], [], []]},
                "turn_index": 0,
            },
            {"type": "state", "state_text": "state-0", "turn_index": 0},
            {"type": "provider_result", "error": None, "turn_index": 0},
            {
                "type": "tool_call",
                "name": "hanoi_move",
                "arguments": {"from_peg": 0, "to_peg": 2},
                "turn_index": 0,
                "action_index": 0,
            },
            {
                "type": "tool_result",
                "result": {"ok": True, "state": {"pegs": [[], [], [1]]}},
                "turn_index": 0,
                "action_index": 0,
            },
        ]
        lines = hanoi_bench._raw_lines_for_events(
            events=events,
            episode_id=1,
            variant_id="single",
            instructions="Solve",
            tool_schemas_payload=[{"name": "hanoi_move"}],
            state_format="text",
            image_config={},
        )
        self.assertEqual(len(lines), 1)
        row = json.loads(lines[0])
        self.assertEqual(len(row["actions"]), 1)
        self.assertIn("tool_call", row)
        self.assertIn("tool_result", row)


if __name__ == "__main__":
    unittest.main()
