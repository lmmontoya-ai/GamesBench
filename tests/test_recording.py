from __future__ import annotations

import unittest
from typing import Any

from games_bench.llm.recording import build_recording


def _state(pegs: list[list[int]]) -> dict[str, Any]:
    n_disks = max((disk for peg in pegs for disk in peg), default=0)
    return {
        "n_disks": n_disks,
        "pegs": pegs,
        "disk_positions": [],
    }


def _base_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"type": "state_snapshot", "state": state},
        {"type": "state", "state": state, "state_text": "state"},
    ]


class TestRecording(unittest.TestCase):
    def test_counts_moves_for_move_tool(self) -> None:
        s0 = _state([[3, 2, 1], [], []])
        s1 = _state([[3, 2], [], [1]])
        events = _base_events(s0) + [
            {"type": "tool_call", "name": "hanoi_move", "arguments": {}},
            {"type": "tool_result", "result": {"ok": True, "state": s1}},
        ]
        recording = build_recording(events=events, metadata={})
        summary = recording["summary"]
        self.assertEqual(summary["total_tool_calls"], 1)
        self.assertEqual(summary["total_moves"], 1)
        self.assertEqual(summary["total_illegal_moves"], 0)

    def test_does_not_count_query_tools_as_moves(self) -> None:
        s0 = _state([[3, 2, 1], [], []])
        s1 = _state([[3, 2], [], [1]])
        events = _base_events(s0) + [
            {"type": "tool_call", "name": "hanoi_get_state", "arguments": {}},
            {"type": "tool_result", "result": {"ok": True, "state": s0}},
            {"type": "tool_call", "name": "hanoi_move", "arguments": {}},
            {"type": "tool_result", "result": {"ok": True, "state": s1}},
        ]
        recording = build_recording(events=events, metadata={})
        summary = recording["summary"]
        self.assertEqual(summary["total_tool_calls"], 2)
        self.assertEqual(summary["total_moves"], 1)
        self.assertEqual(summary["total_illegal_moves"], 0)

    def test_counts_failed_move_as_illegal(self) -> None:
        s0 = _state([[3, 2, 1], [], []])
        events = _base_events(s0) + [
            {"type": "tool_call", "name": "hanoi_move", "arguments": {}},
            {
                "type": "tool_result",
                "result": {"ok": False, "state": s0, "error": "illegal move"},
            },
        ]
        recording = build_recording(events=events, metadata={})
        summary = recording["summary"]
        self.assertEqual(summary["total_tool_calls"], 1)
        self.assertEqual(summary["total_moves"], 0)
        self.assertEqual(summary["total_illegal_moves"], 1)

    def test_step_uses_illegal_action_flag(self) -> None:
        s0 = _state([[3, 2, 1], [], []])
        events = _base_events(s0) + [
            {
                "type": "tool_call",
                "name": "hanoi_step",
                "arguments": {"action": [1, 2]},
            },
            {
                "type": "tool_result",
                "result": {
                    "ok": True,
                    "state": s0,
                    "info": {"illegal_action": True},
                },
            },
        ]
        recording = build_recording(events=events, metadata={})
        summary = recording["summary"]
        self.assertEqual(summary["total_tool_calls"], 1)
        self.assertEqual(summary["total_moves"], 0)
        self.assertEqual(summary["total_illegal_moves"], 1)

    def test_prefers_meta_for_action_counting(self) -> None:
        s0 = _state([[3, 2, 1], [], []])
        events = _base_events(s0) + [
            {"type": "tool_call", "name": "custom_tool", "arguments": {}},
            {
                "type": "tool_result",
                "result": {"ok": True, "state": s0},
                "meta": {"state_mutating": True, "illegal_action": False},
            },
            {"type": "tool_call", "name": "custom_tool", "arguments": {}},
            {
                "type": "tool_result",
                "result": {"ok": True, "state": s0},
                "meta": {"state_mutating": False, "illegal_action": True},
            },
        ]
        recording = build_recording(events=events, metadata={})
        summary = recording["summary"]
        self.assertEqual(summary["total_tool_calls"], 2)
        self.assertEqual(summary["total_moves"], 1)
        self.assertEqual(summary["total_illegal_moves"], 1)

    def test_counts_as_move_meta_controls_move_count(self) -> None:
        s0 = _state([[3, 2, 1], [], []])
        events = _base_events(s0) + [
            {"type": "tool_call", "name": "custom_tool", "arguments": {}},
            {
                "type": "tool_result",
                "result": {"ok": False, "state": s0, "error": "custom"},
                "meta": {
                    "state_mutating": False,
                    "illegal_action": False,
                    "counts_as_move": True,
                },
            },
        ]
        recording = build_recording(events=events, metadata={})
        summary = recording["summary"]
        self.assertEqual(summary["total_tool_calls"], 1)
        self.assertEqual(summary["total_moves"], 1)
        self.assertEqual(summary["total_illegal_moves"], 0)

    def test_non_illegal_failure_meta_not_counted_as_illegal(self) -> None:
        s0 = _state([[3, 2, 1], [], []])
        events = _base_events(s0) + [
            {"type": "tool_call", "name": "custom_tool", "arguments": {}},
            {
                "type": "tool_result",
                "result": {"ok": False, "state": s0, "error": "cannot undo"},
                "meta": {
                    "state_mutating": False,
                    "illegal_action": False,
                    "counts_as_move": False,
                },
            },
        ]
        recording = build_recording(events=events, metadata={})
        summary = recording["summary"]
        self.assertEqual(summary["total_tool_calls"], 1)
        self.assertEqual(summary["total_moves"], 0)
        self.assertEqual(summary["total_illegal_moves"], 0)


if __name__ == "__main__":
    unittest.main()
