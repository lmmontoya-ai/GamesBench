from __future__ import annotations

import json
import unittest
from typing import Any

from games_bench.llm.game_adapter import ToolExecution
from games_bench.llm.harness import run_tool_calling_episode
from games_bench.llm.providers import ProviderResult, ToolCall


class DummyAdapter:
    def __init__(self) -> None:
        self._solved = False
        self._moves = 0
        self.executed: list[str] = []

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "dummy_move",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
            {
                "name": "dummy_noop",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
        ]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        del arguments
        self.executed.append(name)
        if name == "dummy_move":
            self._moves += 1
            self._solved = True
            return ToolExecution(
                result={"ok": True, "state": {"moves": self._moves}},
                meta={"state_mutating": True, "illegal_action": False},
            )
        return ToolExecution(
            result={"ok": True, "state": {"moves": self._moves}},
            meta={"state_mutating": False, "illegal_action": False},
        )

    def get_state_snapshot(self) -> dict[str, Any]:
        return {"moves": self._moves}

    def is_solved(self) -> bool:
        return self._solved

    def default_instructions(self) -> str:
        return "Use tools."

    def format_state(self) -> str:
        return '{"moves":0}'

    def episode_metrics(self) -> dict[str, Any]:
        return {"move_count": self._moves}


class MockProvider:
    supports_images = False

    def __init__(self, results: list[ProviderResult]) -> None:
        self._results = list(results)
        self.calls = 0
        self.conversations: list[list[dict[str, Any]] | None] = []

    def next_tool_calls(
        self,
        *,
        state_text: str,
        tool_schemas: list[dict[str, Any]],
        instructions: str,
        state_image: dict[str, Any] | None = None,
        conversation: list[dict[str, Any]] | None = None,
    ) -> ProviderResult:
        del state_text, tool_schemas, instructions, state_image
        self.calls += 1
        if conversation is None:
            self.conversations.append(None)
        else:
            self.conversations.append(json.loads(json.dumps(conversation)))
        if self._results:
            return self._results.pop(0)
        return ProviderResult(tool_calls=[], raw=None, error="No scripted response")


class TestHarness(unittest.TestCase):
    def test_runs_episode_with_adapter(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_move", {})], raw={})]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=5)
        self.assertTrue(result.solved)
        self.assertEqual(result.move_count, 1)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.illegal_moves, 0)
        self.assertEqual(adapter.executed, ["dummy_move"])

    def test_stateful_is_default_and_includes_history(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [
                ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={}),
                ProviderResult(tool_calls=[ToolCall("dummy_move", {})], raw={}),
            ]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=3)
        self.assertTrue(result.solved)
        self.assertEqual(provider.calls, 2)
        self.assertIsNotNone(provider.conversations[0])
        self.assertEqual(provider.conversations[0][0]["role"], "system")
        self.assertEqual(provider.conversations[0][1]["role"], "user")
        self.assertGreater(
            len(provider.conversations[1]), len(provider.conversations[0])
        )

    def test_stateless_mode_omits_conversation_history(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [
                ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={}),
                ProviderResult(tool_calls=[ToolCall("dummy_move", {})], raw={}),
            ]
        )
        result = run_tool_calling_episode(
            adapter, provider, max_turns=3, stateless=True
        )
        self.assertTrue(result.solved)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.conversations, [None, None])

    def test_rejects_disallowed_tool(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_move", {})], raw={})]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=1,
            allowed_tools=["dummy_noop"],
        )
        self.assertFalse(result.solved)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.illegal_moves, 1)
        self.assertEqual(adapter.executed, [])

    def test_stops_on_provider_error(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider([ProviderResult(tool_calls=[], raw=None, error="boom")])
        result = run_tool_calling_episode(adapter, provider, max_turns=3)
        self.assertFalse(result.solved)
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(result.illegal_moves, 0)

    def test_validates_image_capability(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_move", {})], raw={})]
        )
        with self.assertRaises(ValueError):
            run_tool_calling_episode(
                adapter,
                provider,
                state_image_renderer=lambda _adapter: {"mime_type": "image/png"},
            )

    def test_meta_illegal_action_overrides_result_ok_for_counting(self) -> None:
        adapter = DummyAdapter()

        def non_illegal_failure(
            _name: str, _arguments: dict[str, Any]
        ) -> ToolExecution:
            return ToolExecution(
                result={"ok": False, "state": {"moves": 0}, "error": "cannot undo"},
                meta={
                    "state_mutating": False,
                    "illegal_action": False,
                    "action_kind": "undo",
                    "counts_as_move": False,
                },
            )

        adapter.execute_tool = non_illegal_failure  # type: ignore[method-assign]
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={})]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=1)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.illegal_moves, 0)

    def test_stops_early_on_stagnation(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [
                ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={})
                for _ in range(8)
            ]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=10,
            stagnation_patience=2,
        )
        self.assertFalse(result.solved)
        self.assertTrue(result.terminated_early)
        self.assertEqual(result.termination_reason, "stagnation:2")
        self.assertEqual(result.tool_calls, 2)
        self.assertTrue(
            any(event.get("type") == "early_stop" for event in result.events)
        )

    def test_stops_early_on_deadlock_patience(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [
                ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={})
                for _ in range(8)
            ]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=10,
            deadlock_patience=2,
            deadlock_checker=lambda _adapter: True,
        )
        self.assertFalse(result.solved)
        self.assertTrue(result.terminated_early)
        self.assertEqual(result.termination_reason, "deadlock:2")
        self.assertEqual(result.tool_calls, 1)
        self.assertTrue(
            any(event.get("type") == "early_stop" for event in result.events)
        )

    def test_stops_early_on_adapter_requested_termination(self) -> None:
        adapter = DummyAdapter()

        def terminate_tool(_name: str, _arguments: dict[str, Any]) -> ToolExecution:
            return ToolExecution(
                result={"ok": True},
                meta={
                    "state_mutating": False,
                    "illegal_action": False,
                    "terminate_episode": True,
                    "termination_reason": "deadlock_terminal",
                },
            )

        adapter.execute_tool = terminate_tool  # type: ignore[method-assign]
        provider = MockProvider(
            [
                ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={}),
                ProviderResult(tool_calls=[ToolCall("dummy_move", {})], raw={}),
            ]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=5)
        self.assertFalse(result.solved)
        self.assertTrue(result.terminated_early)
        self.assertEqual(result.termination_reason, "deadlock_terminal")
        self.assertEqual(result.tool_calls, 1)
        self.assertTrue(
            any(event.get("type") == "early_stop" for event in result.events)
        )

    def test_stops_immediately_on_terminal_deadlock_check(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={})]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=5,
            deadlock_checker=lambda _adapter: True,
            deadlock_terminate_on_check=True,
        )
        self.assertFalse(result.solved)
        self.assertTrue(result.terminated_early)
        self.assertEqual(result.termination_reason, "deadlock_terminal")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(provider.calls, 0)

    def test_infers_terminal_deadlock_check_from_adapter_env(self) -> None:
        adapter = DummyAdapter()

        class _DeadlockEnv:
            detect_deadlocks = True
            terminal_on_deadlock = True

            def is_deadlocked(self) -> bool:
                return True

        adapter.env = _DeadlockEnv()  # type: ignore[attr-defined]
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={})]
        )
        result = run_tool_calling_episode(adapter, provider, max_turns=5)
        self.assertFalse(result.solved)
        self.assertTrue(result.terminated_early)
        self.assertEqual(result.termination_reason, "deadlock_terminal")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(provider.calls, 0)

    def test_explicitly_disables_inferred_deadlock_termination(self) -> None:
        adapter = DummyAdapter()

        class _DeadlockEnv:
            detect_deadlocks = True
            terminal_on_deadlock = True

            def is_deadlocked(self) -> bool:
                return True

        adapter.env = _DeadlockEnv()  # type: ignore[attr-defined]
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_noop", {})], raw={})]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=1,
            deadlock_terminate_on_check=False,
        )
        self.assertFalse(result.terminated_early)
        self.assertIsNone(result.termination_reason)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(provider.calls, 1)

    def test_executes_multiple_tool_calls_when_enabled(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [
                ProviderResult(
                    tool_calls=[
                        ToolCall("dummy_noop", {}),
                        ToolCall("dummy_move", {}),
                    ],
                    raw={},
                )
            ]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=2,
            max_tool_calls_per_turn=2,
        )
        self.assertTrue(result.solved)
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(adapter.executed, ["dummy_noop", "dummy_move"])

    def test_truncates_tool_calls_when_per_turn_limit_reached(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [
                ProviderResult(
                    tool_calls=[
                        ToolCall("dummy_noop", {}),
                        ToolCall("dummy_move", {}),
                    ],
                    raw={},
                )
            ]
        )
        result = run_tool_calling_episode(
            adapter,
            provider,
            max_turns=1,
            max_tool_calls_per_turn=1,
        )
        self.assertFalse(result.solved)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(adapter.executed, ["dummy_noop"])
        self.assertTrue(
            any(event.get("type") == "tool_calls_truncated" for event in result.events)
        )

    def test_rejects_invalid_max_tool_calls_per_turn(self) -> None:
        adapter = DummyAdapter()
        provider = MockProvider(
            [ProviderResult(tool_calls=[ToolCall("dummy_move", {})], raw={})]
        )
        with self.assertRaises(ValueError):
            run_tool_calling_episode(
                adapter,
                provider,
                max_turns=1,
                max_tool_calls_per_turn=0,
            )


if __name__ == "__main__":
    unittest.main()
