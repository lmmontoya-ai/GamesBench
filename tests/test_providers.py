from __future__ import annotations

import builtins
import io
import json
import types
import unittest
from pathlib import Path
from unittest import mock

from games_bench.llm.providers import (
    CodexAppServerProvider,
    CodexCLIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
)


def _import_without_openai(
    name: str,
    globals=None,
    locals=None,
    fromlist=(),
    level: int = 0,
):
    if name == "openai":
        raise ImportError("No module named 'openai'")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


_ORIGINAL_IMPORT = builtins.__import__


class TestProviders(unittest.TestCase):
    def test_openrouter_missing_dependency_message_has_install_help(self) -> None:
        provider = OpenRouterProvider(model="openai/gpt-4.1-mini", api_key="test")
        with mock.patch("builtins.__import__", side_effect=_import_without_openai):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )
        self.assertIsNotNone(result.error)
        self.assertIn("games-bench[llm]", result.error or "")
        self.assertIn("uv sync --group llm", result.error or "")

    def test_openai_missing_dependency_message_has_install_help(self) -> None:
        provider = OpenAIResponsesProvider(model="gpt-4.1-mini", api_key="test")
        with mock.patch("builtins.__import__", side_effect=_import_without_openai):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )
        self.assertIsNotNone(result.error)
        self.assertIn("games-bench[llm]", result.error or "")
        self.assertIn("uv sync --group llm", result.error or "")

    def test_openrouter_stream_debug_collects_streamed_tool_call(self) -> None:
        create_calls: list[dict[str, object]] = []

        class FakeChunk:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def model_dump(self) -> dict[str, object]:
                return self._payload

        class FakeCompletions:
            def create(self, **kwargs):
                create_calls.append(dict(kwargs))
                return iter(
                    [
                        FakeChunk(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "function": {"name": "hanoi_move"},
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ]
                            }
                        ),
                        FakeChunk(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "function": {
                                                        "arguments": '{"from_peg":0,"to_peg":2}'
                                                    },
                                                }
                                            ]
                                        },
                                        "finish_reason": "tool_calls",
                                    }
                                ],
                                "usage": {"total_tokens": 10},
                            }
                        ),
                    ]
                )

        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenRouterProvider(
            model="openai/gpt-4.1-mini",
            api_key="test",
            stream_debug=True,
        )
        stderr = io.StringIO()
        with (
            mock.patch("builtins.__import__", side_effect=import_with_fake_openai),
            mock.patch("sys.stderr", stderr),
        ):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertEqual(len(create_calls), 1)
        self.assertTrue(bool(create_calls[0].get("stream")))
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "hanoi_move")
        self.assertEqual(result.tool_calls[0].arguments, {"from_peg": 0, "to_peg": 2})
        self.assertIn("[openrouter stream-debug]", stderr.getvalue())

    def test_openai_responses_parses_typed_function_call_objects(self) -> None:
        class FakeFunctionCall:
            type = "function_call"
            name = "hanoi_move"
            arguments = '{"from_peg":0,"to_peg":2}'

        class FakeResponse:
            def __init__(self) -> None:
                self.output = [FakeFunctionCall()]
                self.usage = {"total_tokens": 11}

            def model_dump(self) -> dict[str, object]:
                return {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "hanoi_move",
                            "arguments": '{"from_peg":0,"to_peg":2}',
                        }
                    ],
                    "usage": {"total_tokens": 11},
                }

        class FakeResponses:
            def create(self, **_kwargs):
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                self.responses = FakeResponses()

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenAIResponsesProvider(model="gpt-4.1-mini", api_key="test")
        with mock.patch("builtins.__import__", side_effect=import_with_fake_openai):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "hanoi_move")
        self.assertEqual(result.tool_calls[0].arguments, {"from_peg": 0, "to_peg": 2})

    def test_openrouter_embedded_5xx_retry_exhaustion_returns_provider_error(
        self,
    ) -> None:
        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace())

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenRouterProvider(
            model="openai/gpt-4.1-mini",
            api_key="test",
            max_retries=2,
            retry_backoff_s=0.0,
        )
        embedded_error_payload = {
            "choices": [
                {"message": {}, "error": {"code": 503, "message": "upstream down"}}
            ],
            "usage": {"total_tokens": 1},
        }

        with (
            mock.patch("builtins.__import__", side_effect=import_with_fake_openai),
            mock.patch.object(
                provider,
                "_completion_payload",
                side_effect=[
                    embedded_error_payload,
                    RuntimeError("retry attempt one failed"),
                    RuntimeError("retry attempt two failed"),
                ],
            ),
        ):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertEqual(result.tool_calls, [])
        self.assertIsNotNone(result.error)
        self.assertIn("Provider error after retries", result.error or "")
        self.assertNotIn("No tool calls returned.", result.error or "")

    def test_openrouter_forwards_parallel_tool_calls_flag(self) -> None:
        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                pass

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenRouterProvider(
            model="openai/gpt-4.1-mini",
            api_key="test",
            parallel_tool_calls=False,
        )
        fake_payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "hanoi_move",
                                    "arguments": '{"from_peg":0,"to_peg":2}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        with (
            mock.patch("builtins.__import__", side_effect=import_with_fake_openai),
            mock.patch.object(
                provider, "_completion_payload", return_value=fake_payload
            ) as patched_completion,
        ):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertIsNone(result.error)
        call_kwargs = patched_completion.call_args[0][1]
        self.assertIn("parallel_tool_calls", call_kwargs)
        self.assertFalse(bool(call_kwargs["parallel_tool_calls"]))

    def test_openrouter_forwards_provider_sort_flag(self) -> None:
        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                pass

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenRouterProvider(
            model="openai/gpt-4.1-mini",
            api_key="test",
            provider_sort="price",
        )
        fake_payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "hanoi_move",
                                    "arguments": '{"from_peg":0,"to_peg":2}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        with (
            mock.patch("builtins.__import__", side_effect=import_with_fake_openai),
            mock.patch.object(
                provider, "_completion_payload", return_value=fake_payload
            ) as patched_completion,
        ):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertIsNone(result.error)
        call_kwargs = patched_completion.call_args[0][1]
        self.assertIn("provider", call_kwargs)
        self.assertEqual(call_kwargs["provider"], {"sort": "price"})

    def test_openrouter_stream_debug_moves_provider_sort_to_extra_body(self) -> None:
        create_calls: list[dict[str, object]] = []

        class FakeChunk:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def model_dump(self) -> dict[str, object]:
                return self._payload

        class FakeCompletions:
            def create(self, **kwargs):
                create_calls.append(dict(kwargs))
                return iter(
                    [
                        FakeChunk(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "function": {
                                                        "name": "hanoi_move",
                                                        "arguments": '{"from_peg":0,"to_peg":2}',
                                                    },
                                                }
                                            ]
                                        },
                                        "finish_reason": "tool_calls",
                                    }
                                ],
                                "usage": {"total_tokens": 1},
                            }
                        )
                    ]
                )

        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenRouterProvider(
            model="openai/gpt-4.1-mini",
            api_key="test",
            stream_debug=True,
            provider_sort="price",
        )
        with mock.patch("builtins.__import__", side_effect=import_with_fake_openai):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertIsNone(result.error)
        self.assertEqual(len(create_calls), 1)
        self.assertNotIn("provider", create_calls[0])
        self.assertIn("extra_body", create_calls[0])
        self.assertEqual(
            create_calls[0]["extra_body"],
            {"provider": {"sort": "price"}},
        )

    def test_openai_responses_forwards_parallel_tool_calls_flag(self) -> None:
        create_calls: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self) -> None:
                self.output = [
                    {
                        "type": "function_call",
                        "name": "hanoi_move",
                        "arguments": '{"from_peg":0,"to_peg":2}',
                    }
                ]
                self.usage = {"total_tokens": 1}

            def model_dump(self) -> dict[str, object]:
                return {"output": self.output, "usage": self.usage}

        class FakeResponses:
            def create(self, **kwargs):
                create_calls.append(dict(kwargs))
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                self.responses = FakeResponses()

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenAIResponsesProvider(
            model="gpt-4.1-mini",
            api_key="test",
            parallel_tool_calls=True,
        )
        with mock.patch("builtins.__import__", side_effect=import_with_fake_openai):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertIsNone(result.error)
        self.assertEqual(len(create_calls), 1)
        self.assertIn("parallel_tool_calls", create_calls[0])
        self.assertTrue(bool(create_calls[0]["parallel_tool_calls"]))

    def test_openai_responses_provider_error_is_returned_not_raised(self) -> None:
        class FakeResponses:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                raise RuntimeError("transient network failure")

        fake_responses = FakeResponses()

        class FakeOpenAI:
            def __init__(self, **_kwargs) -> None:
                self.responses = fake_responses

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        def import_with_fake_openai(
            name: str,
            globals=None,
            locals=None,
            fromlist=(),
            level: int = 0,
        ):
            if name == "openai":
                return fake_openai_module
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        provider = OpenAIResponsesProvider(
            model="gpt-4.1-mini",
            api_key="test",
            max_retries=1,
            retry_backoff_s=0.0,
        )
        with mock.patch("builtins.__import__", side_effect=import_with_fake_openai):
            result = provider.next_tool_calls(
                state_text="{}",
                tool_schemas=[],
                instructions="Use tools.",
            )

        self.assertEqual(result.tool_calls, [])
        self.assertIsNotNone(result.error)
        self.assertIn("Provider error:", result.error or "")
        self.assertEqual(fake_responses.calls, 2)

    def test_codex_app_server_provider_captures_single_tool_call(self) -> None:
        requests = [
            {
                "tool": "emit_tool_call",
                "arguments": {
                    "name": "hanoi_move",
                    "arguments": {"from_peg": 0, "to_peg": 2},
                },
            }
        ]

        class FakeSession:
            def __init__(self) -> None:
                self.turn_calls: list[dict[str, object]] = []

            def run_turn(self, **kwargs):  # noqa: ANN003
                self.turn_calls.append(dict(kwargs))
                on_tool_call = kwargs["on_tool_call"]
                for req in requests:
                    on_tool_call(req)
                return types.SimpleNamespace(
                    thread_id="thread-1",
                    turn_id="turn-1",
                    status="completed",
                    usage={"total_tokens": 12},
                    thread_usage=None,
                    start_result={"turnId": "turn-1"},
                    completion={"status": "completed", "usage": {"total_tokens": 12}},
                    notifications=[],
                    server_requests=[],
                )

            def close(self) -> None:
                return None

        fake_session = FakeSession()
        provider = CodexAppServerProvider(
            model="gpt-5.3-codex",
            max_tool_calls_per_turn=1,
            session_factory=lambda: fake_session,
        )
        result = provider.next_tool_calls(
            state_text="{}",
            tool_schemas=[
                {
                    "name": "hanoi_move",
                    "description": "Move disk",
                    "parameters": {},
                }
            ],
            instructions="Use tools.",
        )

        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "hanoi_move")
        self.assertEqual(result.tool_calls[0].arguments, {"from_peg": 0, "to_peg": 2})
        self.assertEqual(result.usage, {"total_tokens": 12})
        self.assertEqual(len(fake_session.turn_calls), 1)
        self.assertEqual(fake_session.turn_calls[0]["model"], "gpt-5.3-codex")

    def test_codex_app_server_provider_enforces_budget(self) -> None:
        requests = [
            {
                "tool": "emit_tool_call",
                "arguments": {
                    "name": "hanoi_move",
                    "arguments": {"from_peg": 0, "to_peg": 1},
                },
            },
            {
                "tool": "emit_tool_call",
                "arguments": {
                    "name": "hanoi_move",
                    "arguments": {"from_peg": 1, "to_peg": 2},
                },
            },
            {
                "tool": "emit_tool_call",
                "arguments": {
                    "name": "hanoi_move",
                    "arguments": {"from_peg": 2, "to_peg": 0},
                },
            },
        ]

        class FakeSession:
            def run_turn(self, **kwargs):  # noqa: ANN003
                on_tool_call = kwargs["on_tool_call"]
                for req in requests:
                    on_tool_call(req)
                return types.SimpleNamespace(
                    thread_id="thread-1",
                    turn_id="turn-1",
                    status="completed",
                    usage=None,
                    thread_usage=None,
                    start_result={"turnId": "turn-1"},
                    completion={"status": "completed"},
                    notifications=[],
                    server_requests=[],
                )

            def close(self) -> None:
                return None

        provider = CodexAppServerProvider(
            max_tool_calls_per_turn=2,
            session_factory=lambda: FakeSession(),
        )
        result = provider.next_tool_calls(
            state_text="{}",
            tool_schemas=[{"name": "hanoi_move", "parameters": {}}],
            instructions="Use tools.",
        )

        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].arguments, {"from_peg": 0, "to_peg": 1})
        self.assertEqual(result.tool_calls[1].arguments, {"from_peg": 1, "to_peg": 2})
        self.assertIsInstance(result.raw, dict)
        self.assertIn("tool_errors", result.raw)
        self.assertTrue(
            any("budget exceeded" in msg for msg in result.raw["tool_errors"])
        )

    def test_codex_app_server_provider_rejects_invalid_tool_name(self) -> None:
        request = {
            "tool": "emit_tool_call",
            "arguments": {
                "name": "not_allowed_tool",
                "arguments": {"x": 1},
            },
        }

        class FakeSession:
            def run_turn(self, **kwargs):  # noqa: ANN003
                kwargs["on_tool_call"](request)
                return types.SimpleNamespace(
                    thread_id="thread-1",
                    turn_id="turn-1",
                    status="completed",
                    usage=None,
                    thread_usage=None,
                    start_result={"turnId": "turn-1"},
                    completion={"status": "completed"},
                    notifications=[],
                    server_requests=[],
                )

            def close(self) -> None:
                return None

        provider = CodexAppServerProvider(
            max_tool_calls_per_turn=1,
            session_factory=lambda: FakeSession(),
        )
        result = provider.next_tool_calls(
            state_text="{}",
            tool_schemas=[{"name": "hanoi_move", "parameters": {}}],
            instructions="Use tools.",
        )

        self.assertEqual(result.tool_calls, [])
        self.assertIsNotNone(result.error)
        self.assertIn("No valid tool calls emitted", result.error or "")

    def test_codex_app_server_provider_reports_non_success_completion(self) -> None:
        class FakeSession:
            def run_turn(self, **kwargs):  # noqa: ANN003
                return types.SimpleNamespace(
                    thread_id="thread-1",
                    turn_id="turn-1",
                    status="failed",
                    usage=None,
                    thread_usage=None,
                    start_result={"turnId": "turn-1"},
                    completion={"status": "failed"},
                    notifications=[],
                    server_requests=[],
                )

            def close(self) -> None:
                return None

        provider = CodexAppServerProvider(
            max_tool_calls_per_turn=1,
            session_factory=lambda: FakeSession(),
        )
        result = provider.next_tool_calls(
            state_text="{}",
            tool_schemas=[{"name": "hanoi_move", "parameters": {}}],
            instructions="Use tools.",
        )

        self.assertEqual(result.tool_calls, [])
        self.assertIsNotNone(result.error)
        self.assertIn("did not complete successfully", result.error or "")

    def test_codex_app_server_provider_surfaces_session_errors(self) -> None:
        class FakeSession:
            def run_turn(self, **kwargs):  # noqa: ANN003
                raise RuntimeError("timed out")

            def close(self) -> None:
                return None

        provider = CodexAppServerProvider(session_factory=lambda: FakeSession())
        result = provider.next_tool_calls(
            state_text="{}",
            tool_schemas=[{"name": "hanoi_move", "parameters": {}}],
            instructions="Use tools.",
        )

        self.assertEqual(result.tool_calls, [])
        self.assertIsNotNone(result.error)
        self.assertIn("Codex provider error", result.error or "")

    def test_codex_exec_schema_is_strict_and_compatible(self) -> None:
        provider = CodexCLIProvider()
        schema = json.loads(Path(provider._schema_path).read_text())
        self.assertFalse(bool(schema.get("additionalProperties", True)))
        self.assertEqual(schema["properties"]["arguments"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
