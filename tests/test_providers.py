from __future__ import annotations

import builtins
import io
import types
import unittest
from unittest import mock

from games_bench.llm.providers import OpenAIResponsesProvider, OpenRouterProvider


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


if __name__ == "__main__":
    unittest.main()
