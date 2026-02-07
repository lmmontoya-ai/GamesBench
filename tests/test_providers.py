from __future__ import annotations

import builtins
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


if __name__ == "__main__":
    unittest.main()
