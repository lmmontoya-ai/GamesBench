"""LLM provider interfaces and tool-calling harnesses."""

from __future__ import annotations

from games_bench.games.adapter import GameAdapter, ToolExecution
from .harness import EpisodeResult, run_tool_calling_episode
from .recording import build_recording
from .providers import (
    CLIProvider,
    CodexAppServerProvider,
    CodexCLIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
    ToolCall,
)

__all__ = [
    "CLIProvider",
    "CodexAppServerProvider",
    "CodexCLIProvider",
    "EpisodeResult",
    "GameAdapter",
    "OpenAIResponsesProvider",
    "OpenRouterProvider",
    "ToolExecution",
    "ToolCall",
    "build_recording",
    "run_tool_calling_episode",
]
