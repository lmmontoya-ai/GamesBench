# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

games-bench is a framework-agnostic benchmark for evaluating LLM tool-calling and RL agents on game environments. Currently implements Tower of Hanoi; designed for additional games via a registry pattern.

## Commands

```bash
# Setup
uv sync                      # core deps + dev tools (black, pre-commit)
uv sync --group llm          # add LLM provider deps (openai)
uv sync --group viz          # add visualization deps (pillow)

# Tests
uv run python -m unittest discover -s tests          # all tests
uv run python -m unittest tests.test_hanoi            # single module

# Formatting (enforced by pre-commit)
uv run black .

# CLI entry point
uv run games-bench <command>
# Commands: run, provider, render, review, rl, tool-calling, openai-tool-calling, manual-tool-loop

# Batch benchmark
uv run games-bench run --provider openrouter --config configs/hanoi.json

# Single provider episode
OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run games-bench provider --provider openrouter
```

## Architecture

The codebase has two strict layers (see AGENTS.md for full rules):

**Game engine** (`games_bench/games/`) — standalone, no benchmark/LLM imports:
- `games_bench/games/registry.py` — game registration (`GameSpec` + `register_game`)
- `games_bench/games/<game>/env.py` — environment (state, actions, step, rewards)
- `games_bench/games/<game>/prompts/` — prompt templates for LLM benchmarks
- `games_bench/games/<game>/vision.py` — image rendering for vision benchmarks
- `games_bench/games/<game>/render.py` — HTML/video rendering of recordings
- `games_bench/games/<game>/review.py` — review output generation

**Benchmark harness** (`games_bench/bench/` + `games_bench/llm/`) — depends on game engine:
- `games_bench/bench/cli.py` — CLI dispatcher (all subcommands)
- `games_bench/bench/registry.py` — benchmark registration (`BenchSpec`)
- `games_bench/bench/batch.py` — multi-episode batch runner
- `games_bench/llm/providers.py` — LLM providers (OpenRouter, OpenAI Responses, CLI, Codex)
- `games_bench/llm/harness.py` — tool-calling episode loop (`run_tool_calling_episode`)
- `games_bench/config.py` — config loading with env var expansion and game-level overrides

## Adding a new game

1. Create `games_bench/games/<game>/` with `env.py`, `__init__.py`, and optionally `prompts/`, `vision.py`, `render.py`, `review.py`
2. Register in `games_bench/games/registry.py` via `load_builtin_games()`
3. Add a benchmark spec in `games_bench/bench/registry.py` via `load_builtin_benchmarks()`
4. Game engine must NOT import from `games_bench/llm/` or `games_bench/bench/`

## Key conventions

- Python 3.11+, formatted with Black (24.1.1)
- Pre-commit hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, black
- LLM and viz deps are optional groups; keep them out of the game engine layer
- Secrets go in `.env` (gitignored); configs support `$ENV_VAR` expansion
- Artifacts (`runs/`, `renders/`, `reviews/`, `artifacts/`) are gitignored
- `examples/` contains legacy thin wrappers for backward compatibility
- Backward-compatible re-exports: `games_bench/hanoi.py` and `games_bench/vision.py`
