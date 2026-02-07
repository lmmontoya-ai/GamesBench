# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

games-bench is a framework-agnostic benchmark for evaluating LLM tool-calling and RL agents on game environments. Currently implements Tower of Hanoi; designed for additional games via a registry pattern and `GameAdapter` protocol.

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

# Batch benchmark (three modes)
uv run games-bench run hanoi --provider openrouter --n-disks 3,4      # single-game subcommand
uv run games-bench run --provider openrouter --config configs/hanoi.json  # config-driven
uv run games-bench run --provider openrouter --config configs/hanoi.json --game hanoi  # config + filter

# Single provider episode
OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run games-bench provider --provider openrouter
```

## Architecture

Three strict layers (see AGENTS.md for full rules):

**Game engine** (`games_bench/games/`) — standalone, zero imports from `bench/` or `llm/`:
- `games_bench/games/adapter.py` — `GameAdapter` protocol and `ToolExecution` dataclass (the contract between game and harness)
- `games_bench/games/registry.py` — game registration (`GameSpec`)
- `games_bench/games/<game>/env.py` — environment (state, actions, step, rewards)
- `games_bench/games/<game>/adapter.py` — `GameAdapter` implementation (tool dispatch, metrics, state formatting)
- `games_bench/games/<game>/prompts/` — prompt templates
- `games_bench/games/<game>/vision.py` — image rendering
- `games_bench/games/<game>/render.py` — HTML/video rendering of recordings
- `games_bench/games/<game>/review.py` — review output generation

**LLM harness** (`games_bench/llm/`) — game-agnostic, zero game-specific imports:
- `games_bench/llm/harness.py` — tool-calling episode loop; accepts `GameAdapter`, returns `EpisodeResult`
- `games_bench/llm/providers.py` — LLM providers (OpenRouter, OpenAI Responses, CLI, Codex)
- `games_bench/llm/recording.py` — recording builder; classifies moves via `ToolExecution.meta`

**Benchmark orchestration** (`games_bench/bench/`) — wires games to harness via registries:
- `games_bench/bench/cli.py` — CLI dispatcher (all subcommands)
- `games_bench/bench/registry.py` — benchmark registration (`BenchSpec` with `batch_runner`, `add_arguments`, `default_config`, `adapter_factory`, `render_main`, `review_main`)
- `games_bench/bench/batch.py` — multi-game batch runner (config-primary, subcommand convenience)
- `games_bench/bench/common.py` — shared CLI argument definitions
- `games_bench/bench/game_loader.py` — shared utilities for demo entrypoints (resolves game/adapter via registry)
- `games_bench/bench/<game>.py` — game-specific benchmark runner (e.g. `hanoi.py`)
- `games_bench/config.py` — config loading with env var expansion and game-level overrides

## Adding a new game

1. Create `games_bench/games/<game>/` with at minimum:
   - `env.py` — environment class
   - `adapter.py` — `GameAdapter` implementation wrapping the env
   - `__init__.py`
2. Optionally add `prompts/`, `vision.py`, `render.py`, `review.py`
3. Register the env in `games_bench/games/registry.py` → `load_builtin_games()`
4. Create `games_bench/bench/<game>.py` with `run_batch()`, `add_<game>_arguments()`, `default_<game>_config()`, `build_<game>_adapter()`
5. Register the benchmark in `games_bench/bench/registry.py` → `load_builtin_benchmarks()`
6. Add a config section under `"games"` in `configs/`
7. Game engine must NOT import from `llm/` or `bench/`; the adapter protocol (`games/adapter.py`) is the contract boundary

## Key conventions

- Python 3.11+, formatted with Black (24.1.1)
- Pre-commit hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, black
- Base package has zero external deps; `openai` is in `[project.optional-dependencies] llm`, `pillow` in `viz`
- LLM providers lazily import `openai` inside methods; missing-dep errors must include `pip install 'games-bench[llm]'` guidance
- Batch config merge precedence: `BenchSpec.default_config()` < global config < per-game config overrides
- Demo entrypoints (`provider.py`, `rl.py`, `tool_calling.py`, etc.) resolve games via registry/`game_loader.py`, not direct imports
- Secrets go in `.env` (gitignored); configs support `$ENV_VAR` expansion
- Artifacts (`runs/`, `renders/`, `reviews/`, `artifacts/`) are gitignored
- `examples/` contains legacy thin wrappers
- Re-export shims for backward compat: `games_bench/hanoi.py`, `games_bench/vision.py`, `llm/game_adapter.py`, `bench/hanoi_adapter.py`
