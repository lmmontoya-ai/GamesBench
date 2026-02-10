# AGENTS.md

This repo keeps game environments and benchmark harnesses in the same repository,
but they must remain **layered** so the game engine can be used standalone.

## Layer boundaries

- **Game engine (no benchmark dependencies):**
  - `games_bench/games/adapter.py` — `GameAdapter` protocol + `ToolExecution` (the contract)
  - `games_bench/games/registry.py` — game registration (`GameSpec`)
  - `games_bench/games/<game>/` — env, adapter, prompts, rendering, vision
  - `games_bench/hanoi.py`, `games_bench/vision.py` (backward-compat re-exports)
  - `apps/renderer-*` (game-specific interactive renderers)
- **LLM harness (game-agnostic, depends on game engine contract only):**
  - `games_bench/llm/harness.py` — episode loop, consumes `GameAdapter`
  - `games_bench/llm/providers.py` — provider adapters (OpenRouter, OpenAI, CLI, Codex)
  - `games_bench/llm/recording.py` — recording builder
  - `games_bench/llm/game_adapter.py` (backward-compat re-export of `games/adapter.py`)
- **Benchmark orchestration (depends on both layers above):**
  - `games_bench/bench/registry.py` — `BenchSpec` registration
  - `games_bench/bench/batch.py` — multi-game batch dispatcher
  - `games_bench/bench/common.py` — shared CLI argument definitions
  - `games_bench/bench/game_loader.py` — shared utilities for demo entrypoints
  - `games_bench/bench/<game>.py` — game-specific benchmark runner
  - `games_bench/bench/cli.py` — top-level CLI dispatcher
  - `configs/` (benchmark configs)
- **Artifacts (outputs only):** `runs/`, `renders/`, `reviews/`, `artifacts/`

## Expectations between layers

- Game engine must **not** import from `bench/` or `llm/`.
- `llm/` must **not** import from any specific game (`games_bench.games.hanoi`, etc.).
  It consumes games only through the `GameAdapter` protocol defined in `games/adapter.py`.
- `bench/` may import specific games only in game-specific runners (`bench/hanoi.py`)
  and the registry bootstrapper (`bench/registry.py`). Generic modules (`batch.py`,
  `common.py`, `game_loader.py`, demo entrypoints) must resolve games through registries.
- **Rendering belongs to the game engine**: rendering logic lives in
  `games_bench/games/<game>/render.py`, `vision.py`, or `apps/renderer-<game>`.
  Benchmark code orchestrates rendering, not implements it.
- Demo entrypoints (`provider.py`, `rl.py`, `tool_calling.py`, `manual_tool_loop.py`,
  `openai_tool_calling.py`) resolve games via `game_loader.py` and accept `--game`.

## Adding a new game

1. Create `games_bench/games/<game>/` with `env.py`, `adapter.py`, `__init__.py`.
2. The adapter must satisfy `games_bench.games.adapter.GameAdapter` (protocol).
3. Register the env in `games_bench/games/registry.py` → `load_builtin_games()`.
4. Create `games_bench/bench/<game>.py` with:
   - `run_batch(args, config)` — batch runner
   - `add_<game>_arguments(parser)` — game-specific CLI flags
   - `default_<game>_config()` — default config values
   - `build_<game>_adapter(env)` — adapter factory
5. Register the benchmark in `games_bench/bench/registry.py` → `load_builtin_benchmarks()`.
6. Add a config section under `"games"` in a config file.

## Dependency hygiene

- Base package has **zero external deps** (`pyproject.toml: dependencies = []`).
- `openai` is in `[project.optional-dependencies] llm`; providers import it lazily.
- `pillow` is in `[project.optional-dependencies] viz`; vision modules import it lazily.
- Missing-dep errors must include actionable install guidance
  (`pip install 'games-bench[llm]'` or `uv sync --group llm`).
- Secrets belong in `.env` (gitignored).

## Batch CLI contract

- Config-primary: game-specific params belong in config files, not CLI flags.
- Three modes: `games-bench run <game> [flags]` (subcommand), `games-bench run --config ...`
  (multi-game), `games-bench run --config ... --game <name>` (filtered).
- Config merge precedence: `BenchSpec.default_config()` < global config < per-game overrides.
- `--help` on `games-bench run` shows only common flags; `games-bench run <game> --help`
  shows game-specific flags. No flag pollution across games.

## Re-export shims (backward compatibility)

These files are thin re-exports. Do not add logic to them:
- `games_bench/hanoi.py` → `games_bench.games.hanoi.env`
- `games_bench/vision.py` → `games_bench.games.hanoi.vision`
- `games_bench/llm/game_adapter.py` → `games_bench.games.adapter`
- `games_bench/bench/hanoi_adapter.py` → `games_bench.games.hanoi.adapter`
- `games_bench/llm/prompting.py` — deprecated, empty

## Memory bank (living context)

Last updated: 2026-02-10

### Project goals

- Build a reproducible benchmark for **long-horizon planning** and **spatial reasoning**.
- Use diverse games where model behavior can be measured through trajectories, not only final answers.
- Keep the game engine standalone and reusable outside benchmark orchestration.

### Current benchmark status

- Canonical games:
  - `hanoi` (planning/search + long horizon)
  - `sokoban` (spatial reasoning + deadlock/irreversibility)
- Canonical suites:
  - `easy-v1` (small-model-friendly)
  - `standard-v1` (harder canonical benchmark)
- Interaction modes:
  - stateful is default
  - stateless via `--stateless`
  - spec naming is suffixed as `-stateful` / `-stateless`

### Operational maturity progress

Phase 1 (implemented):

- Run lineage/provenance:
  - `run_manifest.json` per run with git/platform/hash metadata.
- Generation/scoring separation:
  - `games-bench score --run-dir <run_dir>` for offline scoring.
  - `--no-score` for generation-only runs.
- Standardized taxonomy:
  - episodes include `outcome_code`, `failure_tags`, `taxonomy_version`.

Phase 2 (implemented):

- Shared bench executor:
  - centralized ordered episode commit + artifact writing in `games_bench/bench/executor.py`.
- Checkpoint/resume:
  - `execution_state.json` tracking completed episodes.
  - CLI flags: `--run-id`, `--resume`, `--strict-resume`, `--checkpoint-interval`.

### Current artifact contract

Runs under `artifacts/runs/...` include:

- `run_config.json`
- `run_manifest.json`
- `execution_state.json`
- `episodes.jsonl`
- `traces.jsonl`
- `summary.json` (unless `--no-score`)
- optional `recordings/` and `raw_generations.jsonl`

### Next priorities

- Phase 3: regression governance via `games-bench compare` (baseline vs candidate + thresholds + exit-code gating).
- Phase 4: additional hardening and CI checks around resume/recovery/compare workflows.

Reference implementation plan: `BENCH_MATURITY_PLAN.md`
