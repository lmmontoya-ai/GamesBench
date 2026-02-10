# Architecture

## Repository Objective

`games-bench` is a reproducible benchmark stack for evaluating long-horizon planning and spatial reasoning in tool-calling agents.

The repository intentionally co-locates:

- reusable game engines (`hanoi`, `sokoban`)
- a game-agnostic LLM tool-calling harness
- benchmark orchestration, scoring, and governance workflows

while preserving strict layering so the game engine can be used standalone.

## Layered Structure

### 1) Game Engine (foundational, standalone)

Core responsibility: define environments, tool schemas, prompt formatting, rendering, and game-specific review assets without benchmark or provider dependencies.

Primary modules:

- `games_bench/games/adapter.py` (`GameAdapter`, `ToolExecution`)
- `games_bench/games/registry.py` (`GameSpec`, built-in game registration)
- `games_bench/games/hanoi/*`
- `games_bench/games/sokoban/*`
- `games_bench/games/vision_types.py`
- `apps/renderer-hanoi/` (interactive renderer app)

Key pattern:

- Each game provides:
  - `env.py`: state machine + validation + tool schemas + toolbox wrappers
  - `adapter.py`: adapter that satisfies `GameAdapter`
  - `prompts/`: text templates and formatting helpers
  - `vision.py`: image rendering (lazy Pillow import)
  - `render.py`, `review.py`: run-artifact visualization/review outputs

### 2) LLM Harness (game-agnostic)

Core responsibility: run one tool-calling episode over any `GameAdapter`.

Primary modules:

- `games_bench/llm/harness.py`: turn loop, event stream, early-stop logic, usage/cost aggregation
- `games_bench/llm/providers.py`: provider adapters (`openrouter`, `openai`, `cli`, `codex`)
- `games_bench/llm/recording.py`: derive recording timeline from episode events

Constraints:

- No imports from specific games (`games_bench.games.hanoi`, etc.)
- Depends only on adapter protocol and generic tool schema payloads

### 3) Benchmark Orchestration (composition/orchestration layer)

Core responsibility: convert benchmark specs/configs into many episode jobs, persist artifacts, score outcomes, and compare regressions.

Primary modules:

- `games_bench/bench/cli.py`: top-level command dispatcher
- `games_bench/bench/batch.py`: `run` command (single-game and multi-game config modes)
- `games_bench/bench/registry.py`: `BenchSpec` registry
- `games_bench/bench/suites.py`: named suite registry (`easy-v1`, `standard-v1`)
- `games_bench/bench/hanoi.py`, `games_bench/bench/sokoban.py`: game-specific batch runners
- `games_bench/bench/executor.py`: shared ordered commit + checkpoint/resume executor
- `games_bench/bench/scoring.py`: offline scoring from artifacts
- `games_bench/bench/compare.py`: baseline/candidate governance
- `games_bench/bench/game_loader.py`: demo entrypoint game resolution

### 4) Artifacts (output-only)

Output roots:

- `artifacts/runs/`
- `artifacts/renders/`
- `artifacts/reviews/`
- legacy `runs/`, `renders/`, `reviews/`

Typical run payload:

- `run_config.json`
- `run_manifest.json`
- `execution_state.json`
- `episodes.jsonl`
- `traces.jsonl`
- `summary.json` (if scoring enabled)
- optional `recordings/`, `raw_generations.jsonl`

## Dependency Direction

Allowed dependency flow:

1. `games` -> standard library only (plus lazy optional deps for viz where needed)
2. `llm` -> `games.adapter` contract only
3. `bench` -> `llm` + `games` registries + game-specific bench modules

Disallowed:

- `games` importing from `bench` or `llm`
- generic `bench` modules importing specific games directly (except designated game-specific runners and registry bootstrap)

## Core Contracts

### `GameAdapter` Protocol

`games_bench/games/adapter.py` defines the stable integration contract consumed by harness and benchmark code:

- `tool_schemas()`
- `execute_tool(name, arguments) -> ToolExecution`
- `get_state_snapshot()`
- `is_solved()`
- `default_instructions()`
- `format_state()`
- `episode_metrics()`

`ToolExecution.meta` is critical for generic accounting (`illegal_action`, `counts_as_move`, `terminate_episode`, etc.).

### Registries

The architecture uses lightweight registries for decoupling:

- `GameSpec` (`games/registry.py`): environment factory registration
- `BenchSpec` (`bench/registry.py`): runner + scoring + render/review registration
- `SuiteSpec` (`bench/suites.py`): named config factory registration

This supports game resolution by name in generic orchestration and demo entrypoints.

## Runtime Flow

### `games-bench run ...` (batch path)

1. `bench/cli.py` dispatches command -> `bench/batch.py`.
2. `batch.py` resolves suite + config overlays, normalizes per-game config, and selects games.
3. For each selected game, corresponding `BenchSpec.batch_runner` is invoked (`bench/hanoi.py` or `bench/sokoban.py`).
4. Game runner expands config into episode jobs (deterministic `episode_id`, `variant_id` composition).
5. Shared executor (`bench/executor.py`) runs jobs (optionally parallel), enforces ordered episode commits, writes JSONL artifacts, and persists checkpoints.
6. Inline scoring (unless `--no-score`) writes `summary.json` via `build_summary_document`.

### Episode loop (within each job)

1. Build environment + adapter.
2. Build provider instance (thread-local + optional in-flight semaphore throttling).
3. Run `run_tool_calling_episode(...)`.
4. Convert result to normalized episode record + taxonomy fields.
5. Optionally build recording and raw generation lines.
6. Commit through shared executor.

### Offline scoring and compare

- `games-bench score --run-dir ...` recomputes `summary.json` from artifacts.
- `games-bench compare --baseline ... --candidate ...` computes pairwise regression report with optional threshold gates and CI-friendly exit code behavior.

## Data and Reliability Design

Implemented reliability controls:

- checkpoint/resume with job-plan hash (`execution_state.json`)
- strict resume validation (`--strict-resume`)
- lineage/provenance manifest (`run_manifest.json`)
- ordered episode commit for deterministic JSONL layout even under parallel execution
- taxonomy normalization (`outcome_code`, `failure_tags`, `taxonomy_version`)

## Configuration Model

Config is JSON and merge-driven:

- precedence: `BenchSpec.default_config()` < global config < per-game config
- modes:
  - `games-bench run <game> ...`
  - `games-bench run --config <file>`
  - `games-bench run --config <file> --game <name>`

State mode is explicit in run metadata:

- default stateful
- `--stateless` opt-in
- spec naming suffixed as `-stateful` / `-stateless`

## Coding Patterns in This Codebase

Recurring implementation patterns:

- immutable, slot-based dataclasses (`@dataclass(frozen=True, slots=True)`) for specs/jobs/results
- strict argument/config validation with immediate `SystemExit` on invalid run definitions
- explicit pure helpers for parsing/normalization
- event-first episode model (`state`, `provider_result`, `tool_call`, `tool_result`, `early_stop`)
- lazy optional dependency imports with actionable install guidance:
  - `openai` via `games-bench[llm]`
  - `pillow` via `games-bench[viz]`
  - `tqdm` via `games-bench[bench]`
- base package keeps zero mandatory third-party dependencies (`dependencies = []`)

## Backward Compatibility Shims

Thin re-export modules preserved for compatibility:

- `games_bench/hanoi.py`
- `games_bench/vision.py`
- `games_bench/llm/game_adapter.py`
- `games_bench/bench/hanoi_adapter.py`
- `games_bench/llm/prompting.py` (deprecated empty shim)

These should remain logic-free.

## Testing Focus

Tests in `tests/` emphasize architecture and contract stability:

- registry loading and built-in availability
- CLI shape and flag-scoping behavior
- config precedence and suite application
- checkpoint/resume and strict-resume failure modes
- scoring/taxonomy behavior
- compare command pairing and threshold gating
- game-specific env/adapter/render/review contracts

## Extending with a New Game

Minimum path:

1. Add `games_bench/games/<game>/` (`env.py`, `adapter.py`, `__init__.py`, prompts/vision/render as needed).
2. Register game in `games_bench/games/registry.py`.
3. Add `games_bench/bench/<game>.py` implementing:
   - `run_batch`
   - `add_<game>_arguments`
   - `default_<game>_config`
   - `build_<game>_adapter`
   - `estimate_episodes`, `score_episodes`, `compare_metrics` (recommended)
4. Register benchmark in `games_bench/bench/registry.py`.
5. Add config entry under `games` in suite/config files.
6. Keep rendering in game layer; benchmark layer should orchestrate only.
