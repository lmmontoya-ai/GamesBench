# AGENTS.md

This repo keeps game environments and benchmark harnesses in the same repository,
but they must remain **layered** so the game engine can be used standalone.

## Layer boundaries
- **Game engine (no benchmark dependencies):**
  - `games_bench/games/**` (env rules, tool schemas, prompts, rendering/vision helpers)
  - `games_bench/vision.py` (re-exports)
  - `apps/renderer-*` (game-specific interactive renderers)
- **Benchmark / harness (depends on game engine):**
  - `games_bench/llm/**` (providers + tool-calling loop)
  - `games_bench/bench/**` (CLI, batch, run orchestration)
  - `configs/` (benchmark configs)
- **Artifacts (outputs only):** `runs/`, `renders/`, `reviews/`, `artifacts/`

## Expectations between layers
- Game engine must **not** import benchmark or provider code.
- Benchmark/harness may import the game engine, but should keep game-specific logic
  thin and delegate to the game module.
- **Rendering belongs to the game engine**:
  - Rendering logic should live in `games_bench/games/<game>/vision.py`
    or `apps/renderer-<game>`.
  - Benchmark code should only orchestrate rendering, not implement it.
- Providers/harness should stay game-agnostic.
- New games go under `games_bench/games/<game>` and must be registered in
  `games_bench/games/registry.py`.

## Refactor guidance (current hot spots)
- `games_bench/games/hanoi/bench.py` contains game-specific benchmarking glue;
  keep it thin or push generic logic into `games_bench/bench/**`.
- `games_bench/bench/render.py` and `games_bench/bench/review.py` contain
  rendering/HTML logic; move game-specific rendering into the game layer.

## Dependency hygiene
- Keep LLM deps optional and out of the game engine.
- Keep visualization deps in `viz` group only.
- Secrets belong in `.env` (gitignored).
