# games-bench Architecture Review

## Goal

This repository contains two independent systems:

1. **Games engine** (`games_bench/games/`) -- reusable game environments for RL, planning, or any experiment.
2. **Benchmark system** (`games_bench/bench/` + `games_bench/llm/`) -- an evaluation wrapper that drives LLMs through those games via tool-calling.

The games should never depend on benchmarking code. The benchmark is a consumer of game environments, not the other way around. Both should be independently useful to the open-source community: researchers can use the environments for their own work, and others can benchmark models.

This report audits how well that separation holds today, identifies every violation, and provides a step-by-step plan to fix them.

---

## Findings

Ordered by severity (highest first).

### F1. Core package forces LLM dependencies on RL-only users

**Severity: High**

`pyproject.toml:7-8` lists `openai>=2.16.0` as a **base dependency**. The `[dependency-groups]` section (`pyproject.toml:16`) has `llm = ["openai"]` but this is redundant since `openai` is already required unconditionally.

A researcher who only wants `TowerOfHanoiEnv` for Q-learning must install the OpenAI SDK. The README even presents LLM deps as optional (`README.md:28`, `README.md:34` -- `uv sync --group llm`), contradicting what `pyproject.toml` actually enforces.

**Impact:** Blocks lightweight standalone use of the game engine. Discourages RL-only adoption.

### F2. LLM harness is hardcoded to Hanoi, not game-agnostic

**Severity: High**

`llm/harness.py:7-8` hard-imports Hanoi-specific types:

```python
from games_bench.games.hanoi.env import HanoiToolbox, TowerOfHanoiEnv, tool_schemas
from games_bench.games.hanoi.prompts import default_instructions
```

Specific violations:
- `harness.py:27-42` -- `_execute_tool()` dispatches on hardcoded `hanoi_*` tool names (`hanoi_get_state`, `hanoi_move`, `hanoi_reset`, etc.).
- `harness.py:61` -- `run_tool_calling_episode(env: TowerOfHanoiEnv, ...)` takes a concrete type, not a protocol.
- `harness.py:73` -- wraps env in `HanoiToolbox(env)` unconditionally.
- `harness.py:83` -- defaults to `default_instructions()` from Hanoi prompts.
- `harness.py:14-20` -- `EpisodeResult` has Hanoi-specific fields: `n_disks`, `optimal_steps`.

Additionally:
- `llm/__init__.py:6` re-exports `default_instructions` from `games_bench.games.hanoi.prompts`.
- `llm/prompting.py:5` is a backward-compat re-export of all Hanoi prompt helpers.

**Impact:** Adding a second game requires rewriting the harness. The multi-game registry pattern in `games/registry.py` is currently underused because the harness it feeds into only speaks Hanoi.

### F3. Batch orchestration is Hanoi-bootstrapped despite multi-game framing

**Severity: High**

- `batch.py:8` -- `from games_bench.bench import hanoi as hanoi_bench`
- `batch.py:21` -- uses `hanoi_bench.build_parser()` as the CLI parser for the entire batch system, meaning all batch CLI flags are Hanoi-specific argparse definitions.
- `batch.py:34` -- `normalize_games_config(config, default_game="hanoi")` hardcodes the default.
- `bench/registry.py:36-43` -- `load_builtin_benchmarks()` only wires Hanoi.

The batch dispatcher (`batch.py:40-43`) iterates games and calls `benchmark.batch_runner(args, game_config)`, but `args` is a `Namespace` shaped by Hanoi's parser. A second game would receive `--n-disks`, `--start-peg`, `--goal-peg` flags that don't apply to it.

**Impact:** The multi-game config format (`configs/hanoi.json` with `"games": { ... }`) and registry pattern exist but cannot actually serve a second game without restructuring the CLI.

### F4. Recording move counter is incorrect when non-move tools are used

**Severity: High (data correctness)**

`recording.py:78-84`:

```python
tool_calls += 1
result = event.get("result", {})
ok = bool(result.get("ok", False))
if ok:
    moves += 1
else:
    illegal_moves += 1
```

Any tool result with `ok: true` increments `moves`. This includes `hanoi_get_state`, `hanoi_is_solved`, and `hanoi_get_legal_moves` -- all of which return `ok: true` but are **not moves**. The `move_and_state` and `all_tools` variants (`bench/hanoi.py:74-83`) enable these non-move tools by default.

Recorded `total_moves` in `summary` and per-step `totals.moves` will be inflated for any variant that allows querying tools. This corrupts benchmark comparison data since move counts are a primary metric.

**Impact:** Benchmark results from non-`move_only` tool variants have incorrect move counts. Comparisons across tool variants are unreliable.

### F5. `--state-format image` silently hard-fails for most providers

**Severity: Medium**

Image rendering is enabled in `bench/hanoi.py:672-688` when `state_format in {"image", "both"}`. But `harness.py:89-90` raises `ValueError` if the provider doesn't support images:

```python
if state_image_renderer and not getattr(provider, "supports_images", False):
    raise ValueError("Provider does not support image inputs.")
```

Provider support: `OpenRouterProvider` sets `supports_images = True` (`providers.py`). `OpenAIResponsesProvider`, `CLIProvider`, and `CodexCLIProvider` all default to `False`.

A batch config with `"state_format": "image"` and `"models": {"openai": [...]}` will crash at the start of the first episode with an unhandled `ValueError`, not a clean error message. There's no pre-flight check or graceful skip.

**Impact:** Easy footgun in batch configs. A long multi-model run can fail partway through.

### F6. Game-layer render/review CLIs assume benchmark run directory layout

**Severity: Low**

`games/hanoi/render.py:15-19` and `games/hanoi/review.py:22-26` parse run directory paths expecting `<provider>/<model>/<run_id>` structure (via `_extract_run_parts()`). These are game-layer modules but they embed benchmark output conventions.

The `bench/render.py` and `bench/review.py` wrappers are thin pass-throughs (3 lines each), confirming the real logic lives in the game layer.

**Impact:** Minor. These are visualization tools for benchmark outputs, so expecting benchmark structure is somewhat reasonable. But it means the game layer has implicit knowledge of the benchmark's file layout.

### F7. Top-level package exports blur the boundary

**Severity: Low**

`games_bench/__init__.py:5`:

```python
from .hanoi import ACTION_SPACE, HanoiState, TowerOfHanoiEnv
```

This makes `from games_bench import TowerOfHanoiEnv` the natural import path, but `games_bench` is the *benchmark package*. The canonical path for standalone game use should be `from games_bench.games.hanoi import TowerOfHanoiEnv`. The top-level re-export signals to contributors that the game *is* the benchmark.

**Impact:** Confusion about where the game boundary is. Not a functional issue.

### F8. Test coverage is narrow

**Severity: Medium**

`tests/test_hanoi.py` has 5 test cases, all for `TowerOfHanoiEnv` basics (reset, move, illegal moves, step). No tests for:
- `llm/harness.py` (episode orchestration)
- `llm/recording.py` (where the move counter bug lives)
- `llm/providers.py` (provider adapters)
- `bench/batch.py` (batch orchestration)
- `bench/registry.py` (game/benchmark registration)
- `config.py` (config loading/merging)
- `games/hanoi/vision.py` (image rendering)

**Impact:** Bugs like F4 have no regression safety net. Refactoring (especially the harness decoupling in F2) is risky without tests to catch regressions.

---

## What is already strong

- **Game engine isolation from benchmark imports.** `games_bench/games/**` has zero imports from `bench/` or `llm/`. The dependency arrow points the right way.
- **MIT license** in `LICENSE`.
- **Clear repo layout documentation** in `README.md` and `AGENTS.md`.
- **Standalone environment usage is documented** in the README's Tower of Hanoi section (benchmark docs are dominant, but not exclusive).
- **Rendering logic lives in the game layer** (`games/hanoi/render.py`, `games/hanoi/review.py`), with benchmark wrappers being thin pass-throughs (`bench/render.py`, `bench/review.py`).
- **Provider abstraction is clean.** `llm/providers.py` has zero game imports -- it's fully generic. `ToolCall`, `ProviderResult`, and the provider classes work for any tool-calling scenario.
- **The Hanoi environment is well-designed.** Dual RL/tool-calling interface, configurable rewards, clean state representation, good error hierarchy.
- **Registry pattern exists** in both `games/registry.py` and `bench/registry.py`. The architecture is there, just not followed through.
- **Existing tests pass** (5/5).

---

## Plan

### Phase 1: Fix data correctness (F4)

This is a bug that affects existing benchmark results. Fix it first.

**Step 1.1: Fix `recording.py` move counter**

In `llm/recording.py`, change the `tool_result` handler to only count `moves` when the tool call is state-mutating, not any successful tool call. Prefer explicit action metadata from the executor/adapter (for example, `state_mutating: true|false`) and use `state_after != state_before` only as a fallback when metadata is unavailable.

Implementation note: avoid relying only on state-diff heuristics. State equality can be brittle across games and representations, so metadata should be the primary signal.

**Step 1.2: Add tests for recording logic**

Write tests in `tests/test_recording.py` that:
- Feed events with only `hanoi_move` calls and verify `total_moves` matches.
- Feed events mixing `hanoi_move` and `hanoi_get_state` calls and verify `total_moves` only counts moves.
- Feed events with failed moves and verify `illegal_moves` counts correctly.

### Phase 2: Decouple dependencies (F1)

**Step 2.1: Move `openai` from base deps to optional extras**

In `pyproject.toml`:
- Remove `openai>=2.16.0` from `dependencies` (make it an empty list or stdlib-only).
- Add `[project.optional-dependencies]` section with `llm = ["openai>=2.16.0"]` and `viz = ["pillow"]`.
- The `[dependency-groups]` dev group stays as-is for development tooling.

After this change:
- `pip install games-bench` installs only the game engine (zero external deps).
- `pip install games-bench[llm]` adds OpenAI for benchmarking.
- `pip install games-bench[viz]` adds Pillow for image rendering.
- `pip install games-bench[llm,viz]` installs everything.

**Step 2.2: Preserve lazy provider imports and improve runtime guidance**

The current provider layer already imports `openai` lazily inside provider methods. Keep that behavior so importing `games_bench.llm` does not eagerly require SDKs. Ensure all provider constructors/methods raise explicit install guidance when dependencies are missing (for example: `pip install games-bench[llm]`).

**Step 2.3: Update README**

Change install instructions to reflect the extras:
- Quick start (RL only): `pip install games-bench`
- Benchmarking: `pip install games-bench[llm]`
- With image rendering: `pip install games-bench[llm,viz]`

### Phase 3: Introduce game-agnostic harness interface (F2)

This is the core architectural change. The goal is: any game that implements a simple protocol can be plugged into the harness without modifying `llm/` code.

**Step 3.1: Define a `GameAdapter` protocol**

Create `games_bench/llm/game_adapter.py` with a `typing.Protocol`:

```python
class GameAdapter(Protocol):
    def tool_schemas(self) -> list[dict[str, Any]]: ...
    def execute_tool(self, name: str, arguments: dict) -> dict[str, Any]: ...
    def get_state_snapshot(self) -> dict[str, Any]: ...
    def is_solved(self) -> bool: ...
    def default_instructions(self) -> str: ...
    def format_state(self) -> str: ...
    def episode_metrics(self) -> dict[str, Any]: ...
```

`episode_metrics()` returns game-specific fields (`n_disks`, `optimal_steps` for Hanoi, different fields for other games) as a flat dict, replacing the hardcoded fields on `EpisodeResult`.

**Step 3.2: Implement `HanoiAdapter`**

Create `games_bench/games/hanoi/adapter.py` that wraps `TowerOfHanoiEnv` + `HanoiToolbox` and implements `GameAdapter`. This consolidates the current `_execute_tool()` dispatch and prompt defaults into the game layer where they belong.

The hardcoded tool dispatch in `harness.py:27-42` moves into `HanoiAdapter.execute_tool()`.

**Step 3.3: Refactor the harness**

Rewrite `llm/harness.py:run_tool_calling_episode()` to accept `adapter: GameAdapter` instead of `env: TowerOfHanoiEnv`. Remove all Hanoi imports. The function becomes:

```python
def run_tool_calling_episode(
    adapter: GameAdapter,
    provider: Any,
    *,
    max_turns: int = 200,
    ...
) -> EpisodeResult:
```

Change `EpisodeResult` to hold `game_metrics: dict[str, Any]` instead of `n_disks` and `optimal_steps` directly.

**Step 3.4: Update `llm/__init__.py` and `llm/prompting.py`**

- Remove the `default_instructions` re-export from `llm/__init__.py`.
- Remove or deprecate `llm/prompting.py` (it only re-exports Hanoi prompts).

**Step 3.5: Update `bench/hanoi.py`**

Update `run_batch()` to construct a `HanoiAdapter` and pass it to `run_tool_calling_episode()` instead of passing the raw env.

**Step 3.6: Add harness tests**

Write `tests/test_harness.py` with a mock `GameAdapter` (a trivial game that solves in 1 move) and a mock provider. Test:
- Normal episode completes and returns correct metrics.
- Episode respects `max_turns`.
- Disallowed tools are rejected.
- Provider errors are handled.

### Phase 4: Make batch orchestration game-agnostic (F3)

**Step 4.1: Split CLI argument parsing**

Extract common batch arguments (provider, model, config, out-dir, timeout, retries, backoff, record flags) into a shared parser in `bench/cli.py` or a new `bench/common.py`. Each game's benchmark module defines only its game-specific arguments.

`batch.py:main()` builds the common parser, then each game's `BenchSpec` contributes its own argument group:

```python
@dataclass
class BenchSpec:
    name: str
    description: str
    batch_runner: Callable
    add_arguments: Callable[[argparse.ArgumentParser], None]  # new
```

**Step 4.2: Refactor `batch.py` to use registry-driven parsing**

Instead of `parser = hanoi_bench.build_parser()`, the batch main:
1. Builds a base parser with shared flags.
2. Loads builtin benchmarks from the registry.
3. Each benchmark adds its arguments via `spec.add_arguments(parser)`.
4. Parses args, dispatches to the matched game's `batch_runner`.

**Step 4.3: Update `bench/hanoi.py` parser**

Move Hanoi-specific flags (`--n-disks`, `--start-peg`, `--goal-peg`, `--prompt-variant`, `--tools-variant`, `--state-format`, `--image-*`) into a `add_hanoi_arguments(parser)` function. Keep `build_parser()` for backward-compatible standalone use.

### Phase 5: Fix provider/image safety (F5)

**Step 5.1: Add pre-flight capability check in batch runner**

In `bench/hanoi.py:run_batch()`, before entering the episode loop, check if `state_format` requires images and the provider doesn't support them. Either:
- Raise a clear error at config validation time (before any episodes run), or
- Skip image rendering for that provider with a warning and fall back to text-only.

Preferred: fail fast with a clear message at batch start, not mid-run.

**Step 5.2: Validate provider capability contracts in tests**

Providers already declare `supports_images` explicitly. Add tests that enforce this contract and verify image-mode pre-flight validation fails fast with clear messages for providers that do not support images.

### Phase 6: Improve open-source readiness (F8 + general)

**Step 6.1: Add tests for critical paths**

Priority test files to add:
- `tests/test_recording.py` -- recording builder, move counting (from Phase 1).
- `tests/test_harness.py` -- episode orchestration (from Phase 3).
- `tests/test_config.py` -- config loading, merging, normalization.
- `tests/test_registry.py` -- game and benchmark registration.

**Step 6.2: Add standalone environment documentation**

Add a section to the README showing standalone game usage without any benchmark/LLM code:

```python
from games_bench.games.hanoi import TowerOfHanoiEnv

env = TowerOfHanoiEnv(n_disks=3, shaping_weight=0.1)
state = env.reset()

for step in range(100):
    action = my_agent.act(state)
    state, reward, done, info = env.step(action)
    if done:
        break
```

**Step 6.3: Consider Gymnasium wrapper (future)**

Not blocking for initial release, but adding a `HanoiGymnasiumEnv(gymnasium.Env)` wrapper would make the environment instantly compatible with Stable-Baselines3, CleanRL, RLlib, and the rest of the RL ecosystem. This can be a separate `games_bench.games.hanoi.gymnasium` module with `gymnasium` as an optional dependency.

---

## Execution order

```
Phase 1 (F4: recording bug)     -- standalone fix, no dependencies
Phase 2 (F1: optional deps)     -- standalone fix, no dependencies
Phase 3 (F2: game adapter)      -- largest change, core decoupling
Phase 4 (F3: batch CLI)         -- depends on Phase 3 patterns
Phase 5 (F5: image safety)      -- small, can parallel with Phase 4
Phase 6 (F8: tests + docs)      -- ongoing, some tests written in earlier phases
```

Phases 1 and 2 are independent and can be done in parallel. Phase 3 is the critical path. Phases 4 and 5 can be parallelized after Phase 3 is complete. Phase 6 is incremental throughout.
