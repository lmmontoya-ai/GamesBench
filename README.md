## games-bench

Framework-agnostic benchmark environments for RL and LLM tool-calling research.

## Setup

- Install dev environment: `uv sync`
- Run all tests: `uv run python -m unittest discover -s tests`

Optional dependency groups:

- LLM providers: `uv sync --group llm`
- Visualization/review images: `uv sync --group viz`
- Everything: `uv sync --group llm --group viz`

Equivalent `pip` installs:

- Core: `pip install games-bench`
- LLM: `pip install 'games-bench[llm]'`
- Viz: `pip install 'games-bench[viz]'`
- All optional: `pip install 'games-bench[llm,viz]'`

CLI invocation:

- Preferred: `uv run games-bench ...`
- Fallback (always works in-repo): `uv run python -m games_bench.bench ...`

## Repo Layout

- `games_bench/games/` game engine (envs/adapters/prompts/vision/render/review), no benchmark deps
- `games_bench/llm/` game-agnostic harness/providers/recording
- `games_bench/bench/` orchestration CLI/registry/batch dispatch
- `configs/` sample config files
- `artifacts/` generated runs/renders/reviews
- `apps/renderer-hanoi/` interactive Three.js Hanoi renderer

## Quick Start

Run a single-game benchmark:

- Hanoi:
  - `uv run games-bench run hanoi --provider cli --cli-cmd 'python -c "print(\"{\\\"name\\\":\\\"hanoi_move\\\",\\\"arguments\\\":{\\\"from_peg\\\":0,\\\"to_peg\\\":2}}\")"' --n-pegs 3 --n-disks 3 --runs-per-variant 1 --prompt-variant minimal --tools-variant move_only`
- Sokoban:
  - `uv run games-bench run sokoban --provider cli --cli-cmd 'python -c "print(\"{\\\"name\\\":\\\"sokoban_move\\\",\\\"arguments\\\":{\\\"direction\\\":\\\"right\\\"}}\")"' --level-id starter-authored-v1:1 --runs-per-level 1 --prompt-variant minimal --tools-variant move_only`
  - Procedural Sokoban:
    - `uv run games-bench run sokoban --provider cli --cli-cmd 'python -c "print(\"{\\\"name\\\":\\\"sokoban_move\\\",\\\"arguments\\\":{\\\"direction\\\":\\\"right\\\"}}\")"' --procgen-grid-size 8x8 --procgen-box-count 2 --procgen-levels-per-combo 2 --procgen-seed 42 --runs-per-level 1 --prompt-variant minimal --tools-variant move_only`

Run config-driven mode (multi-game capable):

- `uv run games-bench run --provider openrouter --config configs/hanoi.json --game hanoi`
- `uv run games-bench run --provider openrouter --config configs/sokoban.json --game sokoban`

Run canonical benchmark suite:

- List suites: `uv run games-bench run --list-suites`
- Run suite as-is: `uv run games-bench run --provider openrouter --model google/gemini-2.5-pro-preview --suite standard-v1`
- Run suite with local overrides from config: `uv run games-bench run --provider openrouter --suite standard-v1 --config configs/standard_v1.json`

## OpenRouter Benchmark Workflows

Set your API key:

- `export OPENROUTER_API_KEY="..."`

Run one model on the canonical suite:

- `uv run games-bench run --provider openrouter --model google/gemini-2.5-pro-preview --suite standard-v1`

Run one model on a single game:

- `uv run games-bench run --provider openrouter --model google/gemini-2.5-pro-preview --suite standard-v1 --game hanoi`
- `uv run games-bench run --provider openrouter --model google/gemini-2.5-pro-preview --suite standard-v1 --game sokoban`

Run a model list (recommended for benchmark studies):

1. Create a config overlay with models:

```json
{
  "models": {
    "openrouter": [
      "google/gemini-2.5-pro-preview",
      "openai/gpt-4.1",
      "anthropic/claude-3.7-sonnet"
    ]
  }
}
```

2. Run with suite + overlay:

- `uv run games-bench run --provider openrouter --suite standard-v1 --config configs/models_openrouter.example.json`

Notes:

- `models.openrouter` can be either a string (single model) or a list.
- `--config` values override suite defaults where keys overlap.
- You can combine `--game` with model lists to benchmark only one game.

## Config Model

Batch config precedence:

- `BenchSpec.default_config() < global config < per-game overrides`

`config.json` supports:

- Global keys: `models`, `out_dir`, `record`, `record_raw`, `record_provider_raw`, `provider_retries`, `provider_backoff`, `stream_debug`
- Per-game keys under `"games"`:
  - Hanoi: `cases` (exact `{n_pegs,n_disks}` tuples), or `n_pegs` + `n_disks` (cartesian product), plus `runs_per_variant`, `prompt_variants`, `tool_variants`, `start_peg`, `goal_peg`, `state_format`, `image_size`, `image_labels`, `image_background`
  - Sokoban (bundled): `level_sets` / `level_ids`, `runs_per_level`, `max_optimal_moves`, `prompt_variants`, `tool_variants`, `detect_deadlocks`, `terminal_on_deadlock`, `state_format`, `image_tile_size`, `image_labels`, `image_background`
  - Sokoban (procedural):
    - cross-product mode: `procgen_grid_sizes`, `procgen_box_counts`, `procgen_levels_per_combo`
    - case mode: `procgen_cases` entries with `grid_size`, `box_count`, optional `levels_per_combo`, `wall_density`, `scramble_steps` (int, `[min,max]`, `"min-max"`, or `"min+"`)
    - shared controls: `procgen_seed` plus standard run keys (`runs_per_level`, `prompt_variants`, etc.)

Sokoban level-source rules:

- Use either bundled levels (`level_sets` / `level_ids`) or procedural generation (`procgen_*`) for a run.
- Procedural runs are deterministic when `procgen_seed` is set.

Hanoi note:

- If `goal_peg` is omitted, it defaults to `n_pegs - 1` per variant.

See examples:

- `configs/hanoi.json`
- `configs/sokoban.json`
- `configs/sokoban_procgen.json`
- `configs/standard_v1.json`
- `configs/models_openrouter.example.json`

## Outputs

Run outputs are written under:

- `artifacts/runs/<game>/<provider>/<model>/<run_id>/`

Each run contains:

- `run_config.json`
- `episodes.jsonl`
- `traces.jsonl`
- `summary.json`
- `recordings/episode_XXXX.json` when `--record` is enabled
- `raw_generations.jsonl` when `--record-raw` is enabled

## Render And Review

Render playback from recordings:

- Hanoi: `uv run games-bench render --game hanoi --run-dir <run_dir> --format html`
- Sokoban: `uv run games-bench render --game sokoban --run-dir <run_dir> --format html`
- ASCII mode (both games): `--format ascii`
- Hanoi video mode: `--format video` (requires `ffmpeg`)

Generate review bundles (prompt + step images):

- Hanoi: `uv run games-bench review --game hanoi --run-dir <run_dir>`
- Sokoban: `uv run games-bench review --game sokoban --run-dir <run_dir>`

Review image rendering requires the viz dependency group (`pillow`).

## Provider Harnesses

Single-episode provider harness:

- OpenRouter: `OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run games-bench provider --provider openrouter`
- OpenAI: `OPENAI_API_KEY=... uv run games-bench provider --provider openai`
- Codex CLI: `uv run games-bench provider --provider codex`
- Generic CLI: `uv run games-bench provider --provider cli --cli-cmd "<command>" --no-stdin`

Streaming diagnostics (OpenRouter):

- Add `--stream-debug` to `games-bench run ...` or `games-bench provider ...` to print incremental stream chunks to `stderr`.

## Standalone Game Engine Usage

Hanoi example:

```python
from games_bench.games.hanoi import TowerOfHanoiEnv

env = TowerOfHanoiEnv(n_disks=3, n_pegs=4)
state = env.reset()
state, reward, done, info = env.step((0, env.goal_peg))
```

Sokoban example:

```python
from games_bench.games.sokoban import SokobanEnv, load_level_by_id

level = load_level_by_id("starter-authored-v1:1")
env = SokobanEnv(level, detect_deadlocks=True)
state = env.reset()
state, reward, done, info = env.step("right")
```

## Notes

- Layering and registry contracts are documented in `AGENTS.md`.
- Development plan and phase details are in `development_design.md`.
