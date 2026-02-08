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

Run config-driven mode (multi-game capable):

- `uv run games-bench run --provider openrouter --config configs/hanoi.json --game hanoi`
- `uv run games-bench run --provider openrouter --config configs/sokoban.json --game sokoban`

## Config Model

Batch config precedence:

- `BenchSpec.default_config() < global config < per-game overrides`

`config.json` supports:

- Global keys: `models`, `out_dir`, `record`, `record_raw`, `record_provider_raw`, `provider_retries`, `provider_backoff`
- Per-game keys under `"games"`:
  - Hanoi: `n_pegs`, `n_disks`, `runs_per_variant`, `prompt_variants`, `tool_variants`, `start_peg`, `goal_peg`, `state_format`, `image_size`, `image_labels`, `image_background`
  - Sokoban: `level_sets` / `level_ids`, `runs_per_level`, `max_optimal_moves`, `prompt_variants`, `tool_variants`, `detect_deadlocks`, `terminal_on_deadlock`, `state_format`, `image_tile_size`, `image_labels`, `image_background`

Hanoi note:

- If `goal_peg` is omitted, it defaults to `n_pegs - 1` per variant.

See examples:

- `configs/hanoi.json`
- `configs/sokoban.json`

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
