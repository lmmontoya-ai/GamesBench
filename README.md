## GamesBench

Benchmark environments and orchestration for evaluating long-horizon planning and spatial reasoning in tool-calling agents.

## Install

- Dev install: `uv sync`
- Run all tests: `uv run python -m unittest discover -s tests`

Optional dependency groups:

- LLM providers: `uv sync --group llm`
- Visualization/review images: `uv sync --group viz`
- Bench progress UI: `uv sync --group bench`
- Everything: `uv sync --group llm --group viz --group bench`

Equivalent `pip` installs:

- Core: `pip install games-bench`
- LLM: `pip install 'games-bench[llm]'`
- Viz: `pip install 'games-bench[viz]'`
- Bench progress: `pip install 'games-bench[bench]'`
- All optional: `pip install 'games-bench[llm,viz,bench]'`

CLI entrypoints:

- Preferred: `uv run games-bench ...`
- Fallback: `uv run python -m games_bench.bench ...`

## Fastest way to run the benchmark

This repo is suite-first. Start with predefined specs:

1. List available suites:

- `uv run games-bench run --list-suites`

2. Run `easy-v1` (small-model-friendly) on one model:

- `uv run games-bench run --provider openrouter --model <model_name> --suite easy-v1`

3. Run `standard-v1` (harder canonical benchmark):

- `uv run games-bench run --provider openrouter --model <model_name> --suite standard-v1`

4. Run only one game from a suite:

- `uv run games-bench run --provider openrouter --model <model_name> --suite standard-v1 --game hanoi`
- `uv run games-bench run --provider openrouter --model <model_name> --suite standard-v1 --game sokoban`

5. Run stateless variant (stateful is default):

- `uv run games-bench run --provider openrouter --model <model_name> --suite standard-v1 --stateless`

OpenRouter setup:

- `export OPENROUTER_API_KEY="..."`

## Common workflows

### 1) Single model, fully reproducible run id

- `uv run games-bench run --provider openrouter --model <model_name> --suite easy-v1 --run-id easy_baseline_modelA`

Notes:

- `--run-id` makes output directories deterministic.
- If multiple models are run in one invocation, a provider/model suffix is appended internally.

### 2) Resume an interrupted run

- `uv run games-bench run --provider openrouter --model <model_name> --suite easy-v1 --run-id easy_baseline_modelA --resume`

Stricter safety checks:

- `uv run games-bench run --provider openrouter --model <model_name> --suite easy-v1 --run-id easy_baseline_modelA --resume --strict-resume`

Checkpoint cadence:

- `--checkpoint-interval <N>` (default 1 committed episode)

### 3) Run a model list (benchmark study)

Create a config overlay:

```json
{
  "models": {
    "openrouter": [
      "model-a",
      "model-b"
    ]
  }
}
```

Run:

- `uv run games-bench run --provider openrouter --suite standard-v1 --config configs/models_openrouter.example.json`

### 4) Generation first, scoring later

Generate artifacts without writing summary:

- `uv run games-bench run --provider openrouter --model <model_name> --suite easy-v1 --no-score`

Score offline from artifacts:

- `uv run games-bench score --run-dir <run_dir>`

Overwrite existing summary after metric/taxonomy updates:

- `uv run games-bench score --run-dir <run_dir> --overwrite`

Optionally write taxonomy fields back into `episodes.jsonl`:

- `uv run games-bench score --run-dir <run_dir> --write-taxonomy`

### 5) Baseline vs candidate governance compare

Compare two scored runs:

- `uv run games-bench compare --baseline <baseline_run_dir> --candidate <candidate_run_dir>`

Apply threshold gating and fail CI on regressions:

- `uv run games-bench compare --baseline <baseline_run_dir> --candidate <candidate_run_dir> --thresholds <thresholds.json> --fail-on-regression`

Example thresholds file:

```json
{
  "metrics": {
    "solve_rate": {"direction": "higher_better", "max_drop": 0.02},
    "avg_illegal_moves": {"direction": "lower_better", "max_increase": 0.20},
    "deadlock_rate": {"direction": "lower_better", "max_increase": 0.03}
  },
  "gating": {
    "require_same_spec": true,
    "require_same_interaction_mode": true
  }
}
```

Machine-readable output is written to `compare_report.json` by default (override with `--report-file`).

## Output artifacts

Runs are stored under:

- `artifacts/runs/<game>/<provider>/<model>/<run_id>/`

Each run directory contains:

- `run_config.json`
- `run_manifest.json` (lineage/provenance)
- `execution_state.json` (checkpoint/resume state)
- `episodes.jsonl`
- `traces.jsonl`
- `summary.json` (if scoring is enabled)
- `recordings/episode_XXXX.json` (when `--record` is enabled)
- `raw_generations.jsonl` (when `--record-raw` is enabled)

Important fields:

- `spec`: `<spec>-stateful` or `<spec>-stateless`
- `interaction_mode`: `stateful` or `stateless`
- `episodes.jsonl` includes taxonomy fields: `outcome_code`, `failure_tags`, `taxonomy_version`

## Config model

Merge precedence:

- `BenchSpec.default_config()` < global config < per-game overrides

Supported run shapes:

- `games-bench run <game> [flags]`
- `games-bench run --config <file>`
- `games-bench run --config <file> --game <name>`

Global config keys (top-level):

- Core: `models`, `spec`, `stateless`, `out_dir`
- Recording: `record`, `record_raw`, `record_provider_raw`
- Provider controls: `provider_retries`, `provider_backoff`, `stream_debug`
- Throughput controls: `parallelism`, `max_inflight_provider`
- Stop controls: `stagnation_patience`
- Resume controls: `run_id`, `resume`, `strict_resume`, `checkpoint_interval`
- Progress controls: `progress`, `progress_refresh_s`
- Scoring controls: `score` (default true), `score_version` (default `score-v1`)

Per-game keys under `games`:

- Hanoi:
  - `cases` or `n_pegs` + `n_disks`
  - `runs_per_variant`, `prompt_variants`, `tool_variants`
  - `start_peg`, `goal_peg`
  - `state_format`, `image_size`, `image_labels`, `image_background`
  - `optimal_turn_cap_multiplier`
- Sokoban (bundled):
  - `level_sets` / `level_ids`, `runs_per_level`, `max_optimal_moves`
  - `prompt_variants`, `tool_variants`
  - `detect_deadlocks`, `terminal_on_deadlock`, `deadlock_patience`
  - `state_format`, `image_tile_size`, `image_labels`, `image_background`
- Sokoban (procedural):
  - `procgen_grid_sizes`, `procgen_box_counts`, `procgen_levels_per_combo`
  - or explicit `procgen_cases` entries with `grid_size`, `box_count`, optional `levels_per_combo`, `wall_density`, `scramble_steps`
  - `procgen_seed`

Example configs:

- `configs/easy_v1.json`
- `configs/standard_v1.json`
- `configs/hanoi.json`
- `configs/sokoban.json`
- `configs/sokoban_procgen.json`
- `configs/models_openrouter.example.json`

## Advanced single-game runs

Hanoi:

- `uv run games-bench run hanoi --provider openrouter --model <model_name> --n-pegs 4 --n-disks 8 --runs-per-variant 3 --prompt-variant full --tools-variant move_only`

Sokoban bundled level:

- `uv run games-bench run sokoban --provider openrouter --model <model_name> --level-id starter-authored-v1:1 --runs-per-level 2 --prompt-variant full --tools-variant move_and_query`

Sokoban procedural:

- `uv run games-bench run sokoban --provider openrouter --model <model_name> --procgen-grid-size 10x10 --procgen-box-count 4 --procgen-levels-per-combo 3 --procgen-seed 42 --runs-per-level 2`

## Render and review

Render from recordings:

- Hanoi HTML: `uv run games-bench render --game hanoi --run-dir <run_dir> --format html`
- Sokoban HTML: `uv run games-bench render --game sokoban --run-dir <run_dir> --format html`
- ASCII mode: add `--format ascii`
- Hanoi video mode: `--format video` (requires `ffmpeg`)

Generate review bundles:

- `uv run games-bench review --game hanoi --run-dir <run_dir>`
- `uv run games-bench review --game sokoban --run-dir <run_dir>`

## Provider harnesses (single-episode demos)

- OpenRouter: `OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run games-bench provider --provider openrouter`
- OpenAI: `OPENAI_API_KEY=... uv run games-bench provider --provider openai`
- Codex CLI: `uv run games-bench provider --provider codex`
- Generic CLI: `uv run games-bench provider --provider cli --cli-cmd "<command>" --no-stdin`

OpenRouter stream diagnostics:

- Add `--stream-debug` to `games-bench run ...` or `games-bench provider ...`

## Repository layout

- `games_bench/games/` game engine (no benchmark deps)
- `games_bench/llm/` game-agnostic harness/providers/recording
- `games_bench/bench/` orchestration CLI/registry/batch/scoring
- `configs/` benchmark configs
- `artifacts/` generated runs/renders/reviews
- `apps/renderer-hanoi/` interactive Hanoi renderer

## Notes

- Layering and boundaries: `AGENTS.md`
- Suite authoring guidance: `easy_spec_guideline.md`, `standard_spec_guidelines.md`
- Design/roadmap context: `development_design.md`, `BENCH_MATURITY_PLAN.md`
