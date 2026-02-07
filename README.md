## games-bench

Framework-agnostic benchmark environments for reinforcement learning (RL) and LLM tool-calling research.

### Setup (uv)

- Create/update the virtual environment (installs dev tools like `black` and `pre-commit`): `uv sync`
- Run tests: `uv run python -m unittest discover -s tests`
- Run a quick demo: `uv run games-bench rl`
- Optional: set API keys in `.env`

If installing with `pip` instead of `uv`:
- Core (no LLM SDKs): `pip install games-bench`
- LLM providers: `pip install 'games-bench[llm]'`
- Visualization helpers: `pip install 'games-bench[viz]'`
- Everything optional: `pip install 'games-bench[llm,viz]'`

### Repo layout

- `games_bench/games/` — game engine (envs, adapters, prompts, rendering); standalone, no LLM deps
- `games_bench/llm/` — game-agnostic LLM harness (providers, episode loop, recording)
- `games_bench/bench/` — benchmark orchestration (CLI, batch runner, registries, demo entrypoints)
- `apps/renderer-hanoi/` — Three.js playback renderer
- `configs/` — sample benchmark configs
- `artifacts/` — run outputs (`runs/`, `renders/`, `reviews/`)

Legacy `examples/` entrypoints are kept as thin wrappers for backward compatibility.

### LLM tool-calling

There are two ways to try tool-calling:

- Manual loop (copy/paste tool calls from any LLM UI): `uv run games-bench manual-tool-loop`
- OpenAI API loop (requires `OPENAI_API_KEY`): `uv sync --group llm` then `uv run games-bench openai-tool-calling`

### Provider harnesses (OpenRouter / CLI agents)

Run a provider-backed benchmark episode:

- `uv sync --group llm`
- OpenRouter: `OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run games-bench provider --provider openrouter`
  - Optional: `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE`
- OpenAI Responses: `OPENAI_API_KEY=... uv run games-bench provider --provider openai`

CLI-backed agents (Codex CLI, Claude Code, etc.):

- Codex CLI (uses `codex exec`): `uv run games-bench provider --provider codex`
- Generic CLI (expects a JSON tool call on stdout):
  - Example with Claude Code: `uv run games-bench provider --provider cli --cli-cmd "claude -p --output-format json {prompt}" --no-stdin`

Command block:

```
uv sync --group llm
OpenRouter: OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run games-bench provider --provider openrouter
OpenAI: OPENAI_API_KEY=... uv run games-bench provider --provider openai
Codex CLI: uv run games-bench provider --provider codex
Claude Code (generic CLI): uv run games-bench provider --provider cli --cli-cmd "claude -p --output-format json {prompt}" --no-stdin
```

### Batch benchmark

Run multiple episodes across variants (n_disks, prompts, allowed tools) and write JSONL traces:

- `uv run games-bench run hanoi --provider openrouter --n-disks 3,4 --prompt-variant minimal --prompt-variant full --tools-variant move_only --tools-variant all_tools --runs-per-variant 5`
- Or use `configs/hanoi.json` to run a list of models:
  - `uv run games-bench run --provider openrouter --config configs/hanoi.json`
  - Limit to a game: `uv run games-bench run --provider openrouter --config configs/hanoi.json --game hanoi`

`config.json` supports:

- Global defaults: `models` (list or map by provider), `out_dir`, `record`, `record_raw`,
  `record_provider_raw`, `provider_retries`, `provider_backoff`, `state_format`,
  `image_size`, `image_labels`, `image_background`
- Per-game overrides under `games` (object or list), e.g. `n_disks`, `prompt_variants`,
  `tool_variants`, `runs_per_variant`, `max_turns`, `start_peg`, `goal_peg`

Example (multi-game ready):

```
{
  "models": { "openrouter": ["openai/gpt-4.1-mini"] },
  "out_dir": "artifacts/runs",
  "games": {
    "hanoi": {
      "n_disks": [3, 4],
      "prompt_variants": ["minimal", "full"],
      "tool_variants": ["move_only", "all_tools"]
    }
  }
}
```

Outputs go to `artifacts/runs/<game>/<provider>/<model>/<run_id>/` with:

- `run_config.json` (run settings)
- `episodes.jsonl` (per-episode metrics)
- `traces.jsonl` (full traces)
- `summary.json` (aggregates)

Recordings (states + actions only):

- Add `--record` to the batch run to write `recordings/episode_XXXX.json` in the run folder.
- Render later with: `uv run games-bench render --run-dir <run_dir> --format html`
- Optional video: `uv sync --group viz` then `uv run games-bench render --run-dir <run_dir> --format video`
- Manual review (prompt + per-step images): `uv run games-bench review --run-dir <run_dir>`
- Raw generations (prompt + model output + tool result): add `--record-raw` to `games-bench run` to write `raw_generations.jsonl`.

### 3D Renderer (Bun + Three.js)

Interactive 3D playback lives under `apps/renderer-hanoi` and supports arbitrary pegs/disks and recording playback.

Run locally:

- `cd apps/renderer-hanoi`
- `bun install`
- `bun run dev`

Load a recording via file picker or URL (e.g., `recordings/episode_0000.json` from a run). Use the export buttons to save PNGs or record a WebM video.

For vision benchmarks or automation, the renderer exposes:

- `window.hanoiRenderer.setState(pegs, nDisks)`
- `window.hanoiRenderer.setRecordingStep(index)`
- `window.hanoiRenderer.exportPNG()`

### Standalone environment usage

The game environments work without any LLM or benchmark code:

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

### Tower of Hanoi

The Tower of Hanoi environment lives in `games_bench/games/hanoi/env.py` (with a backward-compatible re-export in `games_bench/hanoi.py`) and supports:

- RL loop: `state -> action -> step() -> (state, reward, done, info)`
- Tool-calling: JSON schemas via `games_bench.games.hanoi.tool_schemas()` and a prompt-friendly state string via `TowerOfHanoiEnv.format_prompt_state()`

#### State representation

`TowerOfHanoiEnv.get_state()` returns a `HanoiState` snapshot with two synchronized views:

- `pegs`: 3 stacks of disk IDs (bottom->top) — easy to read and to validate legal moves
- `disk_positions`: a fixed-length vector (len = `n_disks`) where `disk_positions[d-1]` is the peg holding disk `d` — convenient for RL feature encodings (e.g., base-3 integer encoding)

#### Illegal moves

- `move(from_peg, to_peg)` is strict and raises `IllegalMoveError` if the move is not legal.
- `step(action)` is configurable via `illegal_action_behavior` (`"penalize"` by default) so RL training loops can keep running.

#### Rewards

The reward function is configurable:

#### Prompts

Prompt templates for the Hanoi benchmark live in `games_bench/games/hanoi/prompts/` (including the explicit goal and image-specific suffix).

- `step_penalty`: applied on each *legal* step (often negative to encourage shorter solutions)
- `illegal_move_penalty`: applied when an illegal action is attempted in `step()`
- `solve_reward`: added when the puzzle is solved
- `shaping_weight`: optional reward shaping based on progress on the goal peg (set to `0.0` for sparse rewards)

Run the demos:

- `uv run games-bench rl`
- `uv run games-bench tool-calling`

Run tests:

- `uv run python -m unittest discover -s tests`
