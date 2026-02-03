## games-bench

Framework-agnostic benchmark environments for reinforcement learning (RL) and LLM tool-calling research.

### Setup (uv)

- Create/update the virtual environment (installs dev tools like `black` and `pre-commit`): `uv sync`
- Run tests: `uv run python -m unittest discover -s tests`
- Run examples: `uv run python -m examples.hanoi_rl`
- Optional: set API keys in `.env`

### LLM tool-calling

There are two ways to try tool-calling:

- Manual loop (copy/paste tool calls from any LLM UI): `uv run python -m examples.hanoi_manual_tool_loop`
- OpenAI API loop (requires `OPENAI_API_KEY`): `uv sync --group llm` then `uv run python -m examples.hanoi_openai_tool_calling`

### Provider harnesses (OpenRouter / CLI agents)

Run a provider-backed benchmark episode:

- `uv sync --group llm`
- OpenRouter: `OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run python -m examples.hanoi_provider_benchmark --provider openrouter`
  - Optional: `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE`
- OpenAI Responses: `OPENAI_API_KEY=... uv run python -m examples.hanoi_provider_benchmark --provider openai`

CLI-backed agents (Codex CLI, Claude Code, etc.):

- Codex CLI (uses `codex exec`): `uv run python -m examples.hanoi_provider_benchmark --provider codex`
- Generic CLI (expects a JSON tool call on stdout):
  - Example with Claude Code: `uv run python -m examples.hanoi_provider_benchmark --provider cli --cli-cmd "claude -p --output-format json {prompt}" --no-stdin`

Command block:

```
uv sync --group llm
OpenRouter: OPENROUTER_API_KEY=... OPENROUTER_MODEL=... uv run python -m examples.hanoi_provider_benchmark --provider openrouter
OpenAI: OPENAI_API_KEY=... uv run python -m examples.hanoi_provider_benchmark --provider openai
Codex CLI: uv run python -m examples.hanoi_provider_benchmark --provider codex
Claude Code (generic CLI): uv run python -m examples.hanoi_provider_benchmark --provider cli --cli-cmd "claude -p --output-format json {prompt}" --no-stdin
```

### Batch benchmark

Run multiple episodes across variants (n_disks, prompts, allowed tools) and write JSONL traces:

- `uv run python -m examples.hanoi_batch_benchmark --provider openrouter --n-disks 3,4 --prompt-variant minimal --prompt-variant full --tools-variant move_only --tools-variant all_tools --runs-per-variant 5`
- Or use `config.json` to run a list of models:
  - `uv run python -m examples.hanoi_batch_benchmark --provider openrouter --config config.json`

`config.json` supports:

- `models` (list or map by provider)
- `n_disks`, `prompt_variants`, `tool_variants`, `runs_per_variant`, `max_turns`, `out_dir`

Outputs go to `runs/hanoi/<provider>/<model>/<run_id>/` with:

- `run_config.json` (run settings)
- `episodes.jsonl` (per-episode metrics)
- `traces.jsonl` (full traces)
- `summary.json` (aggregates)

### Tower of Hanoi

The Tower of Hanoi environment lives in `games_bench/hanoi.py` and supports:

- RL loop: `state -> action -> step() -> (state, reward, done, info)`
- Tool-calling: JSON schemas via `games_bench.hanoi.tool_schemas()` and a prompt-friendly state string via `TowerOfHanoiEnv.format_prompt_state()`

#### State representation

`TowerOfHanoiEnv.get_state()` returns a `HanoiState` snapshot with two synchronized views:

- `pegs`: 3 stacks of disk IDs (bottom->top) — easy to read and to validate legal moves
- `disk_positions`: a fixed-length vector (len = `n_disks`) where `disk_positions[d-1]` is the peg holding disk `d` — convenient for RL feature encodings (e.g., base-3 integer encoding)

#### Illegal moves

- `move(from_peg, to_peg)` is strict and raises `IllegalMoveError` if the move is not legal.
- `step(action)` is configurable via `illegal_action_behavior` (`"penalize"` by default) so RL training loops can keep running.

#### Rewards

The reward function is configurable:

- `step_penalty`: applied on each *legal* step (often negative to encourage shorter solutions)
- `illegal_move_penalty`: applied when an illegal action is attempted in `step()`
- `solve_reward`: added when the puzzle is solved
- `shaping_weight`: optional reward shaping based on progress on the goal peg (set to `0.0` for sparse rewards)

Run the examples:

- `uv run python -m examples.hanoi_rl`
- `uv run python -m examples.hanoi_tool_calling`

Run tests:

- `uv run python -m unittest discover -s tests`
