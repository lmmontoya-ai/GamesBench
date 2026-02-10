# Bench Operational Maturity Plan

Date: 2026-02-10
Owner: `games_bench/bench/*`
Status: Design + implementation plan

## Status Matrix

Last updated: 2026-02-10

| Phase | Scope | Status | Notes |
|---|---|---|---|
| Phase 1 | lineage + taxonomy + score command | Done | Implemented in `lineage.py`, `taxonomy.py`, `scoring.py`; wired into run and CLI (`score`, `--no-score`). |
| Phase 2 | shared executor + checkpoint/resume | Done | Implemented in `executor.py`, `checkpoint.py`; wired flags `--run-id`, `--resume`, `--strict-resume`, `--checkpoint-interval`; added resume tests. |
| Phase 3 | compare + governance | Done | Implemented `compare.py`, CLI `games-bench compare`, threshold gating, regression exit code, and compare tests/docs. |
| Phase 4 | hardening + CI alignment | Done | Added dedicated tests for lineage/taxonomy/score/checkpoint recovery/compare fixtures and legacy artifact compatibility checks. |

## 1. Problem Statement

Current benchmark orchestration is functional, but the largest maturity gaps are:

1. weak run lineage/provenance metadata
2. no first-class resume/checkpoint for interrupted long runs
3. no explicit regression governance workflow
4. duplicated execution/orchestration logic across `hanoi.py` and `sokoban.py`
5. generation and scoring tightly coupled
6. no standardized failure taxonomy across games

## 2. Design Principles (from best-in-class eval stacks)

Mapped practices:

- Inspect AI pattern: generation and scoring should be separable; logs should be rich and replayable.
- OpenAI/Anthropic eval pattern: explicit criteria + repeatable evaluation loop + baseline comparison.
- HELM/lm-eval pattern: normalized schema contracts and adapter-like extensibility.
- BBEH/PurpleLlama pattern: behavior/failure-mode classification should be first-class.

Implication for GameBench:

- keep existing layer boundaries (AGENTS.md) and game adapter contract
- raise operational maturity inside `games_bench/bench/*` only

## 3. Target Architecture

### 3.1 New modules

- `games_bench/bench/lineage.py`
  - Build canonical run manifest (`run_manifest.json`)
  - Compute hashes for config/spec/prompts/tool schemas
  - Capture git/process/environment lineage
- `games_bench/bench/executor.py`
  - Shared concurrent episode execution and ordered commit pipeline
  - Own checkpoint/resume orchestration
- `games_bench/bench/checkpoint.py`
  - Durable execution state and JSONL recovery helpers
- `games_bench/bench/scoring.py`
  - Offline scoring entrypoints and score-version stamping
- `games_bench/bench/taxonomy.py`
  - Shared failure taxonomy classification
- `games_bench/bench/compare.py`
  - Baseline vs candidate comparisons + thresholds + exit codes
- `games_bench/bench/io.py` (optional helper module)
  - JSONL read/append/truncate-safe primitives

### 3.2 Registry extensions

Extend `BenchSpec` in `games_bench/bench/registry.py`:

- `episode_scorer: Callable[[list[dict[str, Any]]], dict[str, Any]] | None`
- `episode_taxonomy: Callable[[dict[str, Any], dict[str, Any]], list[str]] | None`
- `compare_metrics: Callable[[dict[str, Any]], dict[str, float]] | None`

This keeps game-specific metric logic local while executor/scoring/compare remain generic.

## 4. Data Contracts

### 4.1 Run manifest

New file per run directory: `run_manifest.json`

Schema (`run_manifest_version = "v1"`):

- `run_id`, `timestamp_utc`, `created_at_unix`
- `game`, `provider`, `model`, `spec`, `interaction_mode`
- `argv`, `cwd`, `python_version`, `platform`
- `git`: `commit`, `branch`, `is_dirty`, `remote_origin` (best-effort)
- `config_hash`, `suite_hash`, `prompt_hash`, `tool_schema_hash`
- `seed_lineage`: global/procgen/episode seed metadata
- `parent_run_id` (for resumes/rescoring lineage)
- `bench_version` (package version if available)

Compatibility:

- keep existing `run_config.json`
- do not remove or rename existing run files in v1

### 4.2 Execution state/checkpoint

New file: `execution_state.json`

- `checkpoint_version = "v1"`
- `run_id`
- `total_jobs`
- `completed_episode_ids` (sorted list)
- `next_uncommitted_episode_id`
- `last_updated`
- `job_plan_hash` (detects mismatched resume input)

### 4.3 Score metadata

Add/maintain in `summary.json`:

- `score_version` (e.g., `score-v1`)
- `scored_at`
- `scoring_input`: source files + hashes
- `taxonomy_version`

### 4.4 Episode taxonomy fields

Append standardized fields to each episode record:

- `outcome_code`: one of `solved`, `failed_stagnation`, `failed_deadlock_terminal`, `failed_budget`, `failed_provider`, `failed_unknown`
- `failure_tags`: list of normalized tags
- `signals`: optional compact dict (`turn_count`, `illegal_moves`, etc.)

## 5. Shared Executor Refactor

### 5.1 Executor interface

Proposed API in `games_bench/bench/executor.py`:

```python
@dataclass(frozen=True, slots=True)
class EpisodeJob:
    episode_id: int
    variant_id: str
    payload: Any

@dataclass(frozen=True, slots=True)
class EpisodeOutput:
    episode_id: int
    variant_id: str
    episode: dict[str, Any]
    events: list[dict[str, Any]]
    raw_lines: list[str]
    recording: dict[str, Any] | None

@dataclass(frozen=True, slots=True)
class ExecutePlan:
    jobs: list[EpisodeJob]
    parallelism: int
    record: bool
    record_raw: bool
    resume: bool


def execute_plan(
    *,
    out_dir: Path,
    plan: ExecutePlan,
    run_job: Callable[[EpisodeJob], EpisodeOutput],
    progress_reporter: Any | None,
) -> list[dict[str, Any]]:
    ...
```

Responsibilities:

- ordered commit by `episode_id` even with parallel workers
- append-safe writes to `episodes.jsonl`, `traces.jsonl`, `raw_generations.jsonl`
- optional recording writes under `recordings/`
- checkpoint update after each committed episode
- resume skip for already committed episode IDs

### 5.2 What leaves game runners

Remove duplicated logic from `hanoi.py` and `sokoban.py`:

- file-opening/writing boilerplate
- pending-futures + ordered commit loops
- checkpointing and recovery behavior
- summary write trigger

Game runners keep:

- parsing/config selection
- job enumeration
- per-job `run_job` implementation
- game-specific default config and argument validation

## 6. Resume/Checkpoint UX

### 6.1 CLI additions (`games_bench/bench/common.py`)

Add common flags:

- `--resume` (resume existing run directory if present)
- `--run-id <id>` (deterministic run dir naming)
- `--checkpoint-interval <N>` (default 1 committed episode)
- `--strict-resume` (error on any job-plan mismatch)

Behavior:

- if `--resume` and run exists: recover and continue
- if run exists and no `--resume`: fail with actionable message
- if `--run-id` provided, deterministic path is used instead of timestamp-based run id

### 6.2 Recovery rules

On resume:

1. load `execution_state.json`
2. validate `job_plan_hash`
3. parse existing JSONL files; truncate trailing invalid partial line if needed
4. reconstruct `completed_episode_ids`
5. skip completed jobs and continue appending

## 7. Generation/Scoring Separation

### 7.1 Run path

Add flag to `games-bench run` and `games-bench run <game>`:

- `--no-score`: perform generation only (write `episodes.jsonl` + `traces.jsonl`), skip summary

### 7.2 New scoring command

Add CLI command in `games_bench/bench/cli.py`:

- `games-bench score --run-dir <dir> [--game <name>] [--score-version score-v1] [--overwrite]`

Scoring flow:

1. load `run_config.json` and `episodes.jsonl`
2. apply taxonomy classifier
3. compute `overall` and `variants` metrics via game scorer from registry
4. write/update `summary.json` with score metadata

This enables re-scoring after taxonomy/metric changes without rerunning model generations.

## 8. Standardized Failure Taxonomy

### 8.1 Taxonomy v1 core tags

Core tags (game-agnostic):

- `stagnation_stop`
- `deadlock_terminal`
- `turn_budget_exhausted`
- `provider_error`
- `illegal_action_burst`
- `query_loop`
- `unsolved_final`

Sokoban-specific extensions:

- `deadlocked_final`
- `deadlock_patience_stop`

Hanoi-specific extensions:

- `optimality_gap_high` (optional, only if solved and ratio threshold exceeded)

### 8.2 Classifier signature

```python
def classify_failure_tags(
    episode: dict[str, Any],
    *,
    run_config: dict[str, Any],
    game_name: str,
) -> tuple[str, list[str]]:
    ...
```

## 9. Regression Governance + Compare Command

### 9.1 New compare command

Add CLI command:

- `games-bench compare --baseline <run_or_dir> --candidate <run_or_dir> [--thresholds <json>] [--fail-on-regression]`

Supported input modes:

- single run-dir pair
- directory pair containing multiple run dirs; auto-match on key:
  - `(game, spec, interaction_mode, provider, model)`

### 9.2 Threshold schema

`thresholds.json` example:

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

Outputs:

- stdout human table
- `compare_report.json` machine-readable
- exit code `1` if any gated regression when `--fail-on-regression`

## 10. Implementation Plan (phased)

### Phase 1: Foundation (lineage + taxonomy + score command)

1. Add `lineage.py` and write `run_manifest.json` in both runners.
2. Add `taxonomy.py` and include tags in episode records.
3. Add `scoring.py` and `score` CLI command.
4. Add `--no-score` flow in run command.

Deliverable: existing behavior preserved, but run metadata richer and scoring can be rerun.

### Phase 2: Shared executor + checkpoint/resume

1. Add `executor.py` + `checkpoint.py`.
2. Refactor `hanoi.py` and `sokoban.py` to call shared executor.
3. Add flags: `--resume`, `--run-id`, `--checkpoint-interval`, `--strict-resume`.
4. Add file-recovery + job-plan hash guards.

Deliverable: no duplicated commit loops; interrupted runs recover safely.

### Phase 3: Compare + governance

1. Add `compare.py` and CLI command.
2. Add thresholds schema support + fail-on-regression exit behavior.
3. Add minimal baseline workflow docs in README.

Deliverable: reproducible regression gate workflow.

### Phase 4: Hardening

1. Add CI tests for resume/recover/compare/no-score/score.
2. Add deterministic smoke fixtures for compare command.
3. Add compatibility checks for older run dirs (no manifest, no taxonomy).

Deliverable: robust migration path with backward compatibility.

Phase 4 implementation status (2026-02-10):

- Added recovery-focused checkpoint tests in `tests/test_checkpoint_recovery.py`.
- Added dedicated score compatibility tests in `tests/test_score_cli.py`.
- Added dedicated lineage tests in `tests/test_lineage.py`.
- Added dedicated taxonomy tests in `tests/test_failure_taxonomy.py`.
- Added deterministic compare fixtures in `tests/fixtures/compare/simple/`.
- Added fixture-driven compare smoke test in `tests/test_compare_fixtures.py`.

## 11. Test Plan

New tests:

- `tests/test_lineage.py`
  - manifest contains required fields; hashes stable for same config.
- `tests/test_checkpoint_resume.py`
  - interrupted run resumes without duplicate episode IDs.
  - strict resume rejects mismatched job plan.
- `tests/test_score_cli.py`
  - `--no-score` leaves no summary.
  - `games-bench score` regenerates summary with `score_version`.
- `tests/test_failure_taxonomy.py`
  - taxonomy tags/outcome mapping for known episode patterns.
- `tests/test_compare_cli.py`
  - pass/fail threshold behavior and exit codes.

Regression tests to retain:

- all current suite/config/CLI tests
- progress reporter behavior and stdout cleanliness

## 12. Migration + Backward Compatibility

- Keep `run_config.json`, `episodes.jsonl`, `traces.jsonl`, `summary.json` names.
- New files are additive (`run_manifest.json`, `execution_state.json`, compare reports).
- `score` command should support old runs lacking taxonomy fields.
- Compare command should gracefully skip unavailable metrics with explicit warning.

## 13. Concrete File Touch Map

Primary files to modify:

- `games_bench/bench/cli.py`
- `games_bench/bench/common.py`
- `games_bench/bench/registry.py`
- `games_bench/bench/hanoi.py`
- `games_bench/bench/sokoban.py`
- `README.md`

New files:

- `games_bench/bench/lineage.py`
- `games_bench/bench/executor.py`
- `games_bench/bench/checkpoint.py`
- `games_bench/bench/scoring.py`
- `games_bench/bench/taxonomy.py`
- `games_bench/bench/compare.py`
- optional `games_bench/bench/io.py`

New tests:

- `tests/test_lineage.py`
- `tests/test_checkpoint_resume.py`
- `tests/test_score_cli.py`
- `tests/test_failure_taxonomy.py`
- `tests/test_compare_cli.py`

## 14. Definition of Done

This initiative is complete when:

1. both games run through a shared executor (no duplicated commit loop logic)
2. resumed runs are deterministic and do not duplicate committed episodes
3. scoring can be rerun offline from artifacts without model calls
4. failure taxonomy fields are present and versioned
5. compare command can gate regressions with configurable thresholds
6. lineage/provenance metadata is sufficient to reproduce a run and audit its origin
