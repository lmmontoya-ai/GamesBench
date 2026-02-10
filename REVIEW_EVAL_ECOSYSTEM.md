# GameBench External Benchmark Architecture Review

Date: 2026-02-10

## 1. Scope

This report compares the current GameBench benchmark orchestration with recent best practices from:

- Inspect AI
- OpenAI Evals tooling (API + docs)
- Anthropic evaluation workflow docs
- Meta PurpleLlama / CyberSecEval
- Google BIG-bench and BBEH
- HELM and lm-evaluation-harness (widely-used open evaluation frameworks)

Goal: identify best-in-class structural patterns and map them to a concrete roadmap for GameBench.

## 2. Current GameBench Snapshot

Strengths observed in the current repo:

- Clear layering contracts between game engine, harness, and benchmark orchestration (`AGENTS.md`).
- Registry-based benchmark loading (`games_bench/bench/registry.py`) and suite abstraction (`games_bench/bench/suites.py`).
- Deterministic config merge and suite-first override model (`games_bench/bench/batch.py`).
- Stateful/stateless interaction modes integrated into harness and spec naming (`games_bench/bench/common.py`, `games_bench/llm/harness.py`).
- Useful run artifacts (`episodes.jsonl`, `traces.jsonl`, `summary.json`, optional recordings/raw generations).
- Early-stop semantics (stagnation, deadlock, adapter-triggered termination) and broad unit coverage.
- New cross-game progress reporting with total-episode estimation.

Main structural weaknesses:

- High duplication across game runners (`bench/hanoi.py`, `bench/sokoban.py`) for provider setup, job execution, artifact writing, and metrics aggregation.
- Run manifests are thin: missing standardized lineage metadata (git SHA, package/version snapshot, env fingerprint, canonical prompt/spec fingerprints).
- No first-class resume/checkpoint workflow for interrupted runs.
- Scoring is tightly coupled to execution (limited clean separation of generation phase vs scoring/re-scoring phase).
- No centralized, versioned episode schema contract across games.
- No built-in regression gate workflow (baseline comparison, confidence intervals, and fail-on-regression semantics).
- Limited cross-run analytics/comparison UX (beyond per-run summary files).

## 3. Best-in-Class Patterns from External Projects

### 3.1 Inspect AI (strongest architecture signal)

Notable patterns:

- **Eval sets as execution contracts**: central run config supports retries, sample/task limits, sequencing, model fan-out, and explicit error handling policies.
- **Separation of execution vs scoring**: run without scoring and score later from logs (`--no-score` + `inspect score`) enables grader iteration without rerunning expensive generations.
- **Rich run logs**: dedicated log directories with eval metadata, sample-level events, and provenance fields.
- **Tracing and observability built in**: task events are instrumented and exportable.
- **Model role abstraction** (e.g., grader/solver roles) rather than only a single monolithic model per run.

Why this matters for GameBench: your benchmark is moving toward long-horizon agent evaluation; replayability and auditability of trajectories become critical as costs increase.

### 3.2 OpenAI Evals (API + best-practice docs)

Notable patterns:

- **Explicit data source + testing criteria model** in eval creation.
- **Eval run lifecycle APIs** with status checks and stable identifiers.
- **Best-practice emphasis** on golden sets, continuous monitoring, baseline comparisons, and objective pass/fail criteria before deployment.
- **Agent-eval guidance** focusing on full trajectory quality, not only final outputs.

Why this matters for GameBench: your data is trajectory-native already (`traces.jsonl`); formalizing criteria and baseline comparisons would unlock CI-style benchmark regression checks.

### 3.3 Anthropic evaluation workflow

Notable patterns:

- **Prompt iteration with systematic evaluation loops** and explicit guidance to define success criteria first.
- **Evaluation tools integrated with prompt lifecycle**, making prompt changes measurable and reviewable.

Why this matters for GameBench: you already have prompt variants; adding first-class prompt version IDs and run-to-run comparative reports would align with this loop.

### 3.4 Meta PurpleLlama / CyberSecEval

Notable patterns:

- **Task families + auto-scoring + benchmark packaging** with clear domain framing and open artifacts.
- **Domain-specialized evaluation stacks** (security in their case), where scenario definitions and scoring rules are explicit.

Why this matters for GameBench: long-horizon planning/spatial reasoning can adopt similarly explicit task-family governance (difficulty tiers, failure modes, and scoring contracts).

### 3.5 Google BIG-bench / BBEH

Notable patterns:

- **Large, structured task collections** with explicit task metadata and reproducible benchmarking conventions.
- **Behavior-focused benchmark framing** (BBEH) to expose failure regimes not visible in simple accuracy aggregates.

Why this matters for GameBench: your game tasks should expose behavior/failure taxonomies (deadlock loops, illegal-action bursts, search myopia, horizon collapse), not just solve rate.

### 3.6 HELM + lm-evaluation-harness

Notable patterns:

- **Standardized scenario/task abstraction** with transparent metric reporting (HELM).
- **Extensible task configuration and broad model adapter support** (lm-evaluation-harness), including robust CLI-driven reproducibility.

Why this matters for GameBench: your architecture can stay game-centric while still adopting a standard evaluation object model and comparator APIs.

## 4. Gap Matrix (GameBench vs Best Practice)

| Capability | GameBench Today | Best-in-Class Signal | Priority |
|---|---|---|---|
| Layered architecture | Strong | Inspect/HELM also strong | Keep |
| Deterministic suites | Good | Strong in external frameworks | Keep |
| Generation/scoring separation | Partial | Inspect explicit re-score workflow | High |
| Resume/checkpoint runs | Limited | Inspect/OpenAI run lifecycle patterns | High |
| Run provenance metadata | Basic | Inspect log metadata standards | High |
| Baseline regression gates | Missing | OpenAI eval best-practice loop | High |
| Failure taxonomy metrics | Partial | BBEH-style behavior slices | High |
| Cross-run comparison UX | Limited | Inspect + hosted eval UIs | Medium |
| Prompt version governance | Partial | Anthropic/OpenAI workflow alignment | Medium |
| Unified benchmark core to reduce duplication | Weak | Harness-style cores in mature repos | High |

## 5. Recommended Target Structure for GameBench Bench Layer

Keep AGENTS.md boundaries intact. Refactor only within `games_bench/bench/*` plus benchmark-facing schema modules.

Proposed additions:

- `games_bench/bench/run_manifest.py`
  - Emit canonical metadata per run: timestamp, git SHA, dirty flag, package version, provider/model settings, suite/spec hash, prompt variant hash, seed registry.
- `games_bench/bench/episode_schema.py`
  - Single versioned schema for episode/traces summary fields across all games.
- `games_bench/bench/executor.py`
  - Shared worker/execution engine (parallelism, inflight limits, retries, progress callbacks, checkpoints).
- `games_bench/bench/checkpoint.py`
  - Resume support (idempotent episode IDs + completed set persistence).
- `games_bench/bench/score.py`
  - Offline re-scoring from `traces.jsonl`/`episodes.jsonl` and score-version stamping.
- `games_bench/bench/compare.py`
  - Baseline comparison command with absolute and relative deltas and optional tolerance gates.
- `games_bench/bench/failure_taxonomy.py`
  - Shared behavior tags: deadlock_terminal, stagnation_stop, illegal_burst, horizon_exhausted, query_loop, etc.

Game-specific runner responsibilities should shrink to:

- enumerate episode jobs
- build environment/adapter
- define game-specific metrics derivation hooks
- define default configs and argument extensions

Everything else should be owned by shared bench core.

## 6. Prioritized Roadmap

### Phase 1 (Immediate, highest ROI)

1. Implement **run manifest hardening** (lineage + fingerprints + seeds + git metadata).
2. Extract duplicated execution/artifact logic from Hanoi/Sokoban into shared `bench/executor.py`.
3. Add **compare command** for regression checks across two run directories.
4. Define and persist a **failure taxonomy** in episode summaries.

### Phase 2 (Short-term)

1. Implement **checkpoint + resume** for interrupted long runs.
2. Add **offline scoring command** and score-version metadata.
3. Add statistical reporting (CIs/uncertainty over repeated runs).

### Phase 3 (Medium-term)

1. Add CI automation for benchmark smoke + regression guardrails.
2. Build lightweight HTML report generator for run comparisons.
3. Add multi-judge support for non-exact tasks (if future games need semantic grading).

## 7. What Should Stay Unchanged

- The current layer separation contract in `AGENTS.md`.
- Game adapter protocol as the only LLM-harness contract boundary.
- Config-first suite workflow (`easy-v1`, `standard-v1`) with reproducible defaults.
- Artifact-first reproducibility model (JSONL + summary files written to run directories).

## 8. Bottom Line

Your current architecture is strong for an early-stage benchmark engine and already cleaner than many ad hoc eval repos. The biggest gap is not conceptual; it is **operational maturity**:

- repeatable run lineage,
- replayable/re-scoreable trajectories,
- resume/retry robustness,
- baseline regression governance,
- and reduction of duplicated orchestration code.

Closing those gaps would move GameBench from "capable research harness" to "evaluation platform" while preserving your current engine/harness layering.

## 9. Sources

Primary sources reviewed:

- Inspect AI docs and reference pages:
  - [Inspect overview](https://inspect.aisi.org.uk/)
  - [Inspect eval sets](https://inspect.aisi.org.uk/eval-sets.html)
  - [Inspect scorers](https://inspect.aisi.org.uk/scorers.html)
  - [Inspect tracing](https://inspect.aisi.org.uk/tracing.html)
  - [Inspect log files](https://inspect.aisi.org.uk/log-files.html)
- OpenAI:
  - [OpenAI Evals guide](https://platform.openai.com/docs/guides/evals)
  - [OpenAI Evals API reference](https://platform.openai.com/docs/api-reference/evals)
  - [OpenAI eval design guidance](https://developers.openai.com/tracks/evals)
- Anthropic:
  - [Anthropic prompt engineering: test and evaluate](https://docs.anthropic.com/en/docs/prompt-engineering)
- Meta:
  - [PurpleLlama repository](https://github.com/meta-llama/PurpleLlama)
- Google / DeepMind benchmarks:
  - [BIG-bench repository](https://github.com/google/BIG-bench)
  - [BBEH repository](https://github.com/google-deepmind/bbeh)
- Broad evaluation framework references:
  - [HELM repository](https://github.com/stanford-crfm/helm)
  - [lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness)
