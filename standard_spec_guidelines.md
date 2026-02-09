# standard Spec Guidelines

This guideline defines how to add or update the `standard` benchmark family (for example `standard-v1`) used as the canonical comparison suite.

## Goal

`standard` suites are the primary reproducible benchmark for cross-model comparison on long-horizon planning and spatial reasoning.

## Scope and principles

- Prioritize robust cross-run comparability over convenience.
- Include challenging horizon lengths and richer failure modes.
- Keep suite behavior deterministic and versioned.

## Global config expectations

- Include global runtime and logging keys:
  - `out_dir`, `record`, `record_raw`, `record_provider_raw`
  - `provider_retries`, `provider_backoff`, `stream_debug`
  - `parallelism`, `max_inflight_provider`
- Defaults should support stable benchmark throughput, not minimum cost.

## Per-game design rules

For each game added under `games.<game_name>`:

- Use a fixed, explicit difficulty ladder spanning multiple challenge tiers.
- Include at least one genuinely long-horizon tier.
- Use deterministic seeds for any procedural generation.
- Keep prompt/tool variants narrow and intentional to avoid benchmark inflation.

## Difficulty targeting

- Cover easy, medium, and hard tiers inside the same game ladder.
- Include irreversible error regimes where the game supports them.
- Avoid over-indexing on only toy levels.

## Statistical guidance

- Use repeated runs sufficient for stable aggregate metrics.
- Prefer conservative repeat counts for canonical reporting.
- Keep denominators explicit when metrics rely on optional metadata.

## New game checklist for standard suites

When adding a new game to `standard`:

1. Add a fixed multi-tier ladder with at least one long-horizon regime.
2. Set deterministic seeds and explicit procedural parameters.
3. Configure failure-aware early-stop controls where applicable.
4. Keep tool/prompt variants benchmark-faithful (no extra helper leakage).
5. Verify `games-bench run --suite standard-v1 --game <game>` end-to-end.
6. Add/extend tests for suite registration and expected effective config.

## Versioning policy

- Treat released `standard-v*` suites as immutable.
- Introduce `standard-v{n+1}` for any ladder, seed, or variant changes.
