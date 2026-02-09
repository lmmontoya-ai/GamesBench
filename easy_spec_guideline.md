# easy Spec Guideline

This guideline defines how to add or update the `easy` benchmark family (for example `easy-v1`) for smaller-capability LLMs.

## Goal

`easy` suites should still measure planning and spatial reasoning, but at a difficulty where smaller models can make measurable progress.

## Scope and principles

- Keep tasks planning-heavy, but shorten horizon and reduce branching compared to `standard`.
- Keep the suite reproducible and deterministic.
- Keep the config stable after release; create a new version for any behavioral change.

## Global config expectations

- Include global runtime and logging keys:
  - `spec`, `out_dir`, `record`, `record_raw`, `record_provider_raw`
  - `provider_retries`, `provider_backoff`, `stream_debug`
  - `parallelism`, `max_inflight_provider`
  - Optional `stateless` (default is stateful when omitted)
- Prefer lower run cost than `standard`:
  - Lower concurrency defaults and smaller run counts are acceptable.

## Per-game design rules

For each game added under `games.<game_name>`:

- Use fixed, explicit difficulty sets (no hidden defaults).
- Prefer easier prompt/tool variants that reduce avoidable failures.
- Include early-stop controls (`stagnation_patience`, deadlock controls where relevant).
- Keep deterministic seeds for generated content.

## Difficulty targeting

- Include low-to-mid difficulty tiers only.
- Avoid extreme horizons that belong in `standard`.
- Ensure at least two distinct difficulty tiers per game so results are not trivial.

## Statistical guidance

- Use fewer repeats than `standard`, but keep repeats >1 where possible.
- Keep enough episodes for stable trend comparisons across models.

## New game checklist for easy suites

When adding a new game to `easy`:

1. Add a fixed easy ladder (or fixed seeded procedural cases).
2. Pick prompt/tool variants that emphasize solvability and diagnosis.
3. Add explicit termination controls and deterministic seeds.
4. Verify the suite runs via `games-bench run --suite easy-v1 --game <game>`.
5. Add/extend tests for suite registration and effective config content.

## Versioning policy

- Do not mutate old suite versions.
- For any difficulty, seed, or variant changes, publish a new version (for example `easy-v2`).
- Keep `spec` stable for a version (`easy-v1`), and rely on runtime suffixing to report
  `easy-v1-stateful` vs `easy-v1-stateless`.
