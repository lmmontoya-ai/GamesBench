# Sokoban Development Design (Revised)

## 1. Purpose

Sokoban is the second game environment for `games-bench`. It complements Hanoi by
introducing spatial reasoning, irreversible mistakes, and deceptive local heuristics.

Core benchmark question:

- Does interactive tool-calling (state feedback each step) extend effective planning
  horizon on hard Sokoban levels?

This document is intentionally implementation-oriented and resolves the design gaps found
in prior review.

---

## 2. Resolved Findings and Design Decisions

| ID | Prior gap | Decision in this revision |
|----|-----------|---------------------------|
| F1 | Deadlock semantics were contradictory | Deadlock is a state property; terminal behavior is explicit and variant-dependent via `terminal_on_deadlock` policy. |
| F2 | Level licensing/provenance was unspecified | Add mandatory level manifest with license, source URL, and redistribution flag; no dataset is vendored without explicit legal compatibility. |
| F3 | Optimal-based metrics depended on optional data without policy | Add strict `known_optimal` policy and separate denominators for optimal-based aggregates. |
| F4 | `undo` conflicted with illegal-move accounting | Define illegal actions as blocked movement attempts only; `undo` failures are non-illegal tool errors. Add explicit adapter meta contract. |
| F5 | Proposed recording schema drifted from current infra | Keep core recording stable; define optional Sokoban extension fields and compatibility rules. |
| F6 | Module ownership was inconsistent | Normalize module boundaries and file ownership (env vs loader vs adapter vs deadlock). |
| F7 | `SokobanEnv(level: SokobanLevel | str)` was ambiguous | Constructor accepts `SokobanLevel` only; parsing/loading via explicit factory helpers. |
| F8 | CLI argument naming typo | Standardize on `tool_variants` everywhere (`dest="tool_variants"`). |
| F9 | Tool schemas were too permissive | All schemas use `additionalProperties: false`, explicit `required`, and strict empty-object params for query tools. |
| F10 | Dead-square algorithm was underspecified | Specify reverse-push graph with player stand-behind feasibility to avoid false positives. |

---

## 3. Architecture and Layer Boundaries

Sokoban must follow repository layering rules in `AGENTS.md`:

- Game engine (standalone): `games_bench/games/sokoban/**`
- LLM harness (game-agnostic): `games_bench/llm/**`
- Benchmark orchestration: `games_bench/bench/**`

Non-negotiable constraints:

1. `games_bench/games/sokoban/**` must not import from `bench/` or `llm/`.
2. `games_bench/llm/**` must not import `games_bench.games.sokoban` directly.
3. Registry-driven dispatch is required for benchmark and demo entrypoints.

---

## 4. Task and Episode Semantics

### 4.1 Sokoban rules

1. Player (`@`) moves on a grid with walls (`#`).
2. Moving into a box (`$`) pushes it only if the next cell is free floor or goal.
3. No pulling; no chain pushes.
4. Puzzle is solved when all goals (`.`) are occupied by boxes (`*`).

### 4.2 Deadlock semantics (explicit)

Deadlock is a board property: from current state, puzzle cannot be solved under standard
Sokoban rules.

Deadlock does **not** imply universal immediate termination. Termination is policy-driven:

- `terminal_on_deadlock = true`: episode ends unsolved when deadlock is detected.
- `terminal_on_deadlock = false`: episode continues; deadlock reported in `info`/tool result.

Policy by tool variant:

- `move_only`, `move_and_query`: default `terminal_on_deadlock = true`
- `all_tools` (includes `undo`): default `terminal_on_deadlock = false`

Rationale: `undo` would be pointless if deadlock always hard-terminated.

### 4.3 Step termination conditions

`done = true` when any of:

1. `is_solved() == true`
2. `max_steps` reached
3. `terminal_on_deadlock == true` and deadlock detected
4. `illegal_action_behavior == "terminate"` and illegal action attempted

---

## 5. Data Governance (Licensing, Provenance, Reproducibility)

### 5.1 Mandatory manifest

Each bundled level set must include manifest entries (for example,
`games_bench/games/sokoban/levels/manifest.json`) with:

- `set_name`
- `source_name`
- `source_url`
- `license`
- `license_url`
- `redistribution_allowed` (bool)
- `copyright_notice`
- `downloaded_at`
- `sha256`

### 5.2 Inclusion policy

A level file is vendored only if:

1. License allows redistribution in this repo.
2. Attribution obligations are met.
3. Source checksum and provenance are recorded.

Otherwise, provide a fetch script and require user-side download.

### 5.3 Optimal metadata policy

`optimal_moves` and `optimal_pushes` are optional per level.

Required metadata flags:

- `known_optimal` (bool)
- `optimal_source` (string | null)

Benchmark policy:

- Optimal-ratio metrics (`move_ratio`, `push_ratio`) are computed **only** on
  `known_optimal == true` episodes.
- Summaries must include explicit denominators (e.g., `n_with_optimal_moves`).
- Difficulty plots include an `unknown_optimal` bucket when needed.

---

## 6. Level and State Model

### 6.1 XSB format

Supported symbols:

- `#` wall
- ` ` floor
- `@` player
- `+` player on goal
- `$` box
- `*` box on goal
- `.` goal

### 6.2 Dataclasses

```python
@dataclass(frozen=True, slots=True)
class SokobanLevel:
    level_id: str
    title: str | None
    width: int
    height: int
    xsb: str
    walls: frozenset[tuple[int, int]]
    boxes_start: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    player_start: tuple[int, int]
    n_boxes: int
    optimal_moves: int | None
    optimal_pushes: int | None
    known_optimal: bool

@dataclass(frozen=True, slots=True)
class LevelSet:
    name: str
    description: str
    levels: tuple[SokobanLevel, ...]
    source_name: str
    license: str

@dataclass(frozen=True, slots=True)
class SokobanState:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    boxes: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    player: tuple[int, int]
    n_boxes: int

    def to_dict(self) -> dict[str, Any]: ...
    def to_xsb(self) -> str: ...
```

### 6.3 Constructor API (unambiguous)

`SokobanEnv` constructor accepts only parsed levels:

```python
class SokobanEnv:
    def __init__(self, level: SokobanLevel, *, ...): ...
```

Parsing/loading is explicit via helpers:

```python
def parse_xsb_levels(text: str, *, set_name: str) -> list[SokobanLevel]: ...
def load_level_set(path: str | Path) -> LevelSet: ...
def load_level_by_id(level_id: str) -> SokobanLevel: ...
```

No overloaded `str` meaning in env constructor.

---

## 7. Environment API (`SokobanEnv`)

```python
class SokobanEnv:
    def __init__(
        self,
        level: SokobanLevel,
        *,
        step_penalty: float = 0.0,
        push_reward: float = 0.0,
        push_off_penalty: float = 0.0,
        illegal_move_penalty: float = -1.0,
        solve_reward: float = 1.0,
        deadlock_penalty: float = 0.0,
        illegal_action_behavior: Literal["penalize", "raise", "terminate"] = "penalize",
        max_steps: int | None = None,
        record_history: bool = False,
        detect_deadlocks: bool = True,
        terminal_on_deadlock: bool = True,
    ) -> None: ...

    def reset(self) -> SokobanState: ...
    def get_state(self) -> SokobanState: ...
    def is_solved(self) -> bool: ...
    def is_deadlocked(self) -> bool: ...

    def move(self, direction: str) -> SokobanState: ...
    def step(self, action: str | int) -> tuple[SokobanState, float, bool, dict[str, Any]]: ...
    def undo(self) -> SokobanState: ...

    def get_legal_moves(self) -> list[str]: ...
    def format_prompt_state(
        self,
        *,
        include_legal_moves: bool = False,
        include_deadlock_status: bool = False,
    ) -> str: ...
```

Metric properties:

- `move_count` (walks + pushes)
- `push_count`
- `step_count` (includes illegal attempts)
- `boxes_on_goals`

---

## 8. Deadlock Detection Specification

### 8.1 Dead squares (precomputed)

Algorithm: reverse-push reachability from goals.

A cell `c` is reverse-reachable if there exists a sequence of reverse pushes from any goal
that can place a box on `c`, where each reverse step from `to -> from` requires:

1. `from` is floor/goal and not wall
2. `to` is floor/goal and not wall
3. The player stand-behind cell for that reverse push is traversable in the static board
   model (not a wall)

Cells not reverse-reachable are marked dead squares.

### 8.2 Freeze deadlocks (runtime)

After each push, detect frozen boxes recursively:

- Box is axis-blocked if both adjacent cells on that axis are walls or boxes that are
  themselves axis-blocked.
- Box frozen on both axes and not on goal => deadlock.

### 8.3 False-positive safeguards

- Goals are never dead squares.
- Freeze checks must not treat goal occupancy as automatic deadlock.
- Unit tests include known solvable non-trivial states.

---

## 9. Actions and Illegal-Move Semantics

### 9.1 Action space

```python
ACTION_SPACE: tuple[str, ...] = ("up", "down", "left", "right")
ACTION_INDEX: dict[str, int] = {d: i for i, d in enumerate(ACTION_SPACE)}
```

### 9.2 Illegal action definition

`illegal_moves` means only blocked movement attempts:

- Into wall
- Push into wall
- Push into occupied cell (chain push)

Non-movement tool failures (for example, `undo` with empty history) are **not** illegal
moves.

### 9.3 Reward table

| Event | Reward |
|-------|--------|
| Legal walk/push | `step_penalty` |
| Push box onto goal | `push_reward` |
| Push box off goal | `push_off_penalty` |
| Illegal movement attempt | `illegal_move_penalty` |
| Solved | `solve_reward` |
| Deadlock detected | `deadlock_penalty` |

---

## 10. Tool Interface and Schemas

### 10.1 Tool catalog

- `sokoban_move` (mutating)
- `sokoban_get_state` (query)
- `sokoban_is_solved` (query)
- `sokoban_get_legal_moves` (query)
- `sokoban_undo` (mutating, optional by variant)

### 10.2 Strict schema contract

All tools must use strict JSON schemas.

```python
def tool_schemas(tool_prefix: str = "sokoban") -> list[dict[str, Any]]:
    return [
        {
            "name": f"{tool_prefix}_move",
            "description": "Move player in one direction; may push one box.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    }
                },
                "required": ["direction"],
            },
        },
        {
            "name": f"{tool_prefix}_get_state",
            "description": "Return current state and counters.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
        {
            "name": f"{tool_prefix}_is_solved",
            "description": "Check solved status.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
        {
            "name": f"{tool_prefix}_get_legal_moves",
            "description": "Return legal move directions.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
        {
            "name": f"{tool_prefix}_undo",
            "description": "Undo last move if history exists.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
    ]
```

### 10.3 Tool result contract

Movement tool returns include:

- `ok`
- `action_type`: `"walk" | "push"`
- `direction`
- `state` (serialized)
- `boxes_on_goals`
- `deadlocked` (if detection enabled)

Query tool returns include `ok` and relevant fields only.

`undo` no-history response:

- `ok: false`
- `error: "cannot undo: no history"`
- `state` unchanged

### 10.4 Adapter meta contract (for harness + recording)

`ToolExecution.meta` for Sokoban must include:

- `state_mutating`: bool
- `illegal_action`: bool (true only for illegal movement attempts)
- `action_kind`: `"move" | "query" | "undo"`
- `counts_as_move`: bool

Rules:

- `sokoban_move` legal walk/push: `counts_as_move=true`
- `sokoban_move` illegal attempt: `illegal_action=true`, `counts_as_move=false`
- `sokoban_undo` success/failure: `illegal_action=false`, `counts_as_move=false`
- Query tools: both false

This preserves clean illegal-move metrics and avoids counting undo as forward move.

### 10.5 Tool variants

| Variant | Allowed tools | Defaults |
|---------|----------------|----------|
| `move_only` | `sokoban_move` | `terminal_on_deadlock=true` |
| `move_and_query` | `move`, `get_state`, `is_solved`, `get_legal_moves` | `terminal_on_deadlock=true` |
| `all_tools` | all above + `undo` | `terminal_on_deadlock=false` |

---

## 11. Adapter (`SokobanGameAdapter`)

`SokobanGameAdapter` implements `games_bench.games.adapter.GameAdapter`.

```python
class SokobanGameAdapter:
    def __init__(self, env: SokobanEnv, *, tool_prefix: str = "sokoban", instructions: str | None = None) -> None: ...
    def tool_schemas(self) -> list[dict[str, Any]]: ...
    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecution: ...
    def get_state_snapshot(self) -> dict[str, Any]: ...
    def is_solved(self) -> bool: ...
    def default_instructions(self) -> str: ...
    def format_state(self) -> str: ...
    def episode_metrics(self) -> dict[str, Any]: ...
```

Required episode metrics:

- `level_id`, `n_boxes`, `grid_size`
- `move_count`, `push_count`, `boxes_on_goals`, `deadlocked`
- `optimal_moves`, `optimal_pushes`, `known_optimal`
- `history` (when recorded)

---

## 12. Prompt Design

`games_bench/games/sokoban/prompts/`:

- `default.txt`
- `image_suffix.txt`

Prompt variants:

- `minimal`
- `with_legal_moves`
- `with_deadlock_warnings`
- `full`

Prompt policy:

1. Always require exactly one tool call per turn.
2. Keep board symbols stable and explicit.
3. For image mode, keep symbol legend synchronized with renderer colors.

---

## 13. Vision / Rendering

`vision.py` provides:

```python
def render_sokoban_image(state: SokobanState, *, tile_size: int = 48, label_grid: bool = True, background: str = "white") -> StateImage: ...
def render_sokoban_state_image(state: SokobanState, **kwargs) -> StateImage: ...
def render_sokoban_env_image(env: SokobanEnv, **kwargs) -> StateImage: ...
```

Constraints:

- Lazy PIL import with actionable error guidance (`games-bench[viz]` / `uv sync --group viz`).
- Deterministic rendering for regression tests.

---

## 14. Benchmark Runner (`bench/sokoban.py`)

### 14.1 Default config

```python
def default_sokoban_config() -> dict[str, Any]:
    return {
        "level_sets": ["microban"],
        "level_ids": None,
        "max_levels": 20,
        "difficulty_filter": None,
        "runs_per_level": 1,
        "max_turns": 300,
        "prompt_variants": ["minimal"],
        "tool_variants": ["move_only"],
        "state_format": "text",
        "image_tile_size": 48,
        "image_labels": True,
        "image_background": "white",
        "detect_deadlocks": True,
        "terminal_on_deadlock": True,
        "record": False,
        "record_raw": False,
        "record_provider_raw": False,
    }
```

### 14.2 Game-specific CLI args

```python
def add_sokoban_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level-set", action="append")
    parser.add_argument("--level-id", action="append")
    parser.add_argument("--max-levels", type=int)
    parser.add_argument("--max-optimal-moves", type=int)
    parser.add_argument("--prompt-variant", action="append", dest="prompt_variants")
    parser.add_argument("--tool-variant", action="append", dest="tool_variants")
    parser.add_argument("--runs-per-level", type=int)
    parser.add_argument("--state-format", choices=["text", "image", "both"])
    parser.add_argument("--detect-deadlocks", action=argparse.BooleanOptionalAction)
    parser.add_argument("--terminal-on-deadlock", action=argparse.BooleanOptionalAction)
```

### 14.3 Merge precedence

Must follow repository contract:

`BenchSpec.default_config() < global config < per-game overrides`

---

## 15. Metrics and Reporting

### 15.1 Per-episode fields

- `solved`
- `deadlocked`
- `move_count`
- `push_count`
- `illegal_moves`
- `tool_calls`
- `boxes_on_goals`
- `boxes_on_goals_ratio`
- `optimal_moves`, `optimal_pushes`, `known_optimal`
- `move_ratio`, `push_ratio` (only when denominator available)

### 15.2 Aggregate metrics

Include explicit denominators:

- `avg_move_ratio` + `n_with_optimal_moves`
- `avg_push_ratio` + `n_with_optimal_pushes`

Difficulty analysis:

- Primary: optimal-move buckets for known-optimal episodes
- Secondary: `unknown_optimal` bucket using structural descriptors (`n_boxes`, grid area)

---

## 16. Recording Contract (Compatibility-First)

### 16.1 Core compatibility

Keep base recording structure unchanged:

- `metadata`
- `summary`
- `steps`

### 16.2 Optional Sokoban extensions

Allowed additive fields:

- `summary.total_pushes` (optional)
- `steps[].action_type` (`walk`/`push`/`undo`) (optional)

Rules:

1. Existing readers must continue working if extension fields are absent.
2. Sokoban render/review tools may consume extension fields opportunistically.
3. Authoritative push counts remain in episode metrics (`push_count`) even if recording
   extension is disabled.

### 16.3 Generic integration note

To support precise game-specific counting, recording logic should prioritize
`ToolExecution.meta` keys (`counts_as_move`, `action_kind`) over tool-name heuristics.

---

## 17. File Structure (Consistent Ownership)

```text
games_bench/games/sokoban/
├── __init__.py
├── env.py               # SokobanState, SokobanEnv, SokobanToolbox, tool_schemas
├── adapter.py           # SokobanGameAdapter
├── level_loader.py      # XSB parser, level-set loader, manifest loader
├── deadlock.py          # dead-square + freeze logic
├── vision.py            # image rendering
├── render.py            # playback renderer
├── review.py            # manual review bundle
├── prompts/
│   ├── __init__.py
│   ├── default.txt
│   └── image_suffix.txt
└── levels/
    ├── manifest.json
    ├── *.xsb
    └── metadata.json

games_bench/bench/sokoban.py
```

`tool_schemas.py` is intentionally **not** split out in v1 to avoid unnecessary surface area
and to mirror existing Hanoi conventions.

---

## 18. Registry Wiring

Game registry:

```python
from games_bench.games.sokoban import SokobanEnv
register_game(GameSpec(name="sokoban", description="Sokoban", env_factory=SokobanEnv))
```

Benchmark registry:

```python
register_benchmark(BenchSpec(
    name="sokoban",
    description="Sokoban",
    batch_runner=sokoban_bench.run_batch,
    add_arguments=sokoban_bench.add_sokoban_arguments,
    default_config=sokoban_bench.default_sokoban_config,
    adapter_factory=sokoban_bench.build_sokoban_adapter,
    render_main=sokoban_render.main,
    review_main=sokoban_review.main,
))
```

---

## 19. Development Phases and Quality Gates

### Phase 0: Governance and fixtures

Deliverables:

1. Manifest format + validation utility
2. Initial legally approved level subset
3. Metadata policy (`known_optimal`, sources)

Gate:

- No unlicensed datasets in tree.

### Phase 1: Core environment

Deliverables:

1. `SokobanLevel`, `SokobanState`
2. Level loading/parsing
3. `SokobanEnv` movement, step, rewards, undo
4. `SokobanToolbox`, strict tool schemas

Tests:

- Parse edge cases
- Movement/push legality
- Solve detection
- Undo semantics
- Illegal-action behavior

Gate:

- Standalone game tests pass with zero `bench`/`llm` imports.

### Phase 2: Deadlock detection

Deliverables:

1. Dead-square precompute
2. Freeze deadlock checks
3. Env integration and deadlock policy flags

Tests:

- Dead corners/edges
- Freeze deadlocks
- False-positive controls

Gate:

- Deadlock detection precision acceptable on curated test fixtures.

### Phase 3: Adapter + prompts

Deliverables:

1. `SokobanGameAdapter`
2. Prompt variants
3. Episode metrics contract

Tests:

- Tool routing
- Meta correctness (`illegal_action`, `counts_as_move`)
- Prompt formatting

Gate:

- Harness integration works without game-specific imports in `llm/`.

### Phase 4: Vision

Deliverables:

1. PIL renderer
2. State/env image wrappers
3. Image prompt suffix

Tests:

- Dimensions/type checks
- Basic visual regression snapshots

### Phase 5: Benchmark runner integration

Deliverables:

1. `bench/sokoban.py`
2. Registry wiring
3. Config/CLI support
4. Aggregates with denominator-aware optimal metrics

Tests:

- Config precedence
- Args namespace immutability
- `run sokoban --help`
- Config-driven mode

### Phase 6: Render/review integration

Deliverables:

1. Sokoban render/review entrypoints
2. Optional recording extensions support

Tests:

- Playback generation
- Review bundle generation

### Phase 7: Expansion and docs

Deliverables:

1. Additional approved level sets
2. README/AGENTS/CLAUDE updates
3. Example `configs/sokoban.json`

Gate:

- Full repo tests pass; no layering regressions.

---

## 20. Definition of Done (Sokoban v1)

Sokoban v1 is done when all are true:

1. Level data in repo satisfies license/provenance requirements.
2. Game layer is standalone and fully tested.
3. Adapter integrates through generic harness without special-case `llm` logic.
4. Benchmark outputs include denominator-aware optimal metrics.
5. Deadlock/undo/illegal semantics are implemented exactly as specified.
6. Render/review tooling works with recording compatibility guarantees.
