# Sokoban Development Design

## 1. Overview

Sokoban is the second game environment for games-bench, chosen because it maximally
benefits from external tool-calling: simulating a 2D grid with box physics mentally is
extremely hard, but with environment feedback a model can focus on strategy. Irreversible
actions (pushing a box into a corner) create genuine planning pressure absent from Hanoi.

### Relationship to prior work

**SokoBench** (arXiv 2601.20856) evaluated LLMs on Sokoban but used 1D linear corridors
with a single box, no environmental feedback, and required models to output complete
action sequences up front. They found performance collapse beyond ~25 moves.

Our approach is fundamentally different:
- Full 2D boards with multiple boxes (real Sokoban, not simplified corridors)
- **Interactive tool-calling**: model receives updated state after each action
- Curated level sets with known optimal solutions for quantitative evaluation
- Multimodal support (text + image state)
- Deadlock detection to provide richer feedback

This lets us test whether external state feedback shifts the planning horizon beyond
the ~25-move ceiling SokoBench observed.

---

## 2. Game rules

1. The player (`@`) moves on a 2D grid with walls (`#`).
2. The player can push one box (`$`) by moving into it, if the cell behind the box
   is empty floor or a goal square.
3. Only one box can be pushed at a time (no chain pushing).
4. Boxes cannot be pulled — pushes are irreversible.
5. The puzzle is solved when every goal (`.`) has a box on it (`*`).
6. A puzzle becomes unsolvable when a box reaches a dead position (deadlock).

---

## 3. State representation

### 3.1 XSB format (standard)

The XSB format is the universal standard for Sokoban level representation:

| Character | Meaning        |
|-----------|----------------|
| `#`       | Wall           |
| ` ` (space) | Empty floor  |
| `@`       | Player         |
| `+`       | Player on goal |
| `$`       | Box            |
| `*`       | Box on goal    |
| `.`       | Goal           |

Example (Microban level 1):

```
####
# .#
#  ###
#*@  #
#  $ #
#  ###
####
```

### 3.2 Internal state (`SokobanState` dataclass)

```python
@dataclass(frozen=True, slots=True)
class SokobanState:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]       # (row, col) set
    boxes: frozenset[tuple[int, int]]        # current box positions
    goals: frozenset[tuple[int, int]]        # target positions (static)
    player: tuple[int, int]                  # (row, col)
    n_boxes: int                             # len(goals), convenience

    def to_dict(self) -> dict[str, Any]: ...
    def to_xsb(self) -> str: ...            # render as XSB string
```

Design notes:
- `frozenset` for hashability (enables visited-state tracking, deadlock caching).
- Walls and goals are static per level; boxes and player change per step.
- Coordinate system: `(row, col)`, origin at top-left `(0, 0)`.
- `to_xsb()` produces the standard text rendering — this is the primary format
  passed to LLMs as state text.

### 3.3 Text state for LLMs

The XSB grid string is compact and unambiguous. Example output of `format_state()`:

```
Board (7x6):
####
# .#
#  ###
#*@  #
#  $ #
#  ###
####

Boxes on goals: 1/2
```

This mirrors how humans read Sokoban — the same format used in every Sokoban program,
tutorial, and wiki. No translation overhead for the model.

### 3.4 Image state for multimodal models

PIL-rendered top-down grid view:
- Each cell is a fixed-size tile (e.g., 48x48 pixels)
- Color scheme: dark gray walls, light floor, red/orange boxes, green goals,
  yellow box-on-goal, blue player
- Row/column labels on edges for spatial reference
- Configurable: tile size, with/without labels, background color

The rendering follows the same pattern as `render_hanoi_image()` → returns `StateImage`
(dataclass with `data_base64`, `data_url`, `width`, `height`, `mime_type`).

---

## 4. Action space

### 4.1 Core actions

Four directional moves: `up`, `down`, `left`, `right`.

A move is a **push** if the target cell contains a box and the cell beyond it is free.
Otherwise it is a **walk** (player moves without pushing). The environment tracks both
counts separately.

```python
Direction: TypeAlias = Literal["up", "down", "left", "right"]

DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}
```

### 4.2 RL interface

```python
ACTION_SPACE: tuple[str, ...] = ("up", "down", "left", "right")
ACTION_INDEX: dict[str, int] = {d: i for i, d in enumerate(ACTION_SPACE)}
```

`step(action)` accepts either a string direction or an integer index (0-3).

### 4.3 Illegal vs. impossible moves

| Situation | Behavior |
|-----------|----------|
| Walk into wall | Illegal: penalized, state unchanged |
| Push box into wall | Illegal: penalized, state unchanged |
| Push two boxes (chain push) | Illegal: penalized, state unchanged |
| Walk into empty cell | Legal walk: state updated |
| Push box into empty cell/goal | Legal push: state updated |

Configurable via `illegal_action_behavior`: `"penalize"` (default), `"raise"`, `"terminate"`.

---

## 5. Tools (tool-calling interface)

### 5.1 Tool catalog

| Tool | Mutating | Description |
|------|----------|-------------|
| `sokoban_move` | Yes | Move player in a direction (may push a box) |
| `sokoban_get_state` | No | Return current board as XSB string + metadata |
| `sokoban_is_solved` | No | Check if all goals have boxes |
| `sokoban_get_legal_moves` | No | Return list of legal directions |
| `sokoban_undo` | Yes | Undo last move (optional, for tool variants) |

### 5.2 Tool schemas

```python
def tool_schemas(tool_prefix: str = "sokoban") -> list[dict[str, Any]]:
    return [
        {
            "name": f"{tool_prefix}_move",
            "description": "Move the player in the given direction. "
                           "If moving into a box with an empty cell behind it, "
                           "the box is pushed. Returns the updated board state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to move"
                    }
                },
                "required": ["direction"]
            }
        },
        {
            "name": f"{tool_prefix}_get_state",
            "description": "Return the current board state as an ASCII grid, "
                           "the number of boxes on goals, and whether the puzzle "
                           "is solved.",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": f"{tool_prefix}_is_solved",
            "description": "Check whether all goals have a box on them.",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": f"{tool_prefix}_get_legal_moves",
            "description": "Return a list of directions the player can move "
                           "(excluding moves into walls or chain pushes).",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": f"{tool_prefix}_undo",
            "description": "Undo the last move. Returns the restored board state. "
                           "Cannot undo past the initial state.",
            "parameters": {"type": "object", "properties": {}}
        },
    ]
```

### 5.3 Tool result format

Consistent with Hanoi's `{"ok": bool, ...}` pattern:

```python
# Successful move
{"ok": True, "action": "push", "direction": "right",
 "state": "<xsb string>", "boxes_on_goals": 1, "total_goals": 2}

# Illegal move
{"ok": False, "error": "Cannot push: wall behind box",
 "state": "<unchanged xsb>"}

# Deadlock detected (informational, not terminal)
{"ok": True, "action": "push", "direction": "down",
 "state": "<xsb>", "boxes_on_goals": 0, "total_goals": 2,
 "warning": "Deadlock detected: box at (3,1) is stuck"}
```

### 5.4 Tool variants

| Variant | Tools allowed | Tests |
|---------|---------------|-------|
| `move_only` | `sokoban_move` | Raw planning with only move feedback |
| `move_and_query` | `move`, `get_state`, `is_solved`, `get_legal_moves` | Planning with state introspection |
| `all_tools` | All including `undo` | Planning with backtracking |

---

## 6. Level management

### 6.1 Level loading

Levels are loaded from XSB text files. Each file may contain multiple levels separated
by blank lines, with optional title/comment lines (lines starting with `;`).

```python
def parse_xsb(text: str) -> list[SokobanLevel]: ...
def load_level_set(path: str | Path) -> LevelSet: ...
```

### 6.2 `SokobanLevel` and `LevelSet`

```python
@dataclass(frozen=True, slots=True)
class SokobanLevel:
    level_id: str                  # "{set_name}:{index}" e.g. "microban:1"
    title: str | None              # optional level title
    width: int
    height: int
    xsb: str                       # canonical XSB text
    walls: frozenset[tuple[int, int]]
    boxes_start: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    player_start: tuple[int, int]
    n_boxes: int
    optimal_moves: int | None       # known optimal move count
    optimal_pushes: int | None      # known optimal push count

@dataclass(frozen=True, slots=True)
class LevelSet:
    name: str
    description: str
    levels: tuple[SokobanLevel, ...]
    difficulty: str                 # "easy", "medium", "hard"
```

### 6.3 Bundled level sets

Ship a curated selection of freely-available levels in
`games_bench/games/sokoban/levels/`:

| Set | Levels | Boxes | Grid | Difficulty | Source |
|-----|--------|-------|------|------------|--------|
| Microban | 155 | 1-6 | 5x5-10x10 | Easy | David W. Skinner |
| Microban II | 135 | 1-8 | 5x5-12x12 | Easy-Medium | David W. Skinner |
| Sasquatch | 50 | 3-8 | 8x8-15x15 | Medium | David W. Skinner |
| Original | 50 | 3-6 | ~10x10 | Medium-Hard | Thinking Rabbit |
| Boxoban-easy | 100 (curated) | 4 | 10x10 | Easy | DeepMind |
| Boxoban-medium | 100 (curated) | 4 | 10x10 | Medium | DeepMind |
| Boxoban-hard | 100 (curated) | 4 | 10x10 | Hard | DeepMind |

Level files are plain `.xsb` text with a companion `metadata.json` containing known
optimal solutions where available.

### 6.4 Difficulty dimensions

| Parameter | Range | Effect |
|-----------|-------|--------|
| Grid size | 5x5 to 15x15 | Larger grids → longer paths, more dead squares |
| Number of boxes | 1 to 8 | More boxes → exponential state space growth |
| Optimal solution length | 5 to 200+ moves | Direct measure of planning horizon |
| Deadlock density | Low to high | More dead squares → more irreversible traps |
| Level set | Microban → Original | Curated difficulty progression |

The primary benchmark axis is **optimal solution length** (number of pushes or moves),
since this directly maps to long-horizon planning capacity — the core metric we want
to measure against the Illusion of Thinking findings.

---

## 7. Environment design (`SokobanEnv`)

### 7.1 Constructor

```python
class SokobanEnv:
    def __init__(
        self,
        level: SokobanLevel | str,         # level object or XSB string
        *,
        step_penalty: float = 0.0,
        push_reward: float = 0.0,           # reward for pushing box onto goal
        push_off_penalty: float = 0.0,      # penalty for pushing box off goal
        illegal_move_penalty: float = -1.0,
        solve_reward: float = 1.0,
        deadlock_penalty: float = 0.0,      # penalty when deadlock detected
        illegal_action_behavior: Literal["penalize", "raise", "terminate"] = "penalize",
        max_steps: int | None = None,
        record_history: bool = False,
        detect_deadlocks: bool = True,
    ) -> None: ...
```

### 7.2 Core methods

```python
# === State ===
def reset(self) -> SokobanState: ...
def get_state(self) -> SokobanState: ...
def is_solved(self) -> bool: ...
def is_deadlocked(self) -> bool: ...

# === Actions ===
def move(self, direction: str) -> SokobanState:
    """Apply move. Raises IllegalMoveError if illegal."""
def step(self, action: str | int) -> tuple[SokobanState, float, bool, dict]:
    """RL interface. Returns (state, reward, done, info)."""
def undo(self) -> SokobanState:
    """Undo last move. Raises if no history."""

# === Query ===
def get_legal_moves(self) -> list[str]: ...
def format_prompt_state(self) -> str: ...

# === Metrics ===
@property
def move_count(self) -> int: ...    # total moves (walks + pushes)
@property
def push_count(self) -> int: ...    # pushes only
@property
def step_count(self) -> int: ...    # step() calls (including illegal)
@property
def boxes_on_goals(self) -> int: ...
```

### 7.3 Reward function

| Event | Reward |
|-------|--------|
| Legal step (walk or push) | `step_penalty` (typically 0 or small negative) |
| Push box onto goal | `push_reward` |
| Push box off goal | `push_off_penalty` |
| Illegal move | `illegal_move_penalty` |
| Puzzle solved | `solve_reward` |
| Deadlock reached | `deadlock_penalty` |

### 7.4 Deadlock detection

Two levels of deadlock detection, both precomputed or cheap at runtime:

**Simple dead squares** (precomputed per level):
A cell is dead if a box placed there can never reach any goal. Detected by
reverse-reachability: flood-fill from each goal in reverse-push directions.
Any cell not reached is dead.

**Freeze deadlocks** (checked after each push):
A box is frozen if it cannot move along either axis. Checked recursively:
a box is frozen on an axis if both neighbors on that axis are walls or frozen
boxes. If a frozen box is not on a goal, it's a deadlock.

These two cover the vast majority of detectable deadlocks without expensive
computation. Bipartite matching (checking whether all boxes can simultaneously
reach distinct goals) is deferred to a future enhancement.

### 7.5 Exception hierarchy

```python
class SokobanError(Exception): ...
class InvalidLevelError(SokobanError, ValueError): ...
class IllegalMoveError(SokobanError): ...
```

### 7.6 `SokobanToolbox`

Wraps `SokobanEnv` methods, catches exceptions, returns `{"ok": bool, ...}` dicts.
Same pattern as `HanoiToolbox`.

```python
class SokobanToolbox:
    def __init__(self, env: SokobanEnv) -> None: ...

    def move(self, direction: str) -> dict[str, Any]: ...
    def get_state(self) -> dict[str, Any]: ...
    def is_solved(self) -> dict[str, Any]: ...
    def get_legal_moves(self) -> dict[str, Any]: ...
    def undo(self) -> dict[str, Any]: ...
```

---

## 8. Adapter design (`SokobanGameAdapter`)

Implements `GameAdapter` protocol. Mirrors `HanoiGameAdapter` structure exactly.

```python
class SokobanGameAdapter:
    def __init__(
        self,
        env: SokobanEnv,
        *,
        tool_prefix: str = "sokoban",
        instructions: str | None = None,
    ) -> None: ...

    def tool_schemas(self) -> list[dict[str, Any]]: ...

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        # Routes to toolbox methods
        # meta: {"state_mutating": bool, "illegal_action": bool}
        # sokoban_move → mutating=True, illegal if result["ok"] is False
        # sokoban_undo → mutating=True
        # sokoban_get_state, is_solved, get_legal_moves → mutating=False

    def get_state_snapshot(self) -> dict[str, Any]: ...
    def is_solved(self) -> bool: ...
    def default_instructions(self) -> str: ...
    def format_state(self) -> str: ...

    def episode_metrics(self) -> dict[str, Any]:
        return {
            "level_id": self.env.level.level_id,
            "n_boxes": self.env.level.n_boxes,
            "grid_size": f"{self.env.level.width}x{self.env.level.height}",
            "move_count": self.env.move_count,
            "push_count": self.env.push_count,
            "optimal_moves": self.env.level.optimal_moves,
            "optimal_pushes": self.env.level.optimal_pushes,
            "boxes_on_goals": self.env.boxes_on_goals,
            "deadlocked": self.env.is_deadlocked(),
            "history": self.env.history if self.env.record_history else None,
        }
```

---

## 9. Prompts

### 9.1 Prompt templates (`games_bench/games/sokoban/prompts/`)

**`default.txt`:**

```
You are solving a Sokoban puzzle.
Goal: push all boxes ($) onto the goal squares (.).
- You move with: up, down, left, right.
- Walking into a box pushes it if the cell behind it is empty.
- You cannot push two boxes at once.
- You cannot pull boxes — pushes are permanent.
- Walls (#) block movement.
Board symbols: # wall, @ you, $ box, . goal, * box on goal, + you on goal.
Call exactly one tool per turn.
```

**`image_suffix.txt`:**

```
The current state is provided as an image. The grid shows walls (dark gray),
floor (light), boxes (red/orange), goals (green circles), boxes on goals
(yellow), and the player (blue). Row/column indices are labeled on the edges.
```

### 9.2 Prompt variants

| Variant | Description |
|---------|-------------|
| `minimal` | Default instructions, no legal moves, no deadlock info |
| `with_legal_moves` | Includes legal move directions in state |
| `with_deadlock_warnings` | Mentions deadlock risk in instructions |
| `full` | All hints: legal moves + deadlock warnings |

---

## 10. Vision / image rendering

### 10.1 Grid renderer

```python
def render_sokoban_image(
    state: SokobanState,
    *,
    tile_size: int = 48,
    label_grid: bool = True,
    background: str = "white",
) -> StateImage:
```

Tile-based rendering using PIL:

| Element | Color | Shape |
|---------|-------|-------|
| Wall | Dark gray `#404040` | Filled square |
| Floor | Light gray `#E8E8E8` | Filled square |
| Goal | Green `#4CAF50` | Circle/diamond on floor |
| Box | Red/orange `#E65100` | Rounded square |
| Box on goal | Yellow/gold `#FFC107` | Rounded square |
| Player | Blue `#1565C0` | Circle |
| Player on goal | Blue `#1565C0` | Circle on green |

Grid lines separate cells. Optional row/column labels on edges for spatial reference
(important for multimodal models to describe positions).

### 10.2 Integration with harness

Same pattern as Hanoi:
```python
def render_sokoban_state_image(state, **kwargs) -> StateImage: ...
def render_sokoban_env_image(env, **kwargs) -> StateImage: ...
```

Returns `StateImage(mime_type, data_base64, data_url, width, height)`.

---

## 11. Metrics

### 11.1 Per-episode metrics (from adapter)

| Metric | Type | Description |
|--------|------|-------------|
| `level_id` | str | Level identifier (e.g., "microban:1") |
| `n_boxes` | int | Number of boxes in the level |
| `grid_size` | str | "WxH" grid dimensions |
| `solved` | bool | All boxes on goals |
| `move_count` | int | Total moves (walks + pushes) |
| `push_count` | int | Pushes only |
| `optimal_moves` | int\|None | Known optimal move count |
| `optimal_pushes` | int\|None | Known optimal push count |
| `move_ratio` | float\|None | `move_count / optimal_moves` (efficiency) |
| `push_ratio` | float\|None | `push_count / optimal_pushes` |
| `boxes_on_goals` | int | Boxes placed at episode end |
| `boxes_on_goals_ratio` | float | `boxes_on_goals / n_boxes` (partial progress) |
| `deadlocked` | bool | Reached unrecoverable state |
| `illegal_moves` | int | Attempted illegal moves (from harness) |
| `tool_calls` | int | Total tool invocations (from harness) |

### 11.2 Aggregate metrics (batch runner)

| Metric | Description |
|--------|-------------|
| `episodes` | Total episodes run |
| `solved` / `solve_rate` | Count and fraction solved |
| `deadlocked` / `deadlock_rate` | Count and fraction that deadlocked |
| `avg_moves` | Mean moves per episode |
| `avg_pushes` | Mean pushes per episode |
| `avg_move_ratio` | Mean move efficiency (solved episodes only) |
| `avg_push_ratio` | Mean push efficiency (solved episodes only) |
| `avg_boxes_placed` | Mean boxes on goals at episode end |
| `avg_illegal_moves` | Mean illegal move attempts |
| `avg_tool_calls` | Mean tool calls per episode |
| `solve_rate_by_difficulty` | Solve rate bucketed by optimal solution length |
| `token_totals` / `cost_total` | Resource usage |

### 11.3 Difficulty-stratified analysis

Group results by optimal solution length buckets to produce a planning-horizon
curve (analogous to the Apple paper's complexity scaling plots):

| Bucket | Optimal moves |
|--------|---------------|
| Trivial | 1-10 |
| Easy | 11-25 |
| Medium | 26-50 |
| Hard | 51-100 |
| Expert | 100+ |

This directly tests whether tool-calling shifts the performance cliff observed
in the Illusion of Thinking and SokoBench papers.

---

## 12. Benchmark runner (`bench/sokoban.py`)

### 12.1 Configuration

```python
def default_sokoban_config() -> dict[str, Any]:
    return {
        "level_sets": ["microban"],
        "level_ids": None,               # None = all levels in the set
        "max_levels": 20,                 # cap for large sets
        "difficulty_filter": None,        # e.g., {"max_optimal_moves": 50}
        "runs_per_level": 1,
        "max_turns": 300,
        "prompt_variants": ["minimal"],
        "tool_variants": ["move_only"],
        "state_format": "text",
        "image_tile_size": 48,
        "image_labels": True,
        "image_background": "white",
        "detect_deadlocks": True,
        "record": False,
        "record_raw": False,
        "record_provider_raw": False,
    }
```

### 12.2 CLI arguments (game-specific)

```python
def add_sokoban_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level-set", action="append")
    parser.add_argument("--level-id", action="append")
    parser.add_argument("--max-levels", type=int)
    parser.add_argument("--max-optimal-moves", type=int)
    parser.add_argument("--prompt-variant", action="append")
    parser.add_argument("--tool-variant", action="append", dest="tools_variant")
    parser.add_argument("--runs-per-level", type=int)
    parser.add_argument("--state-format", choices=["text", "image", "both"])
    parser.add_argument("--detect-deadlocks", action=argparse.BooleanOptionalAction)
```

### 12.3 Batch loop structure

```
for model in models:
    provider = build_provider(model, ...)
    for level_set in level_sets:
        for level in filtered_levels(level_set):
            for prompt_variant in prompt_variants:
                for tool_variant in tool_variants:
                    for run in range(runs_per_level):
                        env = SokobanEnv(level, ...)
                        adapter = SokobanGameAdapter(env, ...)
                        result = run_tool_calling_episode(adapter, provider, ...)
                        write_episode(result)
    write_summary(all_results)
```

### 12.4 Config file format

```json
{
  "models": {"openrouter": ["openai/gpt-4.1-mini"]},
  "out_dir": "artifacts/runs",
  "games": {
    "sokoban": {
      "level_sets": ["microban", "boxoban-easy"],
      "max_levels": 50,
      "difficulty_filter": {"max_optimal_moves": 100},
      "prompt_variants": ["minimal", "full"],
      "tool_variants": ["move_only", "all_tools"],
      "runs_per_level": 3,
      "max_turns": 300,
      "state_format": "text"
    }
  }
}
```

---

## 13. Rendering and review

### 13.1 Recording format

Same structure as Hanoi recordings — list of steps with state snapshots:

```json
{
  "metadata": {
    "game": "sokoban",
    "level_id": "microban:1",
    "n_boxes": 2,
    "grid_size": "7x6",
    "optimal_moves": 15,
    "optimal_pushes": 8
  },
  "summary": {
    "solved": true,
    "total_moves": 23,
    "total_pushes": 12,
    "total_illegal_moves": 2,
    "total_tool_calls": 25
  },
  "steps": [
    {
      "index": 0,
      "state_before": "<xsb>",
      "action": null,
      "state_after": "<xsb>"
    },
    {
      "index": 1,
      "state_before": "<xsb>",
      "action": {"name": "sokoban_move", "arguments": {"direction": "right"}},
      "state_after": "<xsb>",
      "action_type": "push"
    }
  ]
}
```

### 13.2 HTML renderer (`render.py`)

Interactive browser-based playback:
- Step slider/arrows
- Grid rendered via inline SVG or canvas
- Move/push counter display
- Deadlock indicator
- Color-coded: walks in one color, pushes in another, illegal in red

### 13.3 Review mode

Same pattern as Hanoi review: prompt + per-step images for manual inspection of
model behavior. Useful for debugging model strategy and identifying failure modes.

---

## 14. File structure

```
games_bench/games/sokoban/
├── __init__.py              # public API exports
├── env.py                   # SokobanEnv, SokobanState, SokobanToolbox
├── adapter.py               # SokobanGameAdapter (GameAdapter protocol)
├── tool_schemas.py          # tool_schemas() function
├── level_loader.py          # XSB parser, LevelSet, SokobanLevel
├── deadlock.py              # dead square detection, freeze deadlock check
├── vision.py                # render_sokoban_image() → StateImage
├── render.py                # recording playback (HTML, ASCII, video)
├── review.py                # manual review with per-step images
├── prompts/
│   ├── __init__.py          # format_instructions(), default_instructions()
│   ├── default.txt
│   └── image_suffix.txt
└── levels/
    ├── microban.xsb
    ├── microban_ii.xsb
    ├── sasquatch.xsb
    ├── original.xsb
    └── metadata.json        # optimal solutions where known

games_bench/bench/sokoban.py     # batch runner, CLI args, default config
```

### Registry wiring

In `games_bench/games/registry.py` → `load_builtin_games()`:
```python
from games_bench.games.sokoban import SokobanEnv
register_game(GameSpec(name="sokoban", description="Sokoban", env_factory=SokobanEnv))
```

In `games_bench/bench/registry.py` → `load_builtin_benchmarks()`:
```python
from games_bench.bench import sokoban as sokoban_bench
from games_bench.games.sokoban import render as sokoban_render, review as sokoban_review

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

## 15. Development phases

### Phase 1: Core environment

**Goal:** Standalone `SokobanEnv` that passes unit tests with no LLM/bench deps.

Tasks:
1. Implement `SokobanLevel` and `SokobanState` dataclasses in `env.py`
2. Implement XSB parser in `level_loader.py` (parse single level, multi-level files)
3. Implement `SokobanEnv` core: constructor, `reset()`, `get_state()`, `move()`,
   `is_solved()`, `get_legal_moves()`, `format_prompt_state()`
4. Implement `step()` RL interface with configurable illegal action behavior
5. Implement reward function (step penalty, push reward, solve reward, illegal penalty)
6. Implement `undo()` with history tracking
7. Implement `SokobanToolbox` wrapper
8. Write exception hierarchy (`SokobanError`, `InvalidLevelError`, `IllegalMoveError`)
9. Bundle Microban level set (`.xsb` file)
10. Write tests: level parsing, movement, pushing, illegal moves, solve detection,
    undo, toolbox error handling

**Tests (Phase 1):**
- `test_level_loading` — parse XSB, multi-level files, edge cases
- `test_movement` — walk in all 4 directions, wall collision
- `test_pushing` — push box, chain push rejection, push into wall
- `test_solve` — solve a small level, verify `is_solved()`
- `test_rl_interface` — `step()` returns, reward values, done flag
- `test_undo` — undo push, undo walk, undo past start
- `test_toolbox` — all tool methods return `{"ok": ...}` dicts

### Phase 2: Deadlock detection

**Goal:** Detect common deadlocks without expensive computation.

Tasks:
1. Implement simple dead square detection in `deadlock.py` (reverse-reachability
   flood-fill from goals)
2. Implement freeze deadlock detection (recursive axis check)
3. Integrate into `SokobanEnv`: `is_deadlocked()`, optional `deadlock_penalty`
4. Add deadlock info to toolbox move results (warning field)
5. Write tests: dead corners, dead edges, freeze deadlocks, false-positive checks

**Tests (Phase 2):**
- `test_dead_squares` — corner box, edge box, non-dead positions
- `test_freeze_deadlock` — two boxes blocking each other, box against wall + box
- `test_no_false_positives` — solvable position not flagged

### Phase 3: Adapter and prompts

**Goal:** `SokobanGameAdapter` that plugs into the existing LLM harness.

Tasks:
1. Implement `SokobanGameAdapter` in `adapter.py`
2. Implement `tool_schemas()` in `tool_schemas.py`
3. Create prompt templates in `prompts/`
4. Define prompt variants (minimal, with_legal_moves, with_deadlock_warnings, full)
5. Implement `episode_metrics()` with all per-episode metrics
6. Export public API in `__init__.py`
7. Register in `games/registry.py`
8. Write adapter integration test (execute_tool routing, ToolExecution metadata)

**Tests (Phase 3):**
- `test_adapter_tool_routing` — each tool dispatches correctly
- `test_adapter_metrics` — episode_metrics returns expected fields
- `test_adapter_instructions` — default and custom instructions
- `test_tool_execution_metadata` — mutating/illegal flags set correctly

### Phase 4: Vision

**Goal:** PIL-based grid renderer for multimodal benchmarks.

Tasks:
1. Implement `render_sokoban_image()` in `vision.py`
2. Tile-based rendering with color scheme
3. Optional grid labels (row/col indices)
4. `render_sokoban_state_image()` and `render_sokoban_env_image()` wrappers
5. Create `image_suffix.txt` prompt
6. Write visual regression test (render known state, check dimensions/non-empty)

**Tests (Phase 4):**
- `test_render_dimensions` — output size matches expected tile_size * grid
- `test_render_returns_state_image` — correct dataclass fields

### Phase 5: Benchmark runner

**Goal:** `bench/sokoban.py` fully integrated with batch CLI.

Tasks:
1. Implement `default_sokoban_config()`, `add_sokoban_arguments()`
2. Implement `build_sokoban_adapter()`
3. Implement `run_batch()` with level iteration, variant loops, metrics
4. Register `BenchSpec` in `bench/registry.py`
5. Implement difficulty-stratified summary (solve rate by optimal solution bucket)
6. Verify `games-bench run sokoban --help` works
7. Verify `games-bench run --config configs/sokoban.json` works
8. Write batch runner test (args not mutated, config merge)

**Tests (Phase 5):**
- `test_sokoban_batch_args` — args namespace not mutated across iterations
- `test_sokoban_config_merge` — default < global < per-game precedence

### Phase 6: Rendering and review

**Goal:** Recording playback and manual review tooling.

Tasks:
1. Implement `render.py` — HTML playback with step slider
2. Implement `review.py` — per-step image review
3. Wire `render_main` and `review_main` into BenchSpec
4. Verify `games-bench render --game sokoban --run-dir ...` works
5. Verify `games-bench review --game sokoban --run-dir ...` works

### Phase 7: Additional level sets and polish

**Goal:** Expand level coverage, add optimal solution data, update docs.

Tasks:
1. Bundle additional level sets (Sasquatch, Original, Boxoban subsets)
2. Add `metadata.json` with optimal solution data where available
3. Update `CLAUDE.md`, `AGENTS.md`, `README.md`
4. Add Sokoban section to README (standalone usage example)
5. Create `configs/sokoban.json` sample config
6. Run full test suite, verify no regressions

---

## 16. Architectural decisions

### Why separate `level_loader.py` from `env.py`?

Level parsing/loading is a distinct concern from game simulation. The loader handles
file I/O, XSB parsing, and metadata — the env operates on an already-parsed
`SokobanLevel`. This keeps the env pure (no file I/O) and lets levels be loaded from
strings, files, or databases.

### Why separate `deadlock.py`?

Deadlock detection is algorithmically complex and independently testable. Keeping it
in its own module:
- Allows the env to optionally skip detection (`detect_deadlocks=False`)
- Makes the algorithms easier to test in isolation
- Keeps `env.py` focused on core game mechanics

### Why not procedural level generation?

Procedural generation (like DeepMind's Boxoban) is valuable for RL training but
problematic for benchmarking LLMs:
- Generated levels lack curated difficulty curves
- No guaranteed optimal solutions for comparison
- Reproducibility requires saving seeds + generator version

Instead, we use curated level sets with known solutions. This gives deterministic,
reproducible benchmarks with quantitative efficiency metrics.

### Why XSB format for text state?

- Universal standard — every Sokoban tool, wiki, and paper uses it
- Maximally compact — one character per cell
- Self-documenting — readable without a legend
- Round-trip safe — `parse_xsb(state.to_xsb()) == state`
- Proven with LLMs — SokoBench used XSB successfully

### Why include `undo` as a tool?

Undo transforms Sokoban from an irreversible puzzle into a search problem with
backtracking. This creates a meaningful tool variant axis:
- `move_only`: tests pure forward planning (can the model avoid deadlocks?)
- `all_tools` (with undo): tests search strategy (can the model explore and backtrack?)

Comparing solve rates across these variants measures the value of backtracking
capability — directly relevant to the planning vs. simulation distinction.
