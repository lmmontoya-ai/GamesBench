from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from .deadlock import compute_dead_squares, is_deadlocked as is_deadlocked_state

Position: TypeAlias = tuple[int, int]
Direction: TypeAlias = Literal["up", "down", "left", "right"]

ACTION_SPACE: tuple[Direction, ...] = ("up", "down", "left", "right")
ACTION_INDEX: dict[Direction, int] = {
    direction: idx for idx, direction in enumerate(ACTION_SPACE)
}

DIRECTION_DELTAS: dict[Direction, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class SokobanError(Exception):
    """Base exception for Sokoban environment errors."""


class InvalidLevelError(SokobanError, ValueError):
    """Raised when a level definition is invalid."""


class InvalidActionError(SokobanError, ValueError):
    """Raised when an action cannot be parsed."""


class IllegalMoveError(SokobanError):
    """Raised when a movement request violates Sokoban rules."""


@dataclass(frozen=True, slots=True)
class SokobanLevel:
    level_id: str
    title: str | None
    width: int
    height: int
    xsb: str
    walls: frozenset[Position]
    boxes_start: frozenset[Position]
    goals: frozenset[Position]
    player_start: Position
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
    walls: frozenset[Position]
    boxes: frozenset[Position]
    goals: frozenset[Position]
    player: Position
    n_boxes: int

    def to_xsb(self) -> str:
        board = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for row, col in self.walls:
            board[row][col] = "#"
        for row, col in self.goals:
            if board[row][col] != "#":
                board[row][col] = "."
        for row, col in self.boxes:
            if board[row][col] == ".":
                board[row][col] = "*"
            else:
                board[row][col] = "$"
        player_row, player_col = self.player
        if board[player_row][player_col] == ".":
            board[player_row][player_col] = "+"
        else:
            board[player_row][player_col] = "@"
        return "\n".join("".join(row) for row in board)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "walls": [list(pos) for pos in sorted(self.walls)],
            "boxes": [list(pos) for pos in sorted(self.boxes)],
            "goals": [list(pos) for pos in sorted(self.goals)],
            "player": list(self.player),
            "n_boxes": self.n_boxes,
            "xsb": self.to_xsb(),
        }


def _is_position_in_bounds(width: int, height: int, pos: Position) -> bool:
    row, col = pos
    return 0 <= row < height and 0 <= col < width


def _parse_direction(action: object) -> Direction:
    if isinstance(action, int) and not isinstance(action, bool):
        if 0 <= action < len(ACTION_SPACE):
            return ACTION_SPACE[action]
        raise InvalidActionError(f"action int must be in [0, {len(ACTION_SPACE) - 1}]")

    if isinstance(action, str):
        normalized = action.strip().lower()
        if normalized in DIRECTION_DELTAS:
            return normalized  # type: ignore[return-value]
    raise InvalidActionError(
        "action must be a direction string (up/down/left/right) or int index (0-3)"
    )


def tool_schemas(*, tool_prefix: str = "sokoban") -> list[dict[str, Any]]:
    move_params = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
            }
        },
        "required": ["direction"],
    }

    empty_params = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }

    state_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "walls": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer", "minimum": 0},
                },
            },
            "boxes": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer", "minimum": 0},
                },
            },
            "goals": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer", "minimum": 0},
                },
            },
            "player": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "n_boxes": {"type": "integer", "minimum": 1},
            "xsb": {"type": "string"},
        },
        "required": [
            "width",
            "height",
            "walls",
            "boxes",
            "goals",
            "player",
            "n_boxes",
            "xsb",
        ],
    }
    state_payload_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ok": {"type": "boolean"},
            "state": state_schema,
            "boxes_on_goals": {"type": "integer", "minimum": 0},
            "total_goals": {"type": "integer", "minimum": 1},
            "error": {"type": "string"},
        },
        "required": ["ok", "state", "boxes_on_goals", "total_goals"],
    }

    return [
        {
            "name": f"{tool_prefix}_move",
            "description": "Move the player in one direction; may push one box.",
            "parameters": move_params,
            "response_schema": {
                **state_payload_schema,
                "properties": {
                    **state_payload_schema["properties"],
                    "action_type": {"type": "string", "enum": ["walk", "push"]},
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    },
                    "deadlocked": {"type": "boolean"},
                },
            },
        },
        {
            "name": f"{tool_prefix}_get_state",
            "description": "Return the current Sokoban board and counters.",
            "parameters": empty_params,
            "response_schema": state_payload_schema,
        },
        {
            "name": f"{tool_prefix}_is_solved",
            "description": "Return whether all goals are currently occupied by boxes.",
            "parameters": empty_params,
            "response_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ok": {"type": "boolean"},
                    "solved": {"type": "boolean"},
                },
                "required": ["ok", "solved"],
            },
        },
        {
            "name": f"{tool_prefix}_get_legal_moves",
            "description": "Return legal directions for the current board state.",
            "parameters": empty_params,
            "response_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ok": {"type": "boolean"},
                    "legal_moves": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                        },
                    },
                },
                "required": ["ok", "legal_moves"],
            },
        },
        {
            "name": f"{tool_prefix}_undo",
            "description": "Undo the last legal move.",
            "parameters": empty_params,
            "response_schema": state_payload_schema,
        },
    ]


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
    ) -> None:
        if illegal_action_behavior not in {"penalize", "raise", "terminate"}:
            raise ValueError(
                "illegal_action_behavior must be one of: penalize, raise, terminate"
            )
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.level = level
        self.step_penalty = float(step_penalty)
        self.push_reward = float(push_reward)
        self.push_off_penalty = float(push_off_penalty)
        self.illegal_move_penalty = float(illegal_move_penalty)
        self.solve_reward = float(solve_reward)
        self.deadlock_penalty = float(deadlock_penalty)
        self.illegal_action_behavior = illegal_action_behavior
        self.max_steps = max_steps
        self.record_history = record_history
        self.detect_deadlocks = detect_deadlocks
        self.terminal_on_deadlock = terminal_on_deadlock

        self._boxes: set[Position] = set()
        self._player: Position = level.player_start
        self._undo_stack: list[tuple[frozenset[Position], Position, int, int]] = []
        self._dead_squares = compute_dead_squares(
            width=level.width,
            height=level.height,
            walls=level.walls,
            goals=level.goals,
        )

        self.move_count = 0
        self.push_count = 0
        self.step_count = 0
        self.history: list[dict[str, Any]] = []

        self.reset()

    @property
    def n_boxes(self) -> int:
        return self.level.n_boxes

    @property
    def action_space(self) -> tuple[Direction, ...]:
        return ACTION_SPACE

    @property
    def boxes_on_goals(self) -> int:
        return len(self._boxes.intersection(self.level.goals))

    def get_state(self) -> SokobanState:
        return SokobanState(
            width=self.level.width,
            height=self.level.height,
            walls=self.level.walls,
            boxes=frozenset(self._boxes),
            goals=self.level.goals,
            player=self._player,
            n_boxes=self.level.n_boxes,
        )

    def is_solved(self) -> bool:
        return self._boxes == set(self.level.goals)

    def is_deadlocked(self) -> bool:
        return is_deadlocked_state(
            width=self.level.width,
            height=self.level.height,
            walls=self.level.walls,
            goals=self.level.goals,
            boxes=frozenset(self._boxes),
            dead_squares=self._dead_squares,
        )

    def reset(self) -> SokobanState:
        self._boxes = set(self.level.boxes_start)
        self._player = self.level.player_start
        self._undo_stack = []
        self.move_count = 0
        self.push_count = 0
        self.step_count = 0
        self.history = []
        return self.get_state()

    def _target(self, direction: Direction) -> Position:
        dr, dc = DIRECTION_DELTAS[direction]
        row, col = self._player
        return (row + dr, col + dc)

    def _is_wall_or_oob(self, pos: Position) -> bool:
        if not _is_position_in_bounds(self.level.width, self.level.height, pos):
            return True
        return pos in self.level.walls

    def _apply_max_steps_truncation(self, done: bool, info: dict[str, Any]) -> bool:
        if (
            self.max_steps is not None
            and self.step_count >= self.max_steps
            and not done
        ):
            info["truncated"] = True
            return True
        return done

    def _apply_deadlock_termination(
        self, *, deadlocked: bool, done: bool, info: dict[str, Any]
    ) -> bool:
        if deadlocked and self.terminal_on_deadlock and not done:
            info["deadlock_terminated"] = True
            return True
        return done

    def get_legal_moves(self) -> list[Direction]:
        legal: list[Direction] = []
        for direction in ACTION_SPACE:
            target = self._target(direction)
            if self._is_wall_or_oob(target):
                continue
            if target in self._boxes:
                dr, dc = DIRECTION_DELTAS[direction]
                beyond = (target[0] + dr, target[1] + dc)
                if self._is_wall_or_oob(beyond) or beyond in self._boxes:
                    continue
            legal.append(direction)
        return legal

    def _apply_move(self, direction: Direction) -> tuple[SokobanState, dict[str, Any]]:
        if direction not in DIRECTION_DELTAS:
            raise InvalidActionError(f"unknown direction: {direction}")

        target = self._target(direction)
        if self._is_wall_or_oob(target):
            raise IllegalMoveError("cannot move into wall")

        action_type = "walk"
        push_to_goal = False
        push_off_goal = False

        self._undo_stack.append(
            (frozenset(self._boxes), self._player, self.move_count, self.push_count)
        )

        if target in self._boxes:
            dr, dc = DIRECTION_DELTAS[direction]
            beyond = (target[0] + dr, target[1] + dc)
            if self._is_wall_or_oob(beyond):
                self._undo_stack.pop()
                raise IllegalMoveError("cannot push box into wall")
            if beyond in self._boxes:
                self._undo_stack.pop()
                raise IllegalMoveError("cannot push two boxes at once")

            action_type = "push"
            push_off_goal = target in self.level.goals
            push_to_goal = beyond in self.level.goals
            self._boxes.remove(target)
            self._boxes.add(beyond)
            self.push_count += 1

        self._player = target
        self.move_count += 1

        if self.record_history:
            self.history.append(
                {
                    "direction": direction,
                    "action_type": action_type,
                    "move_count": self.move_count,
                    "push_count": self.push_count,
                }
            )

        return self.get_state(), {
            "action_type": action_type,
            "push_to_goal": push_to_goal,
            "push_off_goal": push_off_goal,
            "direction": direction,
        }

    def _move_with_meta(self, direction: str) -> tuple[SokobanState, dict[str, Any]]:
        parsed_direction = _parse_direction(direction)
        state, meta = self._apply_move(parsed_direction)
        return state, meta

    def move(self, direction: str) -> SokobanState:
        state, _meta = self._move_with_meta(direction)
        return state

    def step(
        self, action: str | int
    ) -> tuple[SokobanState, float, bool, dict[str, Any]]:
        self.step_count += 1
        info: dict[str, Any] = {
            "action_space": list(ACTION_SPACE),
            "step_count": self.step_count,
            "move_count": self.move_count,
            "push_count": self.push_count,
            "boxes_on_goals": self.boxes_on_goals,
            "illegal_action": False,
            "truncated": False,
        }

        try:
            direction = _parse_direction(action)
        except InvalidActionError as exc:
            info["illegal_action"] = True
            info["error"] = str(exc)
            if self.illegal_action_behavior == "raise":
                raise
            done = self.illegal_action_behavior == "terminate"
            done = self._apply_max_steps_truncation(done, info)
            deadlocked = self.detect_deadlocks and self.is_deadlocked()
            done = self._apply_deadlock_termination(
                deadlocked=deadlocked, done=done, info=info
            )
            info["deadlocked"] = deadlocked
            info["solved"] = self.is_solved()
            return (self.get_state(), self.illegal_move_penalty, done, info)

        info["action"] = direction
        try:
            state, move_meta = self._apply_move(direction)
        except IllegalMoveError as exc:
            info["illegal_action"] = True
            info["error"] = str(exc)
            if self.illegal_action_behavior == "raise":
                raise
            done = self.illegal_action_behavior == "terminate"
            done = self._apply_max_steps_truncation(done, info)
            deadlocked = self.detect_deadlocks and self.is_deadlocked()
            done = self._apply_deadlock_termination(
                deadlocked=deadlocked, done=done, info=info
            )
            info["deadlocked"] = deadlocked
            info["solved"] = self.is_solved()
            return (self.get_state(), self.illegal_move_penalty, done, info)

        reward = self.step_penalty
        if move_meta["push_to_goal"]:
            reward += self.push_reward
        if move_meta["push_off_goal"]:
            reward += self.push_off_penalty

        deadlocked = self.detect_deadlocks and self.is_deadlocked()
        if deadlocked:
            reward += self.deadlock_penalty

        solved = self.is_solved()
        done = solved
        if solved:
            reward += self.solve_reward

        done = self._apply_max_steps_truncation(done, info)
        done = self._apply_deadlock_termination(
            deadlocked=deadlocked, done=done, info=info
        )

        info["action_type"] = move_meta["action_type"]
        info["move_count"] = self.move_count
        info["push_count"] = self.push_count
        info["boxes_on_goals"] = self.boxes_on_goals
        info["deadlocked"] = deadlocked
        info["solved"] = solved

        return (state, reward, done, info)

    def undo(self) -> SokobanState:
        if not self._undo_stack:
            raise SokobanError("cannot undo: no history")

        boxes, player, move_count, push_count = self._undo_stack.pop()
        self._boxes = set(boxes)
        self._player = player
        self.move_count = move_count
        self.push_count = push_count
        if self.record_history and self.history:
            self.history.pop()
        return self.get_state()

    def format_prompt_state(
        self,
        *,
        include_legal_moves: bool = False,
        include_deadlock_status: bool = False,
    ) -> str:
        lines = [
            f"Board ({self.level.width}x{self.level.height}):",
            self.get_state().to_xsb(),
            "",
        ]
        lines.append(f"Boxes on goals: {self.boxes_on_goals}/{self.n_boxes}")
        if include_legal_moves:
            legal = ", ".join(self.get_legal_moves())
            lines.append(f"Legal moves: [{legal}]")
        if include_deadlock_status and self.detect_deadlocks:
            lines.append(f"Deadlocked: {self.is_deadlocked()}")
        return "\n".join(lines)


class SokobanToolbox:
    def __init__(self, env: SokobanEnv) -> None:
        self.env = env

    def _state_payload(self) -> dict[str, Any]:
        state = self.env.get_state()
        return {
            "state": state.to_dict(),
            "boxes_on_goals": self.env.boxes_on_goals,
            "total_goals": self.env.n_boxes,
        }

    def move(self, direction: str) -> dict[str, Any]:
        try:
            _state, move_meta = self.env._move_with_meta(direction)
            state_payload = self._state_payload()
            result: dict[str, Any] = {
                "ok": True,
                "direction": move_meta["direction"],
                "action_type": move_meta["action_type"],
                **state_payload,
            }
            if self.env.detect_deadlocks:
                result["deadlocked"] = self.env.is_deadlocked()
            return result
        except SokobanError as exc:
            return {
                "ok": False,
                "error": str(exc),
                **self._state_payload(),
            }

    def get_state(self) -> dict[str, Any]:
        return {"ok": True, **self._state_payload()}

    def is_solved(self) -> dict[str, Any]:
        return {"ok": True, "solved": self.env.is_solved()}

    def get_legal_moves(self) -> dict[str, Any]:
        return {
            "ok": True,
            "legal_moves": list(self.env.get_legal_moves()),
        }

    def undo(self) -> dict[str, Any]:
        try:
            self.env.undo()
            return {"ok": True, **self._state_payload()}
        except SokobanError as exc:
            return {
                "ok": False,
                "error": str(exc),
                **self._state_payload(),
            }
