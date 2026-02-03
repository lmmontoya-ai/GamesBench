from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

PegIndex: TypeAlias = int
Disk: TypeAlias = int
Action: TypeAlias = tuple[PegIndex, PegIndex]

ACTION_SPACE: tuple[Action, ...] = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)


class HanoiError(Exception):
    """Base exception for the Tower of Hanoi environment."""


class InvalidPegError(HanoiError, ValueError):
    """Raised when a peg index is out of range."""


class InvalidActionError(HanoiError, ValueError):
    """Raised when an action cannot be parsed or decoded."""


class IllegalMoveError(HanoiError):
    """Raised when a move violates Tower of Hanoi rules."""


@dataclass(frozen=True, slots=True)
class HanoiState:
    """Immutable snapshot of a Tower of Hanoi configuration.

    Representation notes:
      - `pegs` is a 3-tuple of stacks, each stack listed bottom->top.
      - Disk sizes are integers 1..n, where 1 is the smallest.
      - `disk_positions[d-1]` gives the peg index (0..2) holding disk `d`.
    """

    n_disks: int
    pegs: tuple[tuple[Disk, ...], tuple[Disk, ...], tuple[Disk, ...]]
    disk_positions: tuple[PegIndex, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_disks": self.n_disks,
            "pegs": [list(p) for p in self.pegs],
            "disk_positions": list(self.disk_positions),
        }


def _expected_stack(n_disks: int) -> list[Disk]:
    return list(range(n_disks, 0, -1))


def _validate_n_disks(n_disks: int) -> None:
    if isinstance(n_disks, bool) or not isinstance(n_disks, int):
        raise TypeError(f"n_disks must be int, got {type(n_disks).__name__}")
    if n_disks < 1:
        raise ValueError(f"n_disks must be >= 1, got {n_disks}")


def _validate_peg_index(peg: int) -> None:
    if isinstance(peg, bool) or not isinstance(peg, int):
        raise TypeError(f"peg index must be int, got {type(peg).__name__}")
    if peg < 0 or peg > 2:
        raise InvalidPegError(f"peg index must be in [0, 2], got {peg}")


def _parse_action(action: object) -> Action:
    if isinstance(action, int) and not isinstance(action, bool):
        if 0 <= action < len(ACTION_SPACE):
            return ACTION_SPACE[action]
        raise InvalidActionError(f"action int must be in [0, {len(ACTION_SPACE) - 1}]")
    if isinstance(action, tuple) and len(action) == 2:
        from_peg, to_peg = action
    elif isinstance(action, list) and len(action) == 2:
        from_peg, to_peg = action
    else:
        raise InvalidActionError("action must be an int or a (from_peg, to_peg) pair")

    if (
        isinstance(from_peg, bool)
        or isinstance(to_peg, bool)
        or not isinstance(from_peg, int)
        or not isinstance(to_peg, int)
    ):
        raise InvalidActionError("from_peg and to_peg must be integers")
    return (from_peg, to_peg)


def state_schema() -> dict[str, Any]:
    """JSON schema for HanoiState, suitable for OpenAPI/function-calling tooling."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "n_disks": {"type": "integer", "minimum": 1},
            "pegs": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": "Disk sizes listed bottom->top.",
                },
            },
            "disk_positions": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 2},
                "description": "disk_positions[d-1] is the peg holding disk d.",
            },
        },
        "required": ["n_disks", "pegs", "disk_positions"],
    }


def tool_schemas(*, tool_prefix: str = "hanoi") -> list[dict[str, Any]]:
    """Tool schemas (JSON/OpenAPI-style) for LLM tool-calling integrations."""

    move_params = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "from_peg": {"type": "integer", "minimum": 0, "maximum": 2},
            "to_peg": {"type": "integer", "minimum": 0, "maximum": 2},
        },
        "required": ["from_peg", "to_peg"],
    }
    reset_params = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"n_disks": {"type": "integer", "minimum": 1}},
        "required": ["n_disks"],
    }
    step_params = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "description": "Either an int in [0,5] or a pair [from_peg,to_peg].",
                "anyOf": [
                    {"type": "integer", "minimum": 0, "maximum": len(ACTION_SPACE) - 1},
                    {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "integer", "minimum": 0, "maximum": 2},
                    },
                ],
            }
        },
        "required": ["action"],
    }

    state = state_schema()
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ok": {"type": "boolean"},
            "state": state,
            "error": {"type": "string"},
        },
        "required": ["ok", "state"],
    }
    step_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ok": {"type": "boolean"},
            "state": state,
            "reward": {"type": "number"},
            "done": {"type": "boolean"},
            "info": {"type": "object"},
            "error": {"type": "string"},
        },
        "required": ["ok", "state", "reward", "done", "info"],
    }

    return [
        {
            "name": f"{tool_prefix}_get_state",
            "description": "Return the current Tower of Hanoi configuration.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
            "response_schema": result_schema,
        },
        {
            "name": f"{tool_prefix}_move",
            "description": "Move the top disk from one peg to another (pegs are 0,1,2).",
            "parameters": move_params,
            "response_schema": result_schema,
        },
        {
            "name": f"{tool_prefix}_reset",
            "description": "Reset the puzzle to the initial configuration with n_disks on peg 0.",
            "parameters": reset_params,
            "response_schema": result_schema,
        },
        {
            "name": f"{tool_prefix}_is_solved",
            "description": "Return whether the puzzle is currently solved.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
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
            "description": "Return all currently legal moves as pairs (from_peg,to_peg).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
            "response_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ok": {"type": "boolean"},
                    "legal_moves": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "integer", "minimum": 0, "maximum": 2},
                        },
                    },
                },
                "required": ["ok", "legal_moves"],
            },
        },
        {
            "name": f"{tool_prefix}_step",
            "description": "Apply an action (int or pair) and return (state, reward, done, info).",
            "parameters": step_params,
            "response_schema": step_result_schema,
        },
    ]


def format_state_for_prompt(
    state: HanoiState,
    *,
    move_count: int | None = None,
    step_count: int | None = None,
    legal_moves: list[Action] | None = None,
    action_space: tuple[Action, ...] | None = None,
    compact_json: bool = False,
) -> str:
    """Return a prompt-friendly string representation of the environment state."""

    payload: dict[str, Any] = {"state": state.to_dict()}
    if move_count is not None:
        payload["move_count"] = move_count
    if step_count is not None:
        payload["step_count"] = step_count
    if legal_moves is not None:
        payload["legal_moves"] = [list(m) for m in legal_moves]
    if action_space is not None:
        payload["action_space"] = [list(a) for a in action_space]
        payload["action_space_note"] = (
            "If using int actions, action_space[i] gives [from_peg,to_peg]."
        )

    if compact_json:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return json.dumps(payload, indent=2, sort_keys=True)


class TowerOfHanoiEnv:
    """Framework-agnostic Tower of Hanoi environment.

    This supports both:
      - An RL loop via `step(action) -> (state, reward, done, info)`
      - A tool-calling interface via `tool_schemas()` + `format_state_for_prompt(...)`
    """

    def __init__(
        self,
        n_disks: int = 3,
        *,
        start_peg: PegIndex = 0,
        goal_peg: PegIndex = 2,
        step_penalty: float = 0.0,
        illegal_move_penalty: float = -1.0,
        solve_reward: float = 1.0,
        shaping_weight: float = 0.0,
        illegal_action_behavior: Literal["penalize", "raise", "terminate"] = "penalize",
        max_steps: int | None = None,
        record_history: bool = False,
    ) -> None:
        _validate_n_disks(n_disks)
        _validate_peg_index(start_peg)
        _validate_peg_index(goal_peg)
        if start_peg == goal_peg:
            raise ValueError("start_peg and goal_peg must be different")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.n_disks = n_disks
        self.start_peg = start_peg
        self.goal_peg = goal_peg

        self.step_penalty = float(step_penalty)
        self.illegal_move_penalty = float(illegal_move_penalty)
        self.solve_reward = float(solve_reward)
        self.shaping_weight = float(shaping_weight)
        self.illegal_action_behavior = illegal_action_behavior
        self.max_steps = max_steps
        self.record_history = record_history

        self._pegs: list[list[Disk]] = [[], [], []]
        self._disk_positions: list[PegIndex] = []
        self.move_count = 0
        self.step_count = 0
        self.history: list[Action] = []

        self.reset(n_disks)

    @property
    def action_space(self) -> tuple[Action, ...]:
        """All 6 directed moves (from_peg,to_peg) with from_peg != to_peg."""

        return ACTION_SPACE

    def encode_action(self, action: Action) -> int:
        """Map a (from_peg,to_peg) action to an integer in [0,5]."""

        try:
            return self.action_space.index(action)
        except ValueError as exc:
            raise InvalidActionError(
                f"action must be one of {self.action_space}"
            ) from exc

    def decode_action(self, action: int) -> Action:
        """Map an integer in [0,5] to a (from_peg,to_peg) action."""

        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if 0 <= action < len(self.action_space):
            return self.action_space[action]
        raise InvalidActionError(
            f"action int must be in [0, {len(self.action_space) - 1}]"
        )

    def reset(self, n_disks: int | None = None) -> HanoiState:
        if n_disks is not None:
            _validate_n_disks(n_disks)
            self.n_disks = n_disks

        self._pegs = [[], [], []]
        self._pegs[self.start_peg] = _expected_stack(self.n_disks)
        self._disk_positions = [self.start_peg for _ in range(self.n_disks)]

        self.move_count = 0
        self.step_count = 0
        self.history = []
        return self.get_state()

    def get_state(self) -> HanoiState:
        pegs = (tuple(self._pegs[0]), tuple(self._pegs[1]), tuple(self._pegs[2]))
        positions = tuple(self._disk_positions)
        return HanoiState(n_disks=self.n_disks, pegs=pegs, disk_positions=positions)

    def is_solved(self) -> bool:
        return self._pegs[self.goal_peg] == _expected_stack(self.n_disks)

    def get_legal_moves(self) -> list[Action]:
        legal: list[Action] = []
        for from_peg in range(3):
            if not self._pegs[from_peg]:
                continue
            disk = self._pegs[from_peg][-1]
            for to_peg in range(3):
                if to_peg == from_peg:
                    continue
                if not self._pegs[to_peg] or self._pegs[to_peg][-1] > disk:
                    legal.append((from_peg, to_peg))
        return legal

    def move(self, from_peg: PegIndex, to_peg: PegIndex) -> HanoiState:
        _validate_peg_index(from_peg)
        _validate_peg_index(to_peg)
        if from_peg == to_peg:
            raise IllegalMoveError("from_peg and to_peg must be different")
        if not self._pegs[from_peg]:
            raise IllegalMoveError(f"peg {from_peg} is empty")

        disk = self._pegs[from_peg][-1]
        if self._pegs[to_peg] and self._pegs[to_peg][-1] < disk:
            raise IllegalMoveError(
                f"cannot place disk {disk} on top of smaller disk {self._pegs[to_peg][-1]}"
            )

        self._pegs[from_peg].pop()
        self._pegs[to_peg].append(disk)
        self._disk_positions[disk - 1] = to_peg
        self.move_count += 1
        if self.record_history:
            self.history.append((from_peg, to_peg))
        return self.get_state()

    def optimal_steps(self) -> int:
        return (1 << self.n_disks) - 1

    def _goal_prefix_potential(self) -> int:
        goal_stack = self._pegs[self.goal_peg]
        expected = _expected_stack(self.n_disks)
        count = 0
        for actual, want in zip(goal_stack, expected, strict=False):
            if actual != want:
                break
            count += 1
        return count

    def step(self, action: object) -> tuple[HanoiState, float, bool, dict[str, Any]]:
        """Apply an action.

        Action formats:
          - int in [0,5], where `action_space[i]` is (from_peg,to_peg)
          - (from_peg,to_peg) as a tuple/list of two ints
        """

        self.step_count += 1
        info: dict[str, Any] = {
            "action_space": self.action_space,
            "step_count": self.step_count,
            "move_count": self.move_count,
            "truncated": False,
            "illegal_action": False,
        }

        try:
            from_peg, to_peg = _parse_action(action)
        except InvalidActionError as exc:
            info["illegal_action"] = True
            info["error"] = str(exc)
            if self.illegal_action_behavior == "raise":
                raise
            if self.illegal_action_behavior == "terminate":
                return (self.get_state(), self.illegal_move_penalty, True, info)
            return (self.get_state(), self.illegal_move_penalty, False, info)

        info["action"] = (from_peg, to_peg)
        legal_moves = self.get_legal_moves()
        info["legal_moves"] = legal_moves

        if (from_peg, to_peg) not in legal_moves:
            info["illegal_action"] = True
            info["error"] = "illegal move"
            if self.illegal_action_behavior == "raise":
                raise IllegalMoveError("illegal move")
            if self.illegal_action_behavior == "terminate":
                return (self.get_state(), self.illegal_move_penalty, True, info)
            return (self.get_state(), self.illegal_move_penalty, False, info)

        prev_potential = self._goal_prefix_potential()
        state = self.move(from_peg, to_peg)
        new_potential = self._goal_prefix_potential()

        done = self.is_solved()
        reward = self.step_penalty
        if self.shaping_weight:
            reward += self.shaping_weight * float(new_potential - prev_potential)
        if done:
            reward += self.solve_reward

        if (
            self.max_steps is not None
            and self.step_count >= self.max_steps
            and not done
        ):
            info["truncated"] = True
            done = True

        info["move_count"] = self.move_count
        info["solved"] = done and not info.get("truncated", False)
        if self.record_history:
            info["history"] = list(self.history)
        return (state, reward, done, info)

    def format_prompt_state(
        self,
        *,
        include_legal_moves: bool = True,
        include_action_space: bool = True,
        compact_json: bool = False,
    ) -> str:
        legal_moves = self.get_legal_moves() if include_legal_moves else None
        action_space = self.action_space if include_action_space else None
        return format_state_for_prompt(
            self.get_state(),
            move_count=self.move_count,
            step_count=self.step_count,
            legal_moves=legal_moves,
            action_space=action_space,
            compact_json=compact_json,
        )


class HanoiToolbox:
    """Thin wrappers around an environment instance that return JSONable results.

    This is handy for LLM tool-calling, where you often want tools to *return*
    an error string rather than raising exceptions.
    """

    def __init__(self, env: TowerOfHanoiEnv):
        self.env = env

    def get_state(self) -> dict[str, Any]:
        return {"ok": True, "state": self.env.get_state().to_dict()}

    def move(self, from_peg: int, to_peg: int) -> dict[str, Any]:
        try:
            state = self.env.move(from_peg, to_peg)
            return {"ok": True, "state": state.to_dict()}
        except HanoiError as exc:
            return {
                "ok": False,
                "state": self.env.get_state().to_dict(),
                "error": str(exc),
            }

    def reset(self, n_disks: int) -> dict[str, Any]:
        try:
            state = self.env.reset(n_disks)
            return {"ok": True, "state": state.to_dict()}
        except (TypeError, ValueError) as exc:
            return {
                "ok": False,
                "state": self.env.get_state().to_dict(),
                "error": str(exc),
            }

    def is_solved(self) -> dict[str, Any]:
        return {"ok": True, "solved": self.env.is_solved()}

    def get_legal_moves(self) -> dict[str, Any]:
        return {
            "ok": True,
            "legal_moves": [list(m) for m in self.env.get_legal_moves()],
        }

    def step(self, action: object) -> dict[str, Any]:
        try:
            state, reward, done, info = self.env.step(action)
            return {
                "ok": True,
                "state": state.to_dict(),
                "reward": reward,
                "done": done,
                "info": info,
            }
        except HanoiError as exc:
            return {
                "ok": False,
                "state": self.env.get_state().to_dict(),
                "reward": self.env.illegal_move_penalty,
                "done": self.env.illegal_action_behavior == "terminate",
                "info": {"error": str(exc)},
                "error": str(exc),
            }
