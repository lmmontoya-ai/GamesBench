from __future__ import annotations

from collections import deque
from typing import Literal, TypeAlias

Position: TypeAlias = tuple[int, int]
Axis: TypeAlias = Literal["horizontal", "vertical"]

_DIRECTIONS: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(width: int, height: int, pos: Position) -> bool:
    row, col = pos
    return 0 <= row < height and 0 <= col < width


def _is_wall_or_oob(
    *,
    width: int,
    height: int,
    walls: frozenset[Position],
    pos: Position,
) -> bool:
    return not _in_bounds(width, height, pos) or pos in walls


def _traversable_cells(
    *,
    width: int,
    height: int,
    walls: frozenset[Position],
) -> set[Position]:
    traversable: set[Position] = set()
    for row in range(height):
        for col in range(width):
            pos = (row, col)
            if pos not in walls:
                traversable.add(pos)
    return traversable


def compute_dead_squares(
    *,
    width: int,
    height: int,
    walls: frozenset[Position],
    goals: frozenset[Position],
) -> frozenset[Position]:
    traversable = _traversable_cells(width=width, height=height, walls=walls)
    if not traversable:
        return frozenset()

    reachable: set[Position] = set()
    queue: deque[Position] = deque()

    for goal in goals:
        if goal in traversable:
            reachable.add(goal)
            queue.append(goal)

    while queue:
        to_pos = queue.popleft()
        for dr, dc in _DIRECTIONS:
            from_pos = (to_pos[0] - dr, to_pos[1] - dc)
            player_stand = (from_pos[0] - dr, from_pos[1] - dc)
            if from_pos not in traversable:
                continue
            if player_stand not in traversable:
                continue
            if from_pos in reachable:
                continue
            reachable.add(from_pos)
            queue.append(from_pos)

    dead_squares = {
        pos for pos in traversable if pos not in goals and pos not in reachable
    }
    return frozenset(dead_squares)


def has_dead_square_deadlock(
    *,
    boxes: frozenset[Position],
    goals: frozenset[Position],
    dead_squares: frozenset[Position],
) -> bool:
    for box in boxes:
        if box in goals:
            continue
        if box in dead_squares:
            return True
    return False


def _axis_steps(axis: Axis) -> tuple[Position, Position]:
    if axis == "horizontal":
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def _is_axis_blocked(
    *,
    box: Position,
    axis: Axis,
    boxes: frozenset[Position],
    walls: frozenset[Position],
    width: int,
    height: int,
    memo: dict[tuple[Position, Axis], bool],
    visiting: set[tuple[Position, Axis]],
) -> bool:
    key = (box, axis)
    if key in memo:
        return memo[key]
    if key in visiting:
        # Cyclic same-axis box dependencies are treated as blocked components.
        return True

    visiting.add(key)
    side_steps = _axis_steps(axis)
    side_blocked: list[bool] = []
    for dr, dc in side_steps:
        neighbor = (box[0] + dr, box[1] + dc)
        if _is_wall_or_oob(width=width, height=height, walls=walls, pos=neighbor):
            side_blocked.append(True)
            continue
        if neighbor not in boxes:
            side_blocked.append(False)
            continue
        side_blocked.append(
            _is_axis_blocked(
                box=neighbor,
                axis=axis,
                boxes=boxes,
                walls=walls,
                width=width,
                height=height,
                memo=memo,
                visiting=visiting,
            )
        )

    visiting.remove(key)
    result = side_blocked[0] and side_blocked[1]
    memo[key] = result
    return result


def has_freeze_deadlock(
    *,
    width: int,
    height: int,
    walls: frozenset[Position],
    boxes: frozenset[Position],
    goals: frozenset[Position],
) -> bool:
    memo: dict[tuple[Position, Axis], bool] = {}
    for box in boxes:
        if box in goals:
            continue
        horizontal = _is_axis_blocked(
            box=box,
            axis="horizontal",
            boxes=boxes,
            walls=walls,
            width=width,
            height=height,
            memo=memo,
            visiting=set(),
        )
        if not horizontal:
            continue
        vertical = _is_axis_blocked(
            box=box,
            axis="vertical",
            boxes=boxes,
            walls=walls,
            width=width,
            height=height,
            memo=memo,
            visiting=set(),
        )
        if vertical:
            return True
    return False


def is_deadlocked(
    *,
    width: int,
    height: int,
    walls: frozenset[Position],
    goals: frozenset[Position],
    boxes: frozenset[Position],
    dead_squares: frozenset[Position] | None = None,
) -> bool:
    if dead_squares is None:
        dead_squares_resolved = compute_dead_squares(
            width=width,
            height=height,
            walls=walls,
            goals=goals,
        )
    else:
        dead_squares_resolved = dead_squares
    if has_dead_square_deadlock(
        boxes=boxes,
        goals=goals,
        dead_squares=dead_squares_resolved,
    ):
        return True
    return has_freeze_deadlock(
        width=width,
        height=height,
        walls=walls,
        boxes=boxes,
        goals=goals,
    )
