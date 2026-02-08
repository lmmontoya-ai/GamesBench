from __future__ import annotations

import random
from collections import deque
from typing import Iterable

from .env import DIRECTION_DELTAS, Direction, SokobanEnv, SokobanLevel, SokobanState

Position = tuple[int, int]

_OPPOSITE_DIRECTION: dict[Direction, Direction] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def parse_grid_size(value: str) -> tuple[int, int]:
    text = value.strip().lower()
    parts = text.split("x")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"invalid grid size {value!r}; expected '<width>x<height>'")
    width = int(parts[0])
    height = int(parts[1])
    return (width, height)


def _is_in_bounds(width: int, height: int, pos: Position) -> bool:
    row, col = pos
    return 0 <= row < height and 0 <= col < width


def _interior_cells(width: int, height: int) -> list[Position]:
    return [(row, col) for row in range(1, height - 1) for col in range(1, width - 1)]


def _perimeter_walls(width: int, height: int) -> set[Position]:
    walls: set[Position] = set()
    for row in range(height):
        walls.add((row, 0))
        walls.add((row, width - 1))
    for col in range(width):
        walls.add((0, col))
        walls.add((height - 1, col))
    return walls


def _largest_connected_component(
    *,
    width: int,
    height: int,
    walkable: set[Position],
) -> set[Position]:
    if not walkable:
        return set()
    start = next(iter(walkable))
    seen: set[Position] = {start}
    queue: deque[Position] = deque([start])
    while queue:
        row, col = queue.popleft()
        for dr, dc in DIRECTION_DELTAS.values():
            nxt = (row + dr, col + dc)
            if nxt in walkable and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _sample_walls(
    *,
    width: int,
    height: int,
    n_boxes: int,
    wall_density: float,
    rng: random.Random,
    max_tries: int,
) -> set[Position]:
    perimeter = _perimeter_walls(width, height)
    interior = _interior_cells(width, height)
    if not interior:
        raise ValueError("grid must contain interior cells")

    # Need enough free cells for goals, player, and movement headroom.
    min_open_cells = (2 * n_boxes) + 3
    max_walls = max(0, len(interior) - min_open_cells)
    target_walls = min(max_walls, max(0, int(round(wall_density * len(interior)))))

    for _ in range(max_tries):
        sampled = set(rng.sample(interior, target_walls)) if target_walls else set()
        walkable = set(interior) - sampled
        component = _largest_connected_component(
            width=width, height=height, walkable=walkable
        )
        if len(component) < min_open_cells:
            continue
        if component != walkable:
            continue
        return perimeter | sampled

    # Fallback to perimeter-only walls if stochastic layout search fails.
    return perimeter


def _legal_reverse_walks(
    *,
    width: int,
    height: int,
    walls: set[Position],
    boxes: set[Position],
    player: Position,
) -> list[Direction]:
    legal: list[Direction] = []
    for direction, (dr, dc) in DIRECTION_DELTAS.items():
        nxt = (player[0] + dr, player[1] + dc)
        if not _is_in_bounds(width, height, nxt):
            continue
        if nxt in walls or nxt in boxes:
            continue
        legal.append(direction)
    return legal


def _legal_reverse_pulls(
    *,
    width: int,
    height: int,
    walls: set[Position],
    boxes: set[Position],
    player: Position,
) -> list[Direction]:
    legal: list[Direction] = []
    for direction, (dr, dc) in DIRECTION_DELTAS.items():
        box_pos = (player[0] + dr, player[1] + dc)
        player_next = (player[0] - dr, player[1] - dc)
        if not _is_in_bounds(width, height, box_pos):
            continue
        if not _is_in_bounds(width, height, player_next):
            continue
        if box_pos not in boxes:
            continue
        if player_next in walls or player_next in boxes:
            continue
        legal.append(direction)
    return legal


def _validate_level_shape(width: int, height: int, n_boxes: int) -> None:
    if width < 6 or height < 6:
        raise ValueError("width and height must both be >= 6")
    if n_boxes < 1:
        raise ValueError("n_boxes must be >= 1")
    interior_count = (width - 2) * (height - 2)
    if n_boxes >= interior_count:
        raise ValueError(
            f"n_boxes={n_boxes} is too large for grid {width}x{height}; "
            f"maximum is {interior_count - 1}"
        )


def _validate_density(wall_density: float) -> None:
    if wall_density < 0.0 or wall_density > 0.35:
        raise ValueError("wall_density must be within [0.0, 0.35]")


def _build_level(
    *,
    width: int,
    height: int,
    level_id: str,
    title: str | None,
    walls: Iterable[Position],
    goals: Iterable[Position],
    boxes_start: Iterable[Position],
    player_start: Position,
) -> SokobanLevel:
    walls_set = frozenset(walls)
    goals_set = frozenset(goals)
    boxes_set = frozenset(boxes_start)

    state = SokobanState(
        width=width,
        height=height,
        walls=walls_set,
        boxes=boxes_set,
        goals=goals_set,
        player=player_start,
        n_boxes=len(boxes_set),
    )
    return SokobanLevel(
        level_id=level_id,
        title=title,
        width=width,
        height=height,
        xsb=state.to_xsb(),
        walls=walls_set,
        boxes_start=boxes_set,
        goals=goals_set,
        player_start=player_start,
        n_boxes=state.n_boxes,
        optimal_moves=None,
        optimal_pushes=None,
        known_optimal=False,
    )


def _verify_forward_solution(level: SokobanLevel, solution: list[Direction]) -> None:
    env = SokobanEnv(
        level,
        illegal_action_behavior="raise",
        detect_deadlocks=False,
        terminal_on_deadlock=False,
    )
    for direction in solution:
        env.step(direction)
    if not env.is_solved():
        raise RuntimeError("procedural generation produced unsolved replay sequence")


def generate_procedural_level_with_solution(
    *,
    width: int,
    height: int,
    n_boxes: int,
    seed: int | None = None,
    level_id: str | None = None,
    title: str | None = None,
    wall_density: float = 0.08,
    scramble_steps: int | None = None,
    max_generation_attempts: int = 64,
) -> tuple[SokobanLevel, list[Direction]]:
    _validate_level_shape(width, height, n_boxes)
    _validate_density(wall_density)
    if max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be >= 1")

    rng = random.Random(seed)
    resolved_level_id = (
        level_id
        if level_id is not None
        else f"procgen:{width}x{height}:b{n_boxes}:s{seed if seed is not None else 'random'}"
    )
    resolved_title = title or f"Procedural {width}x{height} ({n_boxes} boxes)"
    target_steps = (
        int(scramble_steps)
        if scramble_steps is not None
        else max(20, n_boxes * (width + height))
    )
    if target_steps < 1:
        raise ValueError("scramble_steps must be >= 1 when provided")

    for _ in range(max_generation_attempts):
        walls = _sample_walls(
            width=width,
            height=height,
            n_boxes=n_boxes,
            wall_density=wall_density,
            rng=rng,
            max_tries=32,
        )
        open_cells = [
            cell for cell in _interior_cells(width, height) if cell not in walls
        ]
        if len(open_cells) < (n_boxes + 1):
            continue

        goals = set(rng.sample(open_cells, n_boxes))
        player_candidates = [cell for cell in open_cells if cell not in goals]
        if not player_candidates:
            continue

        boxes = set(goals)
        player = rng.choice(player_candidates)
        inverse_forward_actions: list[Direction] = []
        pull_count = 0

        for _ in range(target_steps):
            pulls = _legal_reverse_pulls(
                width=width, height=height, walls=walls, boxes=boxes, player=player
            )
            walks = _legal_reverse_walks(
                width=width, height=height, walls=walls, boxes=boxes, player=player
            )
            if not pulls and not walks:
                break

            prefer_pull = bool(pulls) and (not walks or rng.random() < 0.72)
            if prefer_pull:
                direction = rng.choice(pulls)
                dr, dc = DIRECTION_DELTAS[direction]
                box_pos = (player[0] + dr, player[1] + dc)
                player_next = (player[0] - dr, player[1] - dc)
                boxes.remove(box_pos)
                boxes.add(player)
                player = player_next
                # Inverse of reverse pull is forward push in same direction.
                inverse_forward_actions.append(direction)
                pull_count += 1
                continue

            direction = rng.choice(walks)
            dr, dc = DIRECTION_DELTAS[direction]
            player = (player[0] + dr, player[1] + dc)
            # Inverse of reverse walk is forward walk in opposite direction.
            inverse_forward_actions.append(_OPPOSITE_DIRECTION[direction])

        if pull_count < max(1, n_boxes // 2):
            continue
        if boxes == goals:
            continue

        solution = list(reversed(inverse_forward_actions))
        level = _build_level(
            width=width,
            height=height,
            level_id=resolved_level_id,
            title=resolved_title,
            walls=walls,
            goals=goals,
            boxes_start=boxes,
            player_start=player,
        )
        _verify_forward_solution(level, solution)
        return (level, solution)

    raise RuntimeError(
        "Failed to generate a procedural Sokoban level that satisfies constraints."
    )


def generate_procedural_level(
    *,
    width: int,
    height: int,
    n_boxes: int,
    seed: int | None = None,
    level_id: str | None = None,
    title: str | None = None,
    wall_density: float = 0.08,
    scramble_steps: int | None = None,
    max_generation_attempts: int = 64,
) -> SokobanLevel:
    level, _solution = generate_procedural_level_with_solution(
        width=width,
        height=height,
        n_boxes=n_boxes,
        seed=seed,
        level_id=level_id,
        title=title,
        wall_density=wall_density,
        scramble_steps=scramble_steps,
        max_generation_attempts=max_generation_attempts,
    )
    return level


def generate_procedural_levels(
    *,
    width: int,
    height: int,
    n_boxes: int,
    count: int,
    seed: int | None = 0,
    wall_density: float = 0.08,
    scramble_steps: int | None = None,
    level_id_prefix: str | None = None,
) -> list[SokobanLevel]:
    if count < 1:
        raise ValueError("count must be >= 1")
    prefix = (
        level_id_prefix
        if level_id_prefix is not None
        else (
            f"procgen:{width}x{height}:b{n_boxes}:s{seed if seed is not None else 'random'}"
        )
    )
    levels: list[SokobanLevel] = []
    for idx in range(count):
        level_seed = None if seed is None else int(seed) + idx
        levels.append(
            generate_procedural_level(
                width=width,
                height=height,
                n_boxes=n_boxes,
                seed=level_seed,
                level_id=f"{prefix}:i{idx + 1}",
                title=f"Procedural {width}x{height} ({n_boxes} boxes) #{idx + 1}",
                wall_density=wall_density,
                scramble_steps=scramble_steps,
            )
        )
    return levels
