from __future__ import annotations

import base64
import io
from collections.abc import Iterable, Mapping

from .env import SokobanEnv, SokobanState
from ..vision_types import StateImage


def _coerce_position(value: object, *, field_name: str) -> tuple[int, int]:
    if not isinstance(value, Iterable):
        raise ValueError(f"{field_name} must contain [row, col] integer pairs")
    parts = list(value)
    if len(parts) != 2 or not all(isinstance(part, int) for part in parts):
        raise ValueError(f"{field_name} must contain [row, col] integer pairs")
    return (parts[0], parts[1])


def _coerce_positions(value: object, *, field_name: str) -> frozenset[tuple[int, int]]:
    if not isinstance(value, Iterable):
        raise ValueError(f"{field_name} must be an iterable of [row, col] pairs")
    return frozenset(
        _coerce_position(position, field_name=field_name) for position in value
    )


def _state_from_mapping(state: Mapping[str, object]) -> SokobanState:
    width = state.get("width")
    height = state.get("height")
    walls = state.get("walls")
    boxes = state.get("boxes")
    goals = state.get("goals")
    player = state.get("player")
    n_boxes = state.get("n_boxes")

    if not isinstance(width, int) or width < 1:
        raise ValueError("state.width must be a positive integer")
    if not isinstance(height, int) or height < 1:
        raise ValueError("state.height must be a positive integer")
    if not isinstance(boxes, Iterable):
        raise ValueError("state.boxes must be an iterable of [row, col] pairs")
    boxes_list = list(boxes)
    if n_boxes is None:
        n_boxes = len(boxes_list)
    if not isinstance(n_boxes, int) or n_boxes < 1:
        raise ValueError("state.n_boxes must be a positive integer")

    coerced_walls = _coerce_positions(walls, field_name="state.walls")
    coerced_boxes = _coerce_positions(boxes_list, field_name="state.boxes")
    coerced_goals = _coerce_positions(goals, field_name="state.goals")
    coerced_player = _coerce_position(player, field_name="state.player")

    return SokobanState(
        width=width,
        height=height,
        walls=coerced_walls,
        boxes=coerced_boxes,
        goals=coerced_goals,
        player=coerced_player,
        n_boxes=n_boxes,
    )


def _safe_inset(tile_size: int, desired: int) -> int:
    # Keep inner geometry non-inverted for small tiles.
    return min(max(desired, 0), max(0, (tile_size - 1) // 2))


def render_sokoban_image(
    state: SokobanState,
    *,
    tile_size: int = 48,
    label_grid: bool = True,
    background: str = "white",
) -> StateImage:
    if tile_size < 8:
        raise ValueError("tile_size must be >= 8")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing pillow. Install with: pip install 'games-bench[viz]' "
            "or uv sync --group viz"
        ) from exc

    board_width = state.width * tile_size
    board_height = state.height * tile_size
    outer_pad = max(2, tile_size // 12)
    top_gutter = tile_size if label_grid else 0
    left_gutter = tile_size if label_grid else 0

    width = left_gutter + board_width + outer_pad * 2
    height = top_gutter + board_height + outer_pad * 2
    origin_x = left_gutter + outer_pad
    origin_y = top_gutter + outer_pad

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    colors = {
        "floor": "#f4efe2",
        "wall": "#2f3542",
        "goal": "#ffd166",
        "box": "#9c6644",
        "box_on_goal": "#2a9d8f",
        "player": "#e63946",
        "player_on_goal": "#5e60ce",
        "grid": "#d9d2c5",
        "border": "#7a7468",
        "text": "#1f2937",
    }

    for row in range(state.height):
        for col in range(state.width):
            x0 = origin_x + col * tile_size
            y0 = origin_y + row * tile_size
            x1 = x0 + tile_size - 1
            y1 = y0 + tile_size - 1
            draw.rectangle((x0, y0, x1, y1), fill=colors["floor"])

    for row, col in sorted(state.goals):
        x0 = origin_x + col * tile_size
        y0 = origin_y + row * tile_size
        x1 = x0 + tile_size - 1
        y1 = y0 + tile_size - 1
        inset = _safe_inset(tile_size, max(1, tile_size // 4))
        draw.ellipse(
            (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
            fill=colors["goal"],
            outline=colors["border"],
        )

    for row, col in sorted(state.walls):
        x0 = origin_x + col * tile_size
        y0 = origin_y + row * tile_size
        x1 = x0 + tile_size - 1
        y1 = y0 + tile_size - 1
        draw.rectangle((x0, y0, x1, y1), fill=colors["wall"], outline="#1b1f28")

    for row, col in sorted(state.boxes):
        x0 = origin_x + col * tile_size
        y0 = origin_y + row * tile_size
        x1 = x0 + tile_size - 1
        y1 = y0 + tile_size - 1
        inset = _safe_inset(tile_size, max(2, tile_size // 8))
        box_color = (
            colors["box_on_goal"] if (row, col) in state.goals else colors["box"]
        )
        draw.rectangle(
            (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
            fill=box_color,
            outline=colors["border"],
        )
        draw.line(
            (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
            fill=colors["border"],
            width=1,
        )
        draw.line(
            (x0 + inset, y1 - inset, x1 - inset, y0 + inset),
            fill=colors["border"],
            width=1,
        )

    player_row, player_col = state.player
    px0 = origin_x + player_col * tile_size
    py0 = origin_y + player_row * tile_size
    px1 = px0 + tile_size - 1
    py1 = py0 + tile_size - 1
    player_inset = _safe_inset(tile_size, max(3, tile_size // 5))
    player_color = (
        colors["player_on_goal"]
        if (player_row, player_col) in state.goals
        else colors["player"]
    )
    draw.ellipse(
        (
            px0 + player_inset,
            py0 + player_inset,
            px1 - player_inset,
            py1 - player_inset,
        ),
        fill=player_color,
        outline=colors["border"],
    )

    for row in range(state.height + 1):
        y = origin_y + row * tile_size
        draw.line(
            (origin_x, y, origin_x + board_width, y), fill=colors["grid"], width=1
        )
    for col in range(state.width + 1):
        x = origin_x + col * tile_size
        draw.line(
            (x, origin_y, x, origin_y + board_height), fill=colors["grid"], width=1
        )

    draw.rectangle(
        (
            origin_x,
            origin_y,
            origin_x + board_width,
            origin_y + board_height,
        ),
        outline=colors["border"],
        width=1,
    )

    if label_grid:
        for col in range(state.width):
            label = str(col)
            x = origin_x + col * tile_size + tile_size // 2
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = bbox[2] - bbox[0]
                label_h = bbox[3] - bbox[1]
            except Exception:
                label_w = 0
                label_h = 0
            draw.text(
                (
                    x - label_w / 2,
                    max(0, origin_y - top_gutter + (top_gutter - label_h) / 2),
                ),
                label,
                fill=colors["text"],
                font=font,
            )
        for row in range(state.height):
            label = str(row)
            y = origin_y + row * tile_size + tile_size // 2
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = bbox[2] - bbox[0]
                label_h = bbox[3] - bbox[1]
            except Exception:
                label_w = 0
                label_h = 0
            draw.text(
                (
                    max(0, origin_x - left_gutter + (left_gutter - label_w) / 2),
                    y - label_h / 2,
                ),
                label,
                fill=colors["text"],
                font=font,
            )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data = buffer.getvalue()
    data_base64 = base64.b64encode(data).decode("ascii")
    return StateImage(
        mime_type="image/png",
        data_base64=data_base64,
        data_url=f"data:image/png;base64,{data_base64}",
        width=width,
        height=height,
    )


def render_sokoban_state_image(
    state: SokobanState | Mapping[str, object],
    **kwargs: object,
) -> StateImage:
    resolved_state = (
        state if isinstance(state, SokobanState) else _state_from_mapping(state)
    )
    return render_sokoban_image(resolved_state, **kwargs)


def render_sokoban_env_image(env: SokobanEnv, **kwargs: object) -> StateImage:
    return render_sokoban_state_image(env.get_state(), **kwargs)
