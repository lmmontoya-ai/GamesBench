from __future__ import annotations

import base64
import colorsys
import io
from typing import Iterable, Sequence

from .env import HanoiState, TowerOfHanoiEnv
from ..vision_types import StateImage


def render_hanoi_image(
    *,
    pegs: Sequence[Sequence[int]],
    n_disks: int,
    size: tuple[int, int] = (640, 360),
    label_pegs: bool = True,
    background: str = "white",
) -> StateImage:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing pillow. Install with: pip install 'games-bench[viz]' "
            "or uv sync --group viz"
        ) from exc

    width, height = size
    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    peg_count = max(len(pegs), 1)
    margin_x = max(50, width // 8)
    peg_y_top = int(height * 0.2)
    peg_y_bottom = int(height * 0.82)
    span = max(1, peg_count - 1)
    peg_x_positions = [
        int(margin_x + i * (width - 2 * margin_x) / span) for i in range(peg_count)
    ]

    if label_pegs:
        for i, x in enumerate(peg_x_positions):
            label = f"Peg {i}"
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = bbox[2] - bbox[0]
            except Exception:
                label_w = 0
            draw.text(
                (x - label_w / 2, int(height * 0.06)),
                label,
                fill="black",
                font=font,
            )

    disk_h = max(10, int(height * 0.05))
    min_w = max(30, int(width * 0.08))
    max_w = max(120, int(width * 0.28))
    base_height = max(10, int(height * 0.03))
    base_top = min(height - base_height - 6, peg_y_bottom + 6)
    base_bottom = base_top + base_height
    draw.rectangle(
        [margin_x - 30, base_top, width - margin_x + 30, base_bottom], fill="#1f2937"
    )

    for i, peg in enumerate(pegs):
        x = peg_x_positions[i]
        draw.line((x, peg_y_top, x, peg_y_bottom), fill="#6b7280", width=4)
        for j, disk in enumerate(peg):
            ratio = (disk - 1) / (n_disks - 1) if n_disks > 1 else 1
            w = min_w + ratio * (max_w - min_w)
            x0 = x - w / 2
            x1 = x + w / 2
            y1 = peg_y_bottom - j * (disk_h + 4)
            y0 = y1 - disk_h
            hue = 0.6 - 0.55 * ratio
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.65)
            color = (int(r * 255), int(g * 255), int(b * 255))
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="#111827")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data = buffer.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    return StateImage(
        mime_type="image/png",
        data_base64=b64,
        data_url=f"data:image/png;base64,{b64}",
        width=width,
        height=height,
    )


def render_hanoi_state_image(
    state: HanoiState | dict[str, object],
    *,
    size: tuple[int, int] = (640, 360),
    label_pegs: bool = True,
    background: str = "white",
) -> StateImage:
    if isinstance(state, HanoiState):
        pegs = state.pegs
        n_disks = state.n_disks
    else:
        pegs = state.get("pegs")  # type: ignore[assignment]
        n_disks = state.get("n_disks")  # type: ignore[assignment]
    if not isinstance(pegs, Iterable):
        raise ValueError("state.pegs is required to render image")
    if not isinstance(n_disks, int):
        raise ValueError("state.n_disks is required to render image")
    peg_list = [list(peg) for peg in pegs]
    return render_hanoi_image(
        pegs=peg_list,
        n_disks=n_disks,
        size=size,
        label_pegs=label_pegs,
        background=background,
    )


def render_hanoi_env_image(
    env: TowerOfHanoiEnv,
    *,
    size: tuple[int, int] = (640, 360),
    label_pegs: bool = True,
    background: str = "white",
) -> StateImage:
    return render_hanoi_state_image(
        env.get_state(),
        size=size,
        label_pegs=label_pegs,
        background=background,
    )
