from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from .env import tool_schemas
from .prompts import default_instructions, with_image_instructions
from .vision import render_sokoban_state_image


def _load_recording(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_path_part(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    if text in {"", ".", ".."}:
        return fallback
    return text


def _extract_run_parts(run_dir: Path) -> tuple[str, str, str]:
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        try:
            run_config = json.loads(run_config_path.read_text())
        except json.JSONDecodeError:
            run_config = {}
        provider = _safe_path_part(run_config.get("provider", "unknown"))
        model = _safe_path_part(run_config.get("model", "unknown"))
        run_id = _safe_path_part(run_config.get("run_id", run_dir.name))
        return (provider, model, run_id)

    provider = _safe_path_part(run_dir.parent.parent.name)
    model = _safe_path_part(run_dir.parent.name)
    run_id = _safe_path_part(run_dir.name, fallback="recording")
    if provider == "unknown" or model == "unknown":
        return ("unknown", "unknown", run_id)
    return (provider, model, run_id)


def _save_png(image_payload: dict[str, Any], out_path: Path) -> None:
    data_b64 = image_payload.get("data_base64")
    if not data_b64:
        raise ValueError("image payload missing data_base64")
    out_path.write_bytes(base64.b64decode(data_b64))


def _normalized_steps(recording: dict[str, Any]) -> list[dict[str, Any]]:
    steps = list(recording.get("steps", []))
    if not steps:
        return steps
    if steps[0].get("action") is None:
        return steps
    metadata = recording.get("metadata", {})
    initial_state = metadata.get("initial_state")
    if not isinstance(initial_state, dict):
        initial_state = steps[0].get("state_before")
    if not isinstance(initial_state, dict):
        return steps
    init_step = {
        "index": 0,
        "state_before": initial_state,
        "state_text": metadata.get("initial_state_text"),
        "state_after": initial_state,
        "action": None,
        "legal": True,
        "totals": {"moves": 0, "illegal_moves": 0, "tool_calls": 0},
    }
    return [init_step, *steps]


def _coerce_state(
    value: Any, *, fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return fallback if fallback is not None else {}


def _state_text(state: dict[str, Any]) -> str:
    width = state.get("width")
    height = state.get("height")
    walls = state.get("walls")
    goals = state.get("goals")
    boxes = state.get("boxes")
    player = state.get("player")

    if not isinstance(width, int) or not isinstance(height, int):
        return json.dumps(state, indent=2, sort_keys=True)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def _positions(value: Any) -> set[tuple[int, int]]:
        if not isinstance(value, list):
            return set()
        output: set[tuple[int, int]] = set()
        for item in value:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], int)
                and isinstance(item[1], int)
            ):
                output.add((item[0], item[1]))
        return output

    wall_set = _positions(walls)
    goal_set = _positions(goals)
    box_set = _positions(boxes)

    for row, col in wall_set:
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "#"
    for row, col in goal_set:
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "."
    for row, col in box_set:
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "*" if (row, col) in goal_set else "$"
    if (
        isinstance(player, (list, tuple))
        and len(player) == 2
        and isinstance(player[0], int)
        and isinstance(player[1], int)
    ):
        row, col = player
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "+" if (row, col) in goal_set else "@"

    return "\n".join("".join(row) for row in grid)


def _render_image(
    state: dict[str, Any],
    *,
    tile_size: int,
    label_grid: bool,
    background: str,
) -> dict[str, Any]:
    image = render_sokoban_state_image(
        state,
        tile_size=tile_size,
        label_grid=label_grid,
        background=background,
    )
    return {
        "mime_type": image.mime_type,
        "data_base64": image.data_base64,
        "width": image.width,
        "height": image.height,
    }


def _html_template(payload: dict[str, Any]) -> str:
    data = json.dumps(payload)
    template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Sokoban Review</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    .controls button { margin-right: 8px; }
    pre { background: #f7f7f7; padding: 10px; overflow: auto; }
    img { max-width: 100%; border: 1px solid #ccc; }
    .meta { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Sokoban Review</h2>
  <div class="meta" id="meta"></div>
  <div class="controls">
    <button onclick="prevStep()">Prev</button>
    <button onclick="nextStep()">Next</button>
    <input type="range" id="slider" min="0" value="0" step="1" />
  </div>
  <div class="grid">
    <div class="panel">
      <h3>State (Before)</h3>
      <img id="imgBefore" />
    </div>
    <div class="panel">
      <h3>State (After)</h3>
      <img id="imgAfter" />
    </div>
  </div>
  <div class="grid">
    <div class="panel">
      <h3>Prompt</h3>
      <pre id="prompt"></pre>
    </div>
    <div class="panel">
      <h3>Action</h3>
      <pre id="action"></pre>
      <h3>Totals</h3>
      <pre id="totals"></pre>
    </div>
  </div>
  <script>
    const data = __DATA__;
    const steps = data.steps || [];
    const slider = document.getElementById("slider");
    const meta = document.getElementById("meta");
    const imgBefore = document.getElementById("imgBefore");
    const imgAfter = document.getElementById("imgAfter");
    const promptEl = document.getElementById("prompt");
    const actionEl = document.getElementById("action");
    const totalsEl = document.getElementById("totals");
    let idx = 0;
    slider.max = Math.max(steps.length - 1, 0);

    function render() {
      if (!steps.length) return;
      const step = steps[idx];
      meta.textContent = JSON.stringify(data.metadata || {}) + ` | step ${step.index}`;
      imgBefore.src = step.image_before || "";
      imgAfter.src = step.image_after || "";
      promptEl.textContent = step.prompt || "";
      actionEl.textContent = JSON.stringify(step.action || {}, null, 2);
      totalsEl.textContent = JSON.stringify(step.totals || {}, null, 2);
      slider.value = idx;
    }
    function nextStep() { idx = Math.min(idx + 1, steps.length - 1); render(); }
    function prevStep() { idx = Math.max(idx - 1, 0); render(); }
    slider.addEventListener("input", (e) => {
      idx = parseInt(e.target.value, 10);
      render();
    });
    render();
  </script>
</body>
</html>"""
    return template.replace("__DATA__", data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a manual review bundle for a Sokoban run."
    )
    parser.add_argument("--run-dir", help="Run directory containing recordings/")
    parser.add_argument("--recording", help="Path to a specific recording json")
    parser.add_argument("--out-dir", default="artifacts/reviews/sokoban")
    parser.add_argument("--episode-id", type=int, action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-tile-size", type=int, default=None)
    parser.add_argument("--image-background", default=None)
    parser.add_argument("--no-image-labels", action="store_true")

    args = parser.parse_args()
    if not args.run_dir and not args.recording:
        raise SystemExit("Provide --run-dir or --recording.")

    recordings: list[Path] = []
    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.recording:
        recordings.append(Path(args.recording))
    elif run_dir:
        recordings = sorted((run_dir / "recordings").glob("episode_*.json"))

    if args.episode_id:
        wanted = {int(value) for value in args.episode_id}
        recordings = [
            path for path in recordings if int(path.stem.split("_")[-1]) in wanted
        ]
    if args.max_episodes is not None:
        recordings = recordings[: args.max_episodes]

    if not recordings:
        raise SystemExit("No recordings found.")

    if run_dir:
        provider, model, run_id = _extract_run_parts(run_dir)
        run_config_path = run_dir / "run_config.json"
        run_config = (
            json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
        )
    else:
        provider, model, run_id = ("unknown", "unknown", "recording")
        run_config = {}

    prompt_variants = {
        variant["name"]: variant for variant in run_config.get("prompt_variants", [])
    }
    tool_variants = {
        variant["name"]: variant for variant in run_config.get("tool_variants", [])
    }
    state_format = run_config.get("state_format", "text")

    tile_size = args.image_tile_size
    if tile_size is None:
        tile_size = int(run_config.get("image_tile_size", 48))
    image_background = args.image_background or run_config.get(
        "image_background", "white"
    )
    if args.no_image_labels:
        label_grid = False
    else:
        label_grid = bool(run_config.get("image_labels", True))

    out_base = Path(args.out_dir) / provider / model / run_id
    out_base.mkdir(parents=True, exist_ok=True)

    for recording_path in recordings:
        recording = _load_recording(recording_path)
        metadata = recording.get("metadata", {})
        episode_id = metadata.get("episode_id", recording_path.stem)
        episode_dir = out_base / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        prompt_variant_name = metadata.get("prompt_variant")
        tool_variant_name = metadata.get("tools_variant")
        prompt_variant = prompt_variants.get(prompt_variant_name, {})
        tool_variant = tool_variants.get(tool_variant_name, {})

        instructions = prompt_variant.get("instructions") or default_instructions()
        if state_format in {"image", "both"}:
            instructions = with_image_instructions(instructions)

        allowed_tools = tool_variant.get("allowed_tools")
        schemas = tool_schemas()
        if isinstance(allowed_tools, list):
            allowed_set = set(allowed_tools)
            schemas = [schema for schema in schemas if schema["name"] in allowed_set]

        steps = _normalized_steps(recording)
        initial_state = _coerce_state(metadata.get("initial_state"))
        if not initial_state and steps:
            initial_state = _coerce_state(steps[0].get("state_before"))
        current_state = initial_state

        rendered_steps: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            state_before = _coerce_state(
                step.get("state_before"), fallback=current_state
            )
            state_after = _coerce_state(step.get("state_after"), fallback=state_before)
            current_state = state_after

            image_before = _render_image(
                state_before,
                tile_size=tile_size,
                label_grid=label_grid,
                background=image_background,
            )
            image_after = _render_image(
                state_after,
                tile_size=tile_size,
                label_grid=label_grid,
                background=image_background,
            )
            before_path = episode_dir / f"state_before_{index:04d}.png"
            after_path = episode_dir / f"state_after_{index:04d}.png"
            _save_png(image_before, before_path)
            _save_png(image_after, after_path)

            step_state_text = step.get("state_text")
            if isinstance(step_state_text, (dict, list)):
                step_state_text = json.dumps(step_state_text, indent=2, sort_keys=True)
            if not isinstance(step_state_text, str) or not step_state_text.strip():
                step_state_text = _state_text(state_before)

            prompt = (
                f"{instructions}\n\nSTATE:\n{step_state_text}\n\nTOOLS:\n"
                f"{json.dumps(schemas, indent=2)}"
            )
            if state_format in {"image", "both"}:
                prompt = f"{prompt}\n\nNOTE: State image attached."

            rendered_steps.append(
                {
                    "index": index,
                    "original_index": step.get("index"),
                    "state_text": step_state_text,
                    "action": step.get("action"),
                    "legal": step.get("legal"),
                    "totals": step.get("totals"),
                    "image_before": before_path.name,
                    "image_after": after_path.name,
                    "prompt": prompt,
                }
            )

        payload = {
            "metadata": {
                **metadata,
                "provider": provider,
                "model": model,
                "run_id": run_id,
                "state_format": state_format,
            },
            "steps": rendered_steps,
        }
        (episode_dir / "index.html").write_text(_html_template(payload))

    print(f"Review bundle written to: {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
