from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from games_bench.games.hanoi.env import tool_schemas
from games_bench.games.hanoi.prompts import (
    default_instructions,
    format_instructions,
    with_image_instructions,
)
from games_bench.games.hanoi.vision import render_hanoi_state_image


def _load_recording(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _extract_run_parts(run_dir: Path) -> tuple[str, str, str]:
    parts = run_dir.parts
    if len(parts) < 3:
        return ("unknown", "unknown", run_dir.name)
    return (parts[-3], parts[-2], parts[-1])


def _normalized_steps(recording: dict[str, Any]) -> list[dict[str, Any]]:
    steps = list(recording.get("steps", []))
    if not steps:
        return steps
    if steps[0].get("action") is None:
        return steps
    meta = recording.get("metadata", {})
    initial_state = meta.get("initial_state") or steps[0].get("state_before")
    initial_text = meta.get("initial_state_text")
    if initial_state is None:
        return steps
    init_step = {
        "index": 0,
        "state_before": initial_state,
        "state_text": initial_text,
        "state_after": initial_state,
        "action": None,
        "legal": True,
        "totals": {"moves": 0, "illegal_moves": 0, "tool_calls": 0},
    }
    return [init_step, *steps]


def _initial_state(n_disks: int, start_peg: int = 0) -> dict[str, Any]:
    pegs = [[] for _ in range(3)]
    pegs[start_peg] = list(range(n_disks, 0, -1))
    return {
        "n_disks": n_disks,
        "pegs": pegs,
        "disk_positions": [start_peg for _ in range(n_disks)],
    }


def _save_png(image_payload: dict[str, Any], out_path: Path) -> None:
    data_b64 = image_payload.get("data_base64")
    if not data_b64:
        raise ValueError("image payload missing data_base64")
    out_path.write_bytes(base64.b64decode(data_b64))


def _render_image(
    state: dict[str, Any],
    *,
    size: tuple[int, int],
    label_pegs: bool,
    background: str,
) -> dict[str, Any]:
    image = render_hanoi_state_image(
        state,
        size=size,
        label_pegs=label_pegs,
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
  <title>Hanoi Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ border: 1px solid #ddd; padding: 12px; border-radius: 8px; }}
    .controls button {{ margin-right: 8px; }}
    pre {{ background: #f7f7f7; padding: 10px; overflow: auto; }}
    img {{ max-width: 100%; border: 1px solid #ccc; }}
    .meta {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h2>Hanoi Review</h2>
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
    const slider = document.getElementById('slider');
    const meta = document.getElementById('meta');
    const imgBefore = document.getElementById('imgBefore');
    const imgAfter = document.getElementById('imgAfter');
    const promptEl = document.getElementById('prompt');
    const actionEl = document.getElementById('action');
    const totalsEl = document.getElementById('totals');
    let idx = 0;
    slider.max = Math.max(steps.length - 1, 0);

    function render() {{
      if (!steps.length) return;
      const step = steps[idx];
      meta.textContent = JSON.stringify(data.metadata || {}) + ` | step ${step.index}`;
      imgBefore.src = step.image_before || '';
      imgAfter.src = step.image_after || '';
      promptEl.textContent = step.prompt || '';
      actionEl.textContent = JSON.stringify(step.action || {{}}, null, 2);
      totalsEl.textContent = JSON.stringify(step.totals || {{}}, null, 2);
      slider.value = idx;
    }}
    function nextStep() {{ idx = Math.min(idx + 1, steps.length - 1); render(); }}
    function prevStep() {{ idx = Math.max(idx - 1, 0); render(); }}
    slider.addEventListener('input', (e) => {{
      idx = parseInt(e.target.value, 10);
      render();
    }});
    render();
  </script>
</body>
</html>"""
    return template.replace("__DATA__", data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a manual review bundle for a Hanoi run."
    )
    parser.add_argument("--run-dir", help="Run directory containing recordings/")
    parser.add_argument("--recording", help="Path to a specific recording json")
    parser.add_argument("--out-dir", default="artifacts/reviews/hanoi")
    parser.add_argument("--episode-id", type=int, action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-size", default="640x360")
    parser.add_argument("--image-background", default="white")
    parser.add_argument("--no-image-labels", action="store_true")

    args = parser.parse_args()
    if not args.run_dir and not args.recording:
        raise SystemExit("Provide --run-dir or --recording.")

    recordings: list[Path] = []
    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.recording:
        recordings.append(Path(args.recording))
    elif run_dir:
        recordings_dir = run_dir / "recordings"
        recordings = sorted(recordings_dir.glob("episode_*.json"))

    if args.episode_id:
        wanted = {int(x) for x in args.episode_id}
        recordings = [p for p in recordings if int(p.stem.split("_")[-1]) in wanted]
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

    prompt_variants = {v["name"]: v for v in run_config.get("prompt_variants", [])}
    tool_variants = {v["name"]: v for v in run_config.get("tool_variants", [])}
    state_format = run_config.get("state_format", "text")

    size_parts = args.image_size.split("x")
    size = (int(size_parts[0]), int(size_parts[1]))
    label_pegs = not args.no_image_labels

    out_base = Path(args.out_dir) / provider / model / run_id
    out_base.mkdir(parents=True, exist_ok=True)

    for rec_path in recordings:
        recording = _load_recording(rec_path)
        metadata = recording.get("metadata", {})
        episode_id = metadata.get("episode_id", rec_path.stem)
        episode_dir = out_base / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        prompt_variant_name = metadata.get("prompt_variant")
        tool_variant_name = metadata.get("tools_variant")
        prompt_variant = prompt_variants.get(prompt_variant_name, {})
        tool_variant = tool_variants.get(tool_variant_name, {})

        start_peg = run_config.get("start_peg", 0)
        goal_peg = run_config.get("goal_peg", 2)
        instructions = prompt_variant.get("instructions") or default_instructions(
            start_peg=start_peg, goal_peg=goal_peg
        )
        instructions = format_instructions(
            instructions, start_peg=start_peg, goal_peg=goal_peg
        )
        if state_format in {"image", "both"}:
            instructions = with_image_instructions(instructions)
        allowed_tools = tool_variant.get("allowed_tools")
        schemas = tool_schemas()
        if allowed_tools:
            allowed_set = set(allowed_tools)
            schemas = [t for t in schemas if t["name"] in allowed_set]

        n_disks = metadata.get("n_disks") or recording.get("summary", {}).get("n_disks")
        if not isinstance(n_disks, int):
            raise SystemExit("Missing n_disks for recording; cannot render images.")

        steps = _normalized_steps(recording)
        initial_snapshot = metadata.get("initial_state")
        if not isinstance(initial_snapshot, dict):
            initial_snapshot = None
        if steps and isinstance(steps[0].get("state_before"), dict):
            initial_snapshot = initial_snapshot or steps[0].get("state_before")
        current_state = initial_snapshot or _initial_state(n_disks)

        rendered_steps = []
        for idx, step in enumerate(steps):
            step_index = idx
            state_before = (
                step.get("state_before")
                if isinstance(step.get("state_before"), dict)
                else current_state
            )
            state_after = (
                step.get("state_after")
                if isinstance(step.get("state_after"), dict)
                else state_before
            )
            current_state = state_after

            image_before = _render_image(
                state_before,
                size=size,
                label_pegs=label_pegs,
                background=args.image_background,
            )
            image_after = _render_image(
                state_after,
                size=size,
                label_pegs=label_pegs,
                background=args.image_background,
            )
            before_path = episode_dir / f"state_before_{step_index:04d}.png"
            after_path = episode_dir / f"state_after_{step_index:04d}.png"
            _save_png(image_before, before_path)
            _save_png(image_after, after_path)

            state_text = step.get("state_text")
            if state_text is None:
                state_text = step.get("state_before", "")
            if isinstance(state_text, (dict, list)):
                state_text = json.dumps(state_text, indent=2, sort_keys=True)
            prompt = (
                f"{instructions}\n\nSTATE:\n{state_text}\n\nTOOLS:\n"
                f"{json.dumps(schemas, indent=2)}"
            )
            if state_format in {"image", "both"}:
                prompt = prompt + "\n\nNOTE: State image attached."

            rendered_steps.append(
                {
                    "index": step_index,
                    "original_index": step.get("index"),
                    "state_text": state_text,
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
