from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def _position(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    row, col = value
    if not isinstance(row, int) or not isinstance(col, int):
        return None
    return (row, col)


def _positions(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        return set()
    result: set[tuple[int, int]] = set()
    for item in value:
        pos = _position(item)
        if pos is not None:
            result.add(pos)
    return result


def _state_payload(step: dict[str, Any]) -> dict[str, Any]:
    state_after = step.get("state_after")
    if isinstance(state_after, dict):
        return state_after
    state_before = step.get("state_before")
    if isinstance(state_before, dict):
        return state_before
    return {}


def _state_to_xsb(state: dict[str, Any]) -> str:
    width = state.get("width")
    height = state.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return "<missing state dimensions>"

    walls = _positions(state.get("walls"))
    boxes = _positions(state.get("boxes"))
    goals = _positions(state.get("goals"))
    player = _position(state.get("player"))

    rows = [[" " for _ in range(width)] for _ in range(height)]
    for row, col in walls:
        if 0 <= row < height and 0 <= col < width:
            rows[row][col] = "#"
    for row, col in goals:
        if 0 <= row < height and 0 <= col < width:
            rows[row][col] = "."
    for row, col in boxes:
        if 0 <= row < height and 0 <= col < width:
            rows[row][col] = "*" if (row, col) in goals else "$"
    if player is not None:
        row, col = player
        if 0 <= row < height and 0 <= col < width:
            rows[row][col] = "+" if (row, col) in goals else "@"
    return "\n".join("".join(row) for row in rows)


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


def _render_ascii(recording: dict[str, Any]) -> str:
    lines: list[str] = []
    for index, step in enumerate(_normalized_steps(recording)):
        totals = step.get("totals", {})
        lines.append(
            f"Step {index}: moves={totals.get('moves', 0)} "
            f"illegal={totals.get('illegal_moves', 0)} tool_calls={totals.get('tool_calls', 0)}"
        )
        lines.append(f"Action: {step.get('action')}")
        lines.append(_state_to_xsb(_state_payload(step)))
        lines.append("")
    return "\n".join(lines)


def _render_html(recording: dict[str, Any]) -> str:
    payload = {
        "metadata": recording.get("metadata", {}),
        "steps": [
            {
                "index": index,
                "action": step.get("action"),
                "totals": step.get("totals", {}),
                "xsb": _state_to_xsb(_state_payload(step)),
            }
            for index, step in enumerate(_normalized_steps(recording))
        ],
    }
    data = json.dumps(payload)
    template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Sokoban Playback</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    .controls button { margin-right: 8px; }
    pre { background: #f7f7f7; padding: 10px; overflow: auto; }
    .meta { color: #666; font-size: 12px; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
  </style>
</head>
<body>
  <h2>Sokoban Playback</h2>
  <div class="meta" id="meta"></div>
  <div class="controls">
    <button onclick="prevStep()">Prev</button>
    <button onclick="nextStep()">Next</button>
    <input type="range" id="slider" min="0" value="0" step="1" />
  </div>
  <div class="grid">
    <div class="panel">
      <h3>Board</h3>
      <pre id="board"></pre>
    </div>
    <div class="panel">
      <h3>Action</h3>
      <pre id="action"></pre>
      <h3>Totals</h3>
      <pre id="totals"></pre>
    </div>
  </div>
  <script>
    const payload = __DATA__;
    const steps = payload.steps || [];
    const slider = document.getElementById("slider");
    const meta = document.getElementById("meta");
    const board = document.getElementById("board");
    const actionEl = document.getElementById("action");
    const totalsEl = document.getElementById("totals");
    let idx = 0;
    slider.max = Math.max(steps.length - 1, 0);

    function render() {
      if (!steps.length) return;
      const step = steps[idx];
      meta.textContent = JSON.stringify(payload.metadata || {}) + ` | step ${step.index}`;
      board.textContent = step.xsb || "";
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
    parser = argparse.ArgumentParser(description="Render Sokoban recordings.")
    parser.add_argument("--run-dir", help="Run directory containing recordings/")
    parser.add_argument("--recording", help="Path to a specific recording json")
    parser.add_argument("--out-dir", default="artifacts/renders/sokoban")
    parser.add_argument("--format", choices=["html", "ascii"], default="html")
    parser.add_argument("--episode-id", type=int, action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=None)

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
    else:
        provider, model, run_id = ("unknown", "unknown", "recording")

    out_base = Path(args.out_dir) / provider / model / run_id
    out_base.mkdir(parents=True, exist_ok=True)

    for recording_path in recordings:
        recording = _load_recording(recording_path)
        episode_id = recording.get("metadata", {}).get(
            "episode_id", recording_path.stem
        )
        episode_dir = out_base / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        if args.format == "ascii":
            (episode_dir / "playback.txt").write_text(_render_ascii(recording))
        else:
            (episode_dir / "index.html").write_text(_render_html(recording))

    print(f"Rendered to: {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
