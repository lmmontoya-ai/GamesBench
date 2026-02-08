from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _load_recording(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_for_html_script(value: Any) -> str:
    return json.dumps(value).replace("</", "<\\/")


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


def _normalized_steps(recording: dict[str, Any]) -> list[dict[str, Any]]:
    steps = list(recording.get("steps", []))
    if not steps:
        return steps
    if steps[0].get("action") is None:
        return steps
    meta = recording.get("metadata", {})
    initial_state = meta.get("initial_state") or steps[0].get("state_before")
    if initial_state is None:
        n_disks = meta.get("n_disks")
        if isinstance(n_disks, int):
            initial_state = {
                "n_disks": n_disks,
                "pegs": [list(range(n_disks, 0, -1)), [], []],
                "disk_positions": [0 for _ in range(n_disks)],
            }
    if initial_state is None:
        return steps
    init_step = {
        "index": 0,
        "state_before": initial_state,
        "state_after": initial_state,
        "action": None,
        "legal": True,
        "totals": {"moves": 0, "illegal_moves": 0, "tool_calls": 0},
    }
    return [init_step, *steps]


def _normalized_recording(recording: dict[str, Any]) -> dict[str, Any]:
    steps = _normalized_steps(recording)
    for idx, step in enumerate(steps):
        step["render_index"] = idx
    return {**recording, "steps": steps}


def _episode_id_from_path(path: Path) -> int | None:
    suffix = path.stem.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _render_ascii(recording: dict[str, Any]) -> str:
    lines = []
    steps = _normalized_steps(recording)
    for idx, step in enumerate(steps):
        totals = step.get("totals", {})
        state = step.get("state_after") or step.get("state_before") or {}
        pegs = (
            state.get("state", {}).get("pegs")
            if "state" in state
            else state.get("pegs")
        )
        lines.append(
            f"Step {idx}: moves={totals.get('moves', 0)} "
            f"illegal={totals.get('illegal_moves', 0)} tool_calls={totals.get('tool_calls', 0)}"
        )
        lines.append(f"Action: {step.get('action')}")
        if pegs is None:
            lines.append("Pegs: <missing>")
        else:
            for i, peg in enumerate(pegs):
                lines.append(f"Peg {i}: {peg}")
        lines.append("")
    return "\n".join(lines)


def _render_html(recording: dict[str, Any]) -> str:
    data = _json_for_html_script(_normalized_recording(recording))
    template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Hanoi Playback</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    .controls button {{ margin-right: 8px; }}
    .pegs {{ display: flex; gap: 24px; margin-top: 16px; }}
    .peg {{ flex: 1; border: 1px solid #ccc; padding: 8px; min-height: 220px; }}
    .disk {{ height: 16px; margin: 4px auto; border-radius: 4px; background: #4a90e2; }}
    .stats {{ margin-top: 8px; }}
    .meta {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h2>Hanoi Playback</h2>
  <div class="meta" id="meta"></div>
  <div class="controls">
    <button onclick="prevStep()">Prev</button>
    <button onclick="togglePlay()" id="playBtn">Play</button>
    <button onclick="nextStep()">Next</button>
    <input type="range" id="slider" min="0" value="0" step="1" />
  </div>
  <div class="stats" id="stats"></div>
  <div class="pegs" id="pegs"></div>

  <script>
    const recording = __RECORDING_DATA__;
    const steps = recording.steps || [];
    const slider = document.getElementById('slider');
    const stats = document.getElementById('stats');
    const pegsEl = document.getElementById('pegs');
    const meta = document.getElementById('meta');
    const playBtn = document.getElementById('playBtn');
    let idx = 0;
    let interval = null;

    slider.max = Math.max(steps.length - 1, 0);

    function getState(step) {{
      return step.state_after || step.state_before || {{}};
    }}

    function render() {{
      if (!steps.length) return;
      const step = steps[idx];
      const totals = step.totals || {{}};
      const stateObj = getState(step);
      const state = stateObj.state || stateObj;
      const pegs = state.pegs || [[], [], []];
      const nDisks = state.n_disks || Math.max(...pegs.flat(), 1);

      meta.textContent = JSON.stringify(recording.metadata || {{}});
      const renderIndex = step.render_index ?? idx;
      stats.textContent = `Step ${renderIndex} | moves=${totals.moves} illegal=${totals.illegal_moves} tool_calls=${totals.tool_calls}`;
      pegsEl.innerHTML = '';
      pegs.forEach((peg, i) => {{
        const pegDiv = document.createElement('div');
        pegDiv.className = 'peg';
        const title = document.createElement('div');
        title.textContent = `Peg ${i}`;
        pegDiv.appendChild(title);
        const stack = document.createElement('div');
        peg.forEach(disk => {{
          const diskDiv = document.createElement('div');
          const ratio = nDisks > 1 ? (disk - 1) / (nDisks - 1) : 1;
          const width = 40 + ratio * 120;
          diskDiv.className = 'disk';
          diskDiv.style.width = `${width}px`;
          stack.appendChild(diskDiv);
        }});
        pegDiv.appendChild(stack);
        pegsEl.appendChild(pegDiv);
      }});
      slider.value = idx;
    }}

    function nextStep() {{
      idx = Math.min(idx + 1, steps.length - 1);
      render();
    }}
    function prevStep() {{
      idx = Math.max(idx - 1, 0);
      render();
    }}
    function togglePlay() {{
      if (interval) {{
        clearInterval(interval);
        interval = null;
        playBtn.textContent = 'Play';
      }} else {{
        interval = setInterval(() => {{
          if (idx >= steps.length - 1) {{
            togglePlay();
            return;
          }}
          nextStep();
        }}, 600);
        playBtn.textContent = 'Pause';
      }}
    }}
    slider.addEventListener('input', (e) => {{
      idx = parseInt(e.target.value, 10);
      render();
    }});
    render();
  </script>
</body>
</html>"""
    return template.replace("__RECORDING_DATA__", data)


def _render_video(recording: dict[str, Any], out_dir: Path, fps: int) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing pillow. Install with: uv sync --group viz\n" f"ImportError: {exc}"
        )

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found in PATH.")

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    steps = _normalized_steps(recording)
    font = ImageFont.load_default()

    for idx, step in enumerate(steps):
        state = step.get("state_after") or step.get("state_before") or {}
        state = state.get("state") or state
        pegs = state.get("pegs", [[], [], []])
        n_disks = state.get("n_disks") or max([d for peg in pegs for d in peg] or [1])

        img = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Step {step.get('index')}", fill="black", font=font)
        totals = step.get("totals", {})
        draw.text(
            (10, 28),
            f"moves={totals.get('moves', 0)} illegal={totals.get('illegal_moves', 0)} tool_calls={totals.get('tool_calls', 0)}",
            fill="black",
            font=font,
        )

        peg_x = [140, 320, 500]
        peg_bottom = 300
        disk_h = 16
        min_w, max_w = 40, 160

        for i, peg in enumerate(pegs):
            draw.line((peg_x[i], 80, peg_x[i], peg_bottom), fill="gray", width=3)
            for j, disk in enumerate(peg):
                ratio = (disk - 1) / (n_disks - 1) if n_disks > 1 else 1
                w = min_w + ratio * (max_w - min_w)
                x0 = peg_x[i] - w / 2
                x1 = peg_x[i] + w / 2
                y1 = peg_bottom - j * (disk_h + 4)
                y0 = y1 - disk_h
                draw.rectangle([x0, y0, x1, y1], fill="#4a90e2", outline="black")

        frame_path = frames_dir / f"frame_{idx:04d}.png"
        img.save(frame_path)

    output_path = out_dir / "playback.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Hanoi recordings.")
    parser.add_argument("--run-dir", help="Run directory containing recordings/")
    parser.add_argument("--recording", help="Path to a specific recording json")
    parser.add_argument("--out-dir", default="artifacts/renders/hanoi")
    parser.add_argument("--format", choices=["html", "ascii", "video"], default="html")
    parser.add_argument("--episode-id", type=int, action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=2)

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
        selected: list[Path] = []
        for path in recordings:
            episode_id = _episode_id_from_path(path)
            if episode_id is not None and episode_id in wanted:
                selected.append(path)
        recordings = selected
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

    for rec_path in recordings:
        recording = _load_recording(rec_path)
        episode_id = recording.get("metadata", {}).get("episode_id", rec_path.stem)
        episode_dir = out_base / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        if args.format == "ascii":
            text = _render_ascii(recording)
            (episode_dir / "playback.txt").write_text(text)
        elif args.format == "html":
            html = _render_html(recording)
            (episode_dir / "index.html").write_text(html)
        else:
            video_path = _render_video(recording, episode_dir, args.fps)
            print(f"Video written: {video_path}")

    print(f"Rendered to: {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
