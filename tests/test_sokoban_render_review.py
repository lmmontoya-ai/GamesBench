from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from games_bench.bench import render as bench_render
from games_bench.bench import review as bench_review
from games_bench.bench import sokoban as sokoban_bench
from games_bench.games.sokoban import render as sokoban_render
from games_bench.games.sokoban import review as sokoban_review

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None


def _build_sokoban_batch_args(out_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        provider="cli",
        model=None,
        config=None,
        max_turns=2,
        out_dir=out_dir,
        timeout_s=1,
        provider_retries=None,
        provider_backoff=None,
        cli_cmd='python -c "print(\'{\\"name\\":\\"sokoban_move\\",\\"arguments\\":{\\"direction\\":\\"right\\"}}\')"',
        no_stdin=False,
        codex_path="codex",
        codex_args=[],
        record_provider_raw=False,
        no_record_provider_raw=False,
        record=True,
        no_record=False,
        record_raw=False,
        no_record_raw=False,
        level_sets=None,
        level_ids=["starter-authored-v1:1"],
        max_levels=None,
        max_optimal_moves=None,
        runs_per_level=1,
        prompt_variants=["minimal"],
        tool_variants=["move_only"],
        allowed_tools=None,
        state_format="text",
        image_tile_size=24,
        image_background="white",
        image_labels=False,
        no_image_labels=False,
        detect_deadlocks=None,
        terminal_on_deadlock=None,
    )


def _build_recorded_run(tmp_dir: str) -> Path:
    args = _build_sokoban_batch_args(out_dir=tmp_dir)
    run_dirs = sokoban_bench.run_batch(args, config={}, game_name="sokoban")
    return run_dirs[0]


class TestSokobanRenderReview(unittest.TestCase):
    def test_bench_render_dispatches_to_sokoban(self) -> None:
        with patch.object(
            sys, "argv", ["games-bench render", "--game", "sokoban", "--help"]
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    bench_render.main()
        self.assertEqual(ctx.exception.code, 0)

    def test_bench_review_dispatches_to_sokoban(self) -> None:
        with patch.object(
            sys, "argv", ["games-bench review", "--game", "sokoban", "--help"]
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    bench_review.main()
        self.assertEqual(ctx.exception.code, 0)

    def test_render_generates_html_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_recorded_run(tmp)
            out_dir = Path(tmp) / "renders"

            with patch.object(
                sys,
                "argv",
                [
                    "sokoban-render",
                    "--run-dir",
                    str(run_dir),
                    "--out-dir",
                    str(out_dir),
                    "--format",
                    "html",
                    "--max-episodes",
                    "1",
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = sokoban_render.main()
            self.assertEqual(rc, 0)

            provider, model, run_id = sokoban_render._extract_run_parts(run_dir)
            bundle_dir = out_dir / provider / model / run_id
            html_files = list(bundle_dir.glob("episode_*/index.html"))
            self.assertTrue(html_files)
            self.assertIn("Sokoban Playback", html_files[0].read_text())

    @unittest.skipUnless(
        PILImage is not None, "pillow required for review image rendering"
    )
    def test_review_generates_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_recorded_run(tmp)
            out_dir = Path(tmp) / "reviews"

            with patch.object(
                sys,
                "argv",
                [
                    "sokoban-review",
                    "--run-dir",
                    str(run_dir),
                    "--out-dir",
                    str(out_dir),
                    "--max-episodes",
                    "1",
                    "--image-tile-size",
                    "24",
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = sokoban_review.main()
            self.assertEqual(rc, 0)

            provider, model, run_id = sokoban_review._extract_run_parts(run_dir)
            bundle_dir = out_dir / provider / model / run_id
            html_files = list(bundle_dir.glob("episode_*/index.html"))
            before_images = list(bundle_dir.glob("episode_*/state_before_*.png"))
            after_images = list(bundle_dir.glob("episode_*/state_after_*.png"))
            self.assertTrue(html_files)
            self.assertTrue(before_images)
            self.assertTrue(after_images)
            self.assertIn("Sokoban Review", html_files[0].read_text())


if __name__ == "__main__":
    unittest.main()
