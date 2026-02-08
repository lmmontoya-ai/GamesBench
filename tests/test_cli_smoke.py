from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from games_bench.bench import cli as bench_cli
from games_bench.bench.hanoi import run_batch as run_hanoi_batch
from games_bench.bench.sokoban import run_batch as run_sokoban_batch

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None


def _hanoi_args(out_dir: str, *, record: bool) -> argparse.Namespace:
    return argparse.Namespace(
        provider="cli",
        model=None,
        config=None,
        max_turns=1,
        out_dir=out_dir,
        timeout_s=1,
        provider_retries=None,
        provider_backoff=None,
        cli_cmd='python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"',
        no_stdin=False,
        codex_path="codex",
        codex_args=[],
        record_provider_raw=False,
        no_record_provider_raw=False,
        record=record,
        no_record=False,
        record_raw=False,
        no_record_raw=False,
        n_disks=["1"],
        start_peg=None,
        goal_peg=None,
        runs_per_variant=1,
        prompt_variants=["minimal"],
        prompt_file=None,
        tool_variants=["move_only"],
        allowed_tools=None,
        state_format="text",
        image_size="64x64",
        image_background="white",
        image_labels=False,
        no_image_labels=False,
    )


def _sokoban_args(out_dir: str, *, record: bool) -> argparse.Namespace:
    return argparse.Namespace(
        provider="cli",
        model=None,
        config=None,
        max_turns=1,
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
        record=record,
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


class TestCliSmoke(unittest.TestCase):
    def test_cli_run_hanoi_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["games-bench"]):
                with contextlib.redirect_stdout(stdout):
                    rc = bench_cli.main(
                        [
                            "run",
                            "hanoi",
                            "--provider",
                            "cli",
                            "--cli-cmd",
                            'python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"',
                            "--n-disks",
                            "1",
                            "--runs-per-variant",
                            "1",
                            "--max-turns",
                            "1",
                            "--prompt-variant",
                            "minimal",
                            "--tools-variant",
                            "move_only",
                            "--out-dir",
                            tmp,
                        ]
                    )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload["run_dirs"]), 1)

    def test_cli_run_sokoban_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["games-bench"]):
                with contextlib.redirect_stdout(stdout):
                    rc = bench_cli.main(
                        [
                            "run",
                            "sokoban",
                            "--provider",
                            "cli",
                            "--cli-cmd",
                            'python -c "print(\'{\\"name\\":\\"sokoban_move\\",\\"arguments\\":{\\"direction\\":\\"right\\"}}\')"',
                            "--level-id",
                            "starter-authored-v1:1",
                            "--runs-per-level",
                            "1",
                            "--max-turns",
                            "1",
                            "--prompt-variant",
                            "minimal",
                            "--tools-variant",
                            "move_only",
                            "--out-dir",
                            tmp,
                        ]
                    )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload["run_dirs"]), 1)

    def test_cli_render_smoke_for_both_games(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hanoi_run_dir = run_hanoi_batch(
                _hanoi_args(tmp, record=True), config={}, game_name="hanoi"
            )[0]
            sokoban_run_dir = run_sokoban_batch(
                _sokoban_args(tmp, record=True), config={}, game_name="sokoban"
            )[0]

            hanoi_out = Path(tmp) / "renders_hanoi"
            sokoban_out = Path(tmp) / "renders_sokoban"

            with patch.object(sys, "argv", ["games-bench"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc_hanoi = bench_cli.main(
                        [
                            "render",
                            "--game",
                            "hanoi",
                            "--run-dir",
                            str(hanoi_run_dir),
                            "--format",
                            "ascii",
                            "--max-episodes",
                            "1",
                            "--out-dir",
                            str(hanoi_out),
                        ]
                    )
                    rc_sokoban = bench_cli.main(
                        [
                            "render",
                            "--game",
                            "sokoban",
                            "--run-dir",
                            str(sokoban_run_dir),
                            "--format",
                            "ascii",
                            "--max-episodes",
                            "1",
                            "--out-dir",
                            str(sokoban_out),
                        ]
                    )
            self.assertEqual(rc_hanoi, 0)
            self.assertEqual(rc_sokoban, 0)
            self.assertTrue(list(hanoi_out.rglob("playback.txt")))
            self.assertTrue(list(sokoban_out.rglob("playback.txt")))

    @unittest.skipUnless(
        PILImage is not None, "pillow required for review image rendering"
    )
    def test_cli_review_smoke_for_both_games(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hanoi_run_dir = run_hanoi_batch(
                _hanoi_args(tmp, record=True), config={}, game_name="hanoi"
            )[0]
            sokoban_run_dir = run_sokoban_batch(
                _sokoban_args(tmp, record=True), config={}, game_name="sokoban"
            )[0]

            hanoi_out = Path(tmp) / "reviews_hanoi"
            sokoban_out = Path(tmp) / "reviews_sokoban"

            with patch.object(sys, "argv", ["games-bench"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc_hanoi = bench_cli.main(
                        [
                            "review",
                            "--game",
                            "hanoi",
                            "--run-dir",
                            str(hanoi_run_dir),
                            "--max-episodes",
                            "1",
                            "--out-dir",
                            str(hanoi_out),
                        ]
                    )
                    rc_sokoban = bench_cli.main(
                        [
                            "review",
                            "--game",
                            "sokoban",
                            "--run-dir",
                            str(sokoban_run_dir),
                            "--max-episodes",
                            "1",
                            "--image-tile-size",
                            "24",
                            "--out-dir",
                            str(sokoban_out),
                        ]
                    )
            self.assertEqual(rc_hanoi, 0)
            self.assertEqual(rc_sokoban, 0)
            self.assertTrue(list(hanoi_out.rglob("index.html")))
            self.assertTrue(list(sokoban_out.rglob("index.html")))


if __name__ == "__main__":
    unittest.main()
