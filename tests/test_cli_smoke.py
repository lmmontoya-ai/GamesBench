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
        cli_cmd='python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":3}}\')"',
        no_stdin=False,
        codex_path="codex",
        codex_args=[],
        record_provider_raw=False,
        no_record_provider_raw=False,
        record=record,
        no_record=False,
        record_raw=False,
        no_record_raw=False,
        n_pegs=["4"],
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
        procgen_grid_sizes=None,
        procgen_box_counts=None,
        procgen_levels_per_combo=None,
        procgen_seed=None,
        procgen_wall_density=None,
        procgen_scramble_steps=None,
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
                            'python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":3}}\')"',
                            "--n-pegs",
                            "4",
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

    def test_cli_run_sokoban_procgen_smoke(self) -> None:
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
                            "--procgen-grid-size",
                            "8x8",
                            "--procgen-box-count",
                            "2",
                            "--procgen-levels-per-combo",
                            "1",
                            "--procgen-seed",
                            "5",
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
            run_config = json.loads(
                (Path(payload["run_dirs"][0]) / "run_config.json").read_text()
            )
            self.assertEqual(run_config["level_source"], "procgen")
            self.assertTrue(run_config["procgen"]["enabled"])

    def test_cli_run_config_mode_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "sokoban": {
                                "level_ids": ["starter-authored-v1:1"],
                                "runs_per_level": 1,
                                "max_turns": 1,
                                "prompt_variants": ["minimal"],
                                "tool_variants": ["move_only"],
                            }
                        }
                    }
                )
            )

            stdout = io.StringIO()
            with patch.object(sys, "argv", ["games-bench"]):
                with contextlib.redirect_stdout(stdout):
                    rc = bench_cli.main(
                        [
                            "run",
                            "--provider",
                            "cli",
                            "--cli-cmd",
                            'python -c "print(\'{\\"name\\":\\"sokoban_move\\",\\"arguments\\":{\\"direction\\":\\"right\\"}}\')"',
                            "--config",
                            str(config_path),
                            "--game",
                            "sokoban",
                            "--out-dir",
                            tmp,
                        ]
                    )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload["run_dirs"]), 1)

    def test_cli_compare_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline"
            candidate = Path(tmp) / "candidate"
            report = Path(tmp) / "compare_report.json"

            baseline.mkdir(parents=True, exist_ok=True)
            candidate.mkdir(parents=True, exist_ok=True)
            (baseline / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "b1",
                        "game": "hanoi",
                        "spec": "easy-v1-stateful",
                        "interaction_mode": "stateful",
                        "provider": "openrouter",
                        "model": "m1",
                    }
                )
            )
            (baseline / "summary.json").write_text(
                json.dumps({"overall": {"solve_rate": 0.5}, "variants": {}})
            )
            (candidate / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "c1",
                        "game": "hanoi",
                        "spec": "easy-v1-stateful",
                        "interaction_mode": "stateful",
                        "provider": "openrouter",
                        "model": "m1",
                    }
                )
            )
            (candidate / "summary.json").write_text(
                json.dumps({"overall": {"solve_rate": 0.6}, "variants": {}})
            )

            with patch.object(sys, "argv", ["games-bench"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = bench_cli.main(
                        [
                            "compare",
                            "--baseline",
                            str(baseline),
                            "--candidate",
                            str(candidate),
                            "--report-file",
                            str(report),
                        ]
                    )
            self.assertEqual(rc, 0)
            self.assertTrue(report.exists())

    def test_cli_run_missing_provider_errors(self) -> None:
        with patch.object(sys, "argv", ["games-bench"]):
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    bench_cli.main(["run", "hanoi", "--n-disks", "1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_cli_run_unknown_game_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"games": {"hanoi": {}}}))

            with patch.object(sys, "argv", ["games-bench"]):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        bench_cli.main(
                            [
                                "run",
                                "--provider",
                                "cli",
                                "--cli-cmd",
                                'python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"',
                                "--config",
                                str(config_path),
                                "--game",
                                "unknown-game",
                            ]
                        )
            self.assertEqual(ctx.exception.code, 2)

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
