from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from games_bench.bench.sokoban import run_batch


def _base_args(*, out_dir: str, state_format: str = "text") -> argparse.Namespace:
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
        record=False,
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
        state_format=state_format,
        image_tile_size=24,
        image_background="white",
        image_labels=False,
        no_image_labels=False,
        detect_deadlocks=None,
        terminal_on_deadlock=None,
    )


class TestSokobanBatch(unittest.TestCase):
    def test_image_state_format_preflight_for_unsupported_provider(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="image")
        with self.assertRaises(SystemExit) as ctx:
            run_batch(args, config={}, game_name="sokoban")
        self.assertIn("does not support state_format", str(ctx.exception))

    def test_run_batch_does_not_mutate_retry_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            run_batch(args, config={}, game_name="sokoban")
            self.assertIsNone(args.provider_retries)
            self.assertIsNone(args.provider_backoff)

    def test_summary_includes_denominator_aware_optimal_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            run_dirs = run_batch(args, config={}, game_name="sokoban")
            self.assertEqual(len(run_dirs), 1)

            summary_path = Path(run_dirs[0]) / "summary.json"
            summary = json.loads(summary_path.read_text())
            overall = summary["overall"]
            self.assertIn("avg_move_ratio", overall)
            self.assertIn("n_with_optimal_moves", overall)
            self.assertIn("avg_push_ratio", overall)
            self.assertIn("n_with_optimal_pushes", overall)
            self.assertGreaterEqual(overall["n_with_optimal_moves"], 1)
            self.assertGreaterEqual(overall["n_with_optimal_pushes"], 1)


if __name__ == "__main__":
    unittest.main()
