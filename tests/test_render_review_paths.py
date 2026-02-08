from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from games_bench.games.hanoi import render as hanoi_render
from games_bench.games.hanoi import review as hanoi_review
from games_bench.games.sokoban import render as sokoban_render
from games_bench.games.sokoban import review as sokoban_review


class TestRenderReviewPathParsing(unittest.TestCase):
    def test_extract_run_parts_shallow_absolute_path_is_safe(self) -> None:
        path = Path("/tmp/run123")
        self.assertEqual(
            hanoi_render._extract_run_parts(path),
            ("unknown", "unknown", "run123"),
        )
        self.assertEqual(
            hanoi_review._extract_run_parts(path),
            ("unknown", "unknown", "run123"),
        )
        self.assertEqual(
            sokoban_render._extract_run_parts(path),
            ("unknown", "unknown", "run123"),
        )
        self.assertEqual(
            sokoban_review._extract_run_parts(path),
            ("unknown", "unknown", "run123"),
        )

    def test_extract_run_parts_prefers_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run123"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_config = {
                "provider": "openrouter",
                "model": "openai/gpt-4.1-mini",
                "run_id": "20260207_openrouter_model",
            }
            (run_dir / "run_config.json").write_text(json.dumps(run_config))

            expected = (
                "openrouter",
                "openai_gpt-4.1-mini",
                "20260207_openrouter_model",
            )
            self.assertEqual(hanoi_render._extract_run_parts(run_dir), expected)
            self.assertEqual(hanoi_review._extract_run_parts(run_dir), expected)
            self.assertEqual(sokoban_render._extract_run_parts(run_dir), expected)
            self.assertEqual(sokoban_review._extract_run_parts(run_dir), expected)

    def test_html_templates_escape_script_terminators(self) -> None:
        marker = "unsafe </script> marker"
        hanoi_html = hanoi_render._render_html(
            {"metadata": {"note": marker}, "steps": []}
        )
        self.assertIn("unsafe <\\/script> marker", hanoi_html)
        self.assertNotIn(marker, hanoi_html)

        hanoi_review_html = hanoi_review._html_template(
            {"metadata": {"note": marker}, "steps": []}
        )
        self.assertIn("unsafe <\\/script> marker", hanoi_review_html)
        self.assertNotIn(marker, hanoi_review_html)

        sokoban_html = sokoban_render._render_html(
            {"metadata": {"note": marker}, "steps": []}
        )
        self.assertIn("unsafe <\\/script> marker", sokoban_html)
        self.assertNotIn(marker, sokoban_html)

        sokoban_review_html = sokoban_review._html_template(
            {"metadata": {"note": marker}, "steps": []}
        )
        self.assertIn("unsafe <\\/script> marker", sokoban_review_html)
        self.assertNotIn(marker, sokoban_review_html)

    def test_episode_id_helpers_ignore_non_numeric_suffix(self) -> None:
        self.assertEqual(
            hanoi_render._episode_id_from_path(Path("episode_0007.json")), 7
        )
        self.assertEqual(
            hanoi_review._episode_id_from_path(Path("episode_0007.json")), 7
        )
        self.assertEqual(
            sokoban_render._episode_id_from_path(Path("episode_0007.json")), 7
        )
        self.assertEqual(
            sokoban_review._episode_id_from_path(Path("episode_0007.json")), 7
        )

        self.assertIsNone(hanoi_render._episode_id_from_path(Path("episode_bad.json")))
        self.assertIsNone(hanoi_review._episode_id_from_path(Path("episode_bad.json")))
        self.assertIsNone(
            sokoban_render._episode_id_from_path(Path("episode_bad.json"))
        )
        self.assertIsNone(
            sokoban_review._episode_id_from_path(Path("episode_bad.json"))
        )

    def test_hanoi_render_normalized_steps_respects_n_pegs_metadata(self) -> None:
        recording = {
            "metadata": {"n_disks": 2, "n_pegs": 4, "start_peg": 0},
            "steps": [
                {
                    "index": 0,
                    "action": {
                        "name": "hanoi_move",
                        "arguments": {"from_peg": 0, "to_peg": 1},
                    },
                    "state_after": {
                        "n_disks": 2,
                        "n_pegs": 4,
                        "pegs": [[2], [1], [], []],
                    },
                    "totals": {"moves": 1, "illegal_moves": 0, "tool_calls": 1},
                }
            ],
        }
        steps = hanoi_render._normalized_steps(recording)
        self.assertEqual(len(steps[0]["state_before"]["pegs"]), 4)


if __name__ == "__main__":
    unittest.main()
