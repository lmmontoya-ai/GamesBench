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


if __name__ == "__main__":
    unittest.main()
