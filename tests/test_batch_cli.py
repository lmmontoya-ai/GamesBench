from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from games_bench.bench import batch


class TestBatchCli(unittest.TestCase):
    def test_run_help_shows_only_common_flags(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                batch.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        output = stdout.getvalue()
        self.assertIn("--provider", output)
        self.assertIn("--game", output)
        self.assertNotIn("--level-set", output)
        self.assertNotIn("--n-disks", output)

    def test_run_sokoban_help_shows_game_specific_flags(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                batch.main(["sokoban", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        output = stdout.getvalue()
        self.assertIn("--level-set", output)
        self.assertIn("--terminal-on-deadlock", output)
        self.assertNotIn("--n-disks", output)

    def test_config_precedence_per_game_over_global_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config = {
                "max_turns": 5,
                "games": {
                    "sokoban": {
                        "max_turns": 1,
                        "runs_per_level": 1,
                        "level_ids": ["starter-authored-v1:1"],
                        "prompt_variants": ["minimal"],
                        "tool_variants": ["move_only"],
                    }
                },
            }
            config_path.write_text(json.dumps(config))

            cmd = (
                'python -c "print(\'{\\"name\\":\\"sokoban_move\\",'
                '\\"arguments\\":{\\"direction\\":\\"right\\"}}\')"'
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = batch.main(
                    [
                        "--provider",
                        "cli",
                        "--cli-cmd",
                        cmd,
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
            run_dir = Path(payload["run_dirs"][0])
            run_config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(run_config["max_turns"], 1)


if __name__ == "__main__":
    unittest.main()
