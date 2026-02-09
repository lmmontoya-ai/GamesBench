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
        self.assertIn("--stream-debug", output)
        self.assertIn("--parallelism", output)
        self.assertIn("--max-inflight-provider", output)
        self.assertIn("--stagnation-patience", output)
        self.assertIn("--stateless", output)
        self.assertIn("--suite", output)
        self.assertIn("--list-suites", output)
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

    def test_list_suites_outputs_builtin_suite(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = batch.main(["--list-suites"])
        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        names = {entry["name"] for entry in payload["suites"]}
        self.assertIn("easy-v1", names)
        self.assertIn("standard-v1", names)

    def test_suite_easy_v1_applies_hanoi_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = (
                'python -c "print(\'{\\"name\\":\\"hanoi_move\\",'
                '\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"'
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = batch.main(
                    [
                        "--provider",
                        "cli",
                        "--cli-cmd",
                        cmd,
                        "--suite",
                        "easy-v1",
                        "--game",
                        "hanoi",
                        "--max-turns",
                        "1",
                        "--out-dir",
                        tmp,
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload["run_dirs"]), 1)
            run_dir = Path(payload["run_dirs"][0])
            run_config = json.loads((run_dir / "run_config.json").read_text())
            expected_cases = {
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
                (4, 4),
                (4, 5),
            }
            self.assertEqual(len(run_config["cases"]), len(expected_cases))
            actual_cases = {
                (int(case["n_pegs"]), int(case["n_disks"]))
                for case in run_config["cases"]
            }
            self.assertEqual(actual_cases, expected_cases)
            self.assertEqual(run_config["runs_per_variant"], 3)
            self.assertEqual(run_config["spec"], "easy-v1-stateful")
            self.assertEqual(run_config["interaction_mode"], "stateful")
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["spec"], "easy-v1-stateful")
            self.assertEqual(summary["interaction_mode"], "stateful")
            episodes = (run_dir / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(episodes), 18)

    def test_suite_standard_v1_applies_hanoi_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = (
                'python -c "print(\'{\\"name\\":\\"hanoi_move\\",'
                '\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"'
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = batch.main(
                    [
                        "--provider",
                        "cli",
                        "--cli-cmd",
                        cmd,
                        "--suite",
                        "standard-v1",
                        "--game",
                        "hanoi",
                        "--max-turns",
                        "1",
                        "--out-dir",
                        tmp,
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload["run_dirs"]), 1)
            run_dir = Path(payload["run_dirs"][0])
            run_config = json.loads((run_dir / "run_config.json").read_text())
            expected_cases = {
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 10),
                (3, 20),
                (4, 4),
                (4, 5),
                (4, 10),
                (4, 20),
            }
            self.assertEqual(len(run_config["cases"]), len(expected_cases))
            actual_cases = {
                (int(case["n_pegs"]), int(case["n_disks"]))
                for case in run_config["cases"]
            }
            self.assertEqual(actual_cases, expected_cases)
            self.assertEqual(run_config["runs_per_variant"], 5)
            self.assertEqual(run_config["spec"], "standard-v1-stateful")
            self.assertEqual(run_config["interaction_mode"], "stateful")
            episodes = (run_dir / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(episodes), 45)

    def test_suite_standard_v1_stateless_sets_spec_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = (
                'python -c "print(\'{\\"name\\":\\"hanoi_move\\",'
                '\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"'
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = batch.main(
                    [
                        "--provider",
                        "cli",
                        "--cli-cmd",
                        cmd,
                        "--suite",
                        "standard-v1",
                        "--stateless",
                        "--game",
                        "hanoi",
                        "--max-turns",
                        "1",
                        "--out-dir",
                        tmp,
                    ]
                )
            self.assertEqual(rc, 0)
            run_dir = Path(json.loads(stdout.getvalue())["run_dirs"][0])
            run_config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(run_config["spec"], "standard-v1-stateless")
            self.assertEqual(run_config["interaction_mode"], "stateless")
            self.assertTrue(run_config["stateless"])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["spec"], "standard-v1-stateless")
            self.assertEqual(summary["interaction_mode"], "stateless")

    def test_config_overrides_suite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "override.json"
            config_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "hanoi": {
                                "cases": [{"n_pegs": 3, "n_disks": 1}],
                                "runs_per_variant": 1,
                                "max_turns": 1,
                                "prompt_variants": ["minimal"],
                                "tool_variants": ["move_only"],
                            }
                        }
                    }
                )
            )
            cmd = (
                'python -c "print(\'{\\"name\\":\\"hanoi_move\\",'
                '\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"'
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = batch.main(
                    [
                        "--provider",
                        "cli",
                        "--cli-cmd",
                        cmd,
                        "--suite",
                        "standard-v1",
                        "--config",
                        str(config_path),
                        "--game",
                        "hanoi",
                        "--out-dir",
                        tmp,
                    ]
                )
            self.assertEqual(rc, 0)
            run_dir = Path(json.loads(stdout.getvalue())["run_dirs"][0])
            run_config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(run_config["runs_per_variant"], 1)
            self.assertEqual(len(run_config["cases"]), 1)
            self.assertEqual(run_config["cases"][0]["n_pegs"], 3)
            episodes = (run_dir / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(episodes), 1)

    def test_unknown_suite_raises(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            batch.main(
                [
                    "--provider",
                    "cli",
                    "--cli-cmd",
                    'python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"',
                    "--suite",
                    "unknown-suite",
                    "--game",
                    "hanoi",
                ]
            )
        self.assertIn("Unknown suite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
