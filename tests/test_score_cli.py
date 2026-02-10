from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from games_bench.bench import cli as bench_cli
from games_bench.bench import scoring
from games_bench.bench.registry import BenchSpec


class TestScoreCli(unittest.TestCase):
    def test_score_old_run_dir_without_manifest_or_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "legacy_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "legacy-run",
                        "game": "hanoi",
                        "provider": "cli",
                        "model": "legacy",
                    },
                    indent=2,
                )
            )
            (run_dir / "episodes.jsonl").write_text(
                json.dumps(
                    {
                        "episode_id": 0,
                        "variant_id": "legacy",
                        "solved": False,
                        "turn_count": 4,
                        "max_turns": 4,
                        "illegal_moves": 1,
                        "tool_calls": 1,
                    }
                )
                + "\n"
            )
            self.assertFalse((run_dir / "run_manifest.json").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = bench_cli.main(
                    [
                        "score",
                        "--run-dir",
                        str(run_dir),
                        "--write-taxonomy",
                    ]
                )
            self.assertEqual(rc, 0)

            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["score_version"], "score-v1")
            self.assertEqual(summary["taxonomy_version"], "taxonomy-v1")
            self.assertEqual(summary["overall"]["episodes"], 1)
            self.assertEqual(summary["scoring_input"]["parent_run_id"], "legacy-run")

            episode = json.loads(
                (run_dir / "episodes.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(episode["taxonomy_version"], "taxonomy-v1")
            self.assertIn("outcome_code", episode)
            self.assertIn("failure_tags", episode)

            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "legacy-run")
            self.assertEqual(manifest["parent_run_id"], "legacy-run")
            self.assertTrue(
                any(
                    event.get("event") == "rescored"
                    for event in manifest.get("lineage_events", [])
                    if isinstance(event, dict)
                )
            )

    def test_score_requires_overwrite_when_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(
                json.dumps({"run_id": "r1", "game": "hanoi"}, indent=2)
            )
            (run_dir / "episodes.jsonl").write_text("{}\n")
            (run_dir / "summary.json").write_text(
                json.dumps({"legacy": True}, indent=2)
            )

            with self.assertRaises(SystemExit) as ctx:
                scoring.score_run_dir(run_dir)
            self.assertIn("Summary already exists", str(ctx.exception))

            scoring.score_run_dir(run_dir, overwrite=True)
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["score_version"], "score-v1")

    def test_score_uses_registry_episode_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "hooked_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "hooked-run",
                        "game": "custom-game",
                    },
                    indent=2,
                )
            )
            (run_dir / "episodes.jsonl").write_text(
                json.dumps({"episode_id": 0, "variant_id": "v1", "solved": False})
                + "\n"
            )

            spec = BenchSpec(
                name="custom-game",
                description="custom",
                batch_runner=lambda _args, _cfg: [],
                episode_scorer=lambda episodes: {
                    "episodes": len(episodes),
                    "hooked_metric": 1.0,
                },
                episode_taxonomy=lambda episode, _run_config: {
                    "taxonomy_version": "taxonomy-hook",
                    "outcome_code": "hooked_outcome",
                    "failure_tags": ["hooked_tag"],
                    "episode_id": episode.get("episode_id"),
                },
            )

            with (
                patch(
                    "games_bench.bench.registry.load_builtin_benchmarks",
                    return_value=None,
                ),
                patch("games_bench.bench.registry.get_benchmark", return_value=spec),
            ):
                summary_path = scoring.score_run_dir(run_dir, write_taxonomy=True)

            self.assertEqual(summary_path, (run_dir / "summary.json").resolve())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["overall"]["hooked_metric"], 1.0)

            episode = json.loads(
                (run_dir / "episodes.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(episode["taxonomy_version"], "taxonomy-hook")
            self.assertEqual(episode["outcome_code"], "hooked_outcome")
            self.assertEqual(episode["failure_tags"], ["hooked_tag"])


if __name__ == "__main__":
    unittest.main()
