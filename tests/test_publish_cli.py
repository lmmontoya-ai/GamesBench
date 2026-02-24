from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from games_bench.bench import cli as bench_cli
from games_bench.bench import publish


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_events(step_count: int, *, include_reasoning: bool) -> list[dict]:
    events: list[dict] = []
    for turn in range(step_count):
        events.append(
            {
                "type": "state_snapshot",
                "turn_index": turn,
                "state": {"peg": turn},
            }
        )
        events.append(
            {
                "type": "tool_call",
                "turn_index": turn,
                "action_index": 0,
                "name": "hanoi_move",
                "arguments": {"from_peg": 0, "to_peg": 2},
            }
        )
        events.append(
            {
                "type": "tool_result",
                "turn_index": turn,
                "action_index": 0,
                "result": {"ok": True, "state": {"peg": turn + 1}},
            }
        )
        events.append(
            {
                "type": "action_state",
                "turn_index": turn,
                "action_index": 0,
                "state": {"peg": turn + 1},
            }
        )
    events.append(
        {
            "type": "provider_result",
            "turn_index": 0,
            "error": None,
            "raw": (
                {
                    "choices": [
                        {
                            "message": {
                                "content": "I should move the smallest disk first."
                            }
                        }
                    ]
                }
                if include_reasoning
                else None
            ),
        }
    )
    return events


def _create_run_fixture(root: Path) -> Path:
    run_dir = root / "artifacts" / "runs" / "hanoi" / "openrouter" / "demo" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": "run_001",
        "game": "hanoi",
        "spec": "standard-v1-stateful",
        "spec_base": "standard-v1",
        "interaction_mode": "stateful",
        "provider": "openrouter",
        "model": "acme/model-v1",
    }
    _write_json(run_dir / "run_config.json", run_config)

    summary = {
        "spec_base": "standard-v1",
        "spec": "standard-v1-stateful",
        "interaction_mode": "stateful",
        "score_version": "score-v1",
        "taxonomy_version": "taxonomy-v1",
        "scored_at": "2026-02-24T00:00:00Z",
        "overall": {
            "episodes": 4,
            "solved": 1,
            "solve_rate": 0.25,
            "avg_tool_calls": 5.0,
            "avg_illegal_moves": 1.0,
            "token_totals": {
                "prompt_tokens": 80.0,
                "completion_tokens": 20.0,
                "total_tokens": 100.0,
            },
            "cost_total": 2.0,
        },
        "variants": {
            "p3_n3": {"episodes": 4, "solved": 1, "solve_rate": 0.25},
        },
    }
    _write_json(run_dir / "summary.json", summary)

    episodes = [
        {
            "episode_id": 0,
            "variant_id": "p3_n3",
            "solved": True,
            "move_count": 8,
            "optimal_steps": 7,
            "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
            "cost": 1.0,
            "outcome_code": "solved",
            "failure_tags": [],
        },
        {
            "episode_id": 1,
            "variant_id": "p3_n3",
            "solved": False,
            "move_count": 5,
            "optimal_steps": 7,
            "usage": None,
            "cost": None,
            "outcome_code": "failed_deadlock_terminal",
            "failure_tags": ["deadlock_terminal", "unsolved_final"],
        },
        {
            "episode_id": 2,
            "variant_id": "p3_n3",
            "solved": False,
            "move_count": 6,
            "optimal_steps": 7,
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
            "cost": 1.0,
            "outcome_code": "failed_budget",
            "failure_tags": ["turn_budget_exhausted", "unsolved_final"],
        },
        {
            "episode_id": 3,
            "variant_id": "p3_n3",
            "solved": False,
            "move_count": 1,
            "optimal_steps": 7,
            "usage": None,
            "cost": None,
            "outcome_code": "failed_provider",
            "failure_tags": ["provider_error", "unsolved_final"],
        },
    ]
    _write_jsonl(run_dir / "episodes.jsonl", episodes)

    traces = [
        {
            "episode_id": 0,
            "variant_id": "p3_n3",
            "events": _build_events(1, include_reasoning=False),
        },
        {
            "episode_id": 1,
            "variant_id": "p3_n3",
            "events": _build_events(3, include_reasoning=False),
        },
        {
            "episode_id": 2,
            "variant_id": "p3_n3",
            "events": _build_events(2, include_reasoning=True),
        },
        {
            "episode_id": 3,
            "variant_id": "p3_n3",
            "events": _build_events(1, include_reasoning=False),
        },
    ]
    _write_jsonl(run_dir / "traces.jsonl", traces)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "run_001",
            "git": {"commit": "abc123"},
        },
    )

    return run_dir


class TestPublishCli(unittest.TestCase):
    def test_publish_pack_via_main_cli_writes_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _create_run_fixture(Path(tmp))
            out_root = Path(tmp) / "bench_results" / "releases"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = bench_cli.main(
                    [
                        "publish",
                        "pack",
                        "--run-dir",
                        str(run_dir),
                        "--release-id",
                        "model-v1-2026-02",
                        "--release-date",
                        "2026-02-24",
                        "--out-root",
                        str(out_root),
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(Path(payload["record_path"]).exists())

    def test_pack_computes_coverage_and_taxonomy_rollups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _create_run_fixture(Path(tmp))
            out_root = Path(tmp) / "bench_results" / "releases"

            record_path = publish.pack_run_record(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                release_date="2026-02-24",
                out_root=out_root,
            )
            record = json.loads(record_path.read_text())

            self.assertEqual(record["derived"]["episodes_total"], 4)
            self.assertEqual(record["derived"]["episodes_with_usage"], 2)
            self.assertEqual(record["derived"]["episodes_with_cost"], 2)
            self.assertAlmostEqual(record["derived"]["token_coverage_rate"], 0.5)
            self.assertAlmostEqual(record["derived"]["cost_coverage_rate"], 0.5)
            self.assertAlmostEqual(record["derived"]["tokens_per_solved"], 100.0)
            self.assertAlmostEqual(record["derived"]["cost_per_solved"], 2.0)

            self.assertEqual(record["outcome_counts"]["solved"], 1)
            self.assertEqual(record["outcome_counts"]["failed_budget"], 1)
            self.assertEqual(record["failure_tag_counts"]["unsolved_final"], 3)
            self.assertEqual(record["git_commit"], "abc123")

    def test_pack_trajectories_selects_three_canonical_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _create_run_fixture(Path(tmp))
            out_root = Path(tmp) / "bench_results" / "trajectories"

            index_path = publish.pack_trajectories(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                out_root=out_root,
                max_trajectories=3,
            )
            index_payload = json.loads(index_path.read_text())
            self.assertEqual(len(index_payload["selected"]), 3)

            slots = [row["slot"] for row in index_payload["selected"]]
            episode_ids = [row["episode_id"] for row in index_payload["selected"]]
            self.assertEqual(
                slots,
                [
                    "best_solved",
                    "highest_progress_failure",
                    "representative_hard_failure",
                ],
            )
            self.assertEqual(episode_ids, [0, 2, 1])

    def test_pack_trajectories_sets_reasoning_null_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _create_run_fixture(Path(tmp))
            out_root = Path(tmp) / "bench_results" / "trajectories"

            index_path = publish.pack_trajectories(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                out_root=out_root,
                max_trajectories=3,
            )
            index_payload = json.loads(index_path.read_text())
            best = next(
                item
                for item in index_payload["selected"]
                if item["slot"] == "best_solved"
            )
            episode_path = index_path.parent / best["path"]
            episode_payload = json.loads(episode_path.read_text())

            self.assertIsNone(episode_payload["reasoning"])
            self.assertEqual(episode_payload["reasoning_source"], "none")

    def test_pack_trajectories_extracts_reasoning_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _create_run_fixture(Path(tmp))
            out_root = Path(tmp) / "bench_results" / "trajectories"

            index_path = publish.pack_trajectories(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                out_root=out_root,
                max_trajectories=3,
            )
            index_payload = json.loads(index_path.read_text())
            progress_failure = next(
                item
                for item in index_payload["selected"]
                if item["slot"] == "highest_progress_failure"
            )
            episode_payload = json.loads(
                (index_path.parent / progress_failure["path"]).read_text()
            )

            self.assertEqual(episode_payload["reasoning_source"], "provider_raw")
            self.assertIn("smallest disk", episode_payload["reasoning"])  # from fixture

    def test_build_site_creates_canonical_and_model_game_pointer_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _create_run_fixture(root)

            release_root = root / "bench_results" / "releases"
            trajectory_root = root / "bench_results" / "trajectories"
            output_dir = root / "dist" / "site_data"

            record_path = publish.pack_run_record(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                release_date="2026-02-24",
                out_root=release_root,
            )
            record = json.loads(record_path.read_text())
            run_key = record["run_key"]

            publish.pack_trajectories(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                out_root=trajectory_root,
                max_trajectories=3,
            )

            payload = publish.build_site_data(
                input_releases=release_root,
                input_trajectories=trajectory_root,
                output_dir=output_dir,
                strict=True,
            )

            self.assertEqual(payload["run_count"], 1)
            self.assertTrue((output_dir / "index.json").exists())
            self.assertTrue((output_dir / "leaderboard.json").exists())
            self.assertTrue(
                (output_dir / "runs" / "model-v1-2026-02" / f"{run_key}.json").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "trajectories"
                    / "model-v1-2026-02"
                    / run_key
                    / "index.json"
                ).exists()
            )

            pointer_path = (
                output_dir
                / "by_model_game"
                / publish._slug("acme/model-v1")
                / "hanoi"
                / "latest.json"
            )
            pointer = json.loads(pointer_path.read_text())
            self.assertEqual(pointer["run_key"], run_key)
            self.assertEqual(len(pointer["trajectory_paths"]), 3)

    def test_pack_trajectories_is_deterministic_across_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _create_run_fixture(Path(tmp))
            out_root = Path(tmp) / "bench_results" / "trajectories"

            first = publish.pack_trajectories(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                out_root=out_root,
                max_trajectories=3,
            )
            first_selected = json.loads(first.read_text())["selected"]

            second = publish.pack_trajectories(
                run_dir=run_dir,
                release_id="model-v1-2026-02",
                out_root=out_root,
                max_trajectories=3,
            )
            second_selected = json.loads(second.read_text())["selected"]

            self.assertEqual(first_selected, second_selected)


if __name__ == "__main__":
    unittest.main()
