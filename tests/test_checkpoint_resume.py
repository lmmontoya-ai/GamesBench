from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from games_bench.bench import hanoi as hanoi_bench
from games_bench.bench import sokoban as sokoban_bench


def _base_args(*, out_dir: str, run_id: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        provider="cli",
        model=None,
        config=None,
        max_turns=1,
        out_dir=out_dir,
        timeout_s=1,
        provider_retries=None,
        provider_backoff=None,
        stream_debug=None,
        cli_cmd='python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":2}}\')"',
        no_stdin=False,
        codex_path="codex",
        codex_args=[],
        record_provider_raw=False,
        no_record_provider_raw=False,
        record=False,
        no_record=False,
        record_raw=False,
        no_record_raw=False,
        run_id=run_id,
        resume=False,
        strict_resume=False,
        checkpoint_interval=1,
        no_score=False,
        score_version=None,
        cases=["3x1"],
        n_pegs=None,
        n_disks=None,
        start_peg=None,
        goal_peg=None,
        runs_per_variant=2,
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


def _sokoban_args(*, out_dir: str, run_id: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        provider="cli",
        model=None,
        config=None,
        max_turns=1,
        out_dir=out_dir,
        timeout_s=1,
        provider_retries=None,
        provider_backoff=None,
        stream_debug=None,
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
        run_id=run_id,
        resume=False,
        strict_resume=False,
        checkpoint_interval=1,
        no_score=False,
        score_version=None,
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
        runs_per_level=2,
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
        deadlock_patience=None,
    )


class TestCheckpointResume(unittest.TestCase):
    def test_resume_recovers_after_interrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, run_id="resume-hanoi")
            real_fn = hanoi_bench._run_hanoi_episode_job
            has_failed = {"value": False}

            def flaky_run(*inner_args, **inner_kwargs):  # noqa: ANN002,ANN003
                job = inner_args[0]
                if job.episode_id == 1 and not has_failed["value"]:
                    has_failed["value"] = True
                    raise RuntimeError("intentional interruption")
                return real_fn(*inner_args, **inner_kwargs)

            with patch("games_bench.bench.hanoi._run_hanoi_episode_job", flaky_run):
                with self.assertRaises(RuntimeError):
                    hanoi_bench.run_batch(args, config={}, game_name="hanoi")

            run_dir = Path(tmp) / "hanoi" / "cli" / "default" / "resume-hanoi"
            self.assertTrue(run_dir.exists())
            episodes_after_interrupt = (
                (run_dir / "episodes.jsonl").read_text().splitlines()
            )
            self.assertEqual(len(episodes_after_interrupt), 1)

            state = json.loads((run_dir / "execution_state.json").read_text())
            self.assertEqual(state["completed_episode_ids"], [0])

            args.resume = True
            run_dirs = hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertEqual(len(run_dirs), 1)

            lines = (run_dir / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            episode_ids = [json.loads(line)["episode_id"] for line in lines]
            self.assertEqual(episode_ids, [0, 1])
            self.assertTrue((run_dir / "summary.json").exists())
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest.get("parent_run_id"), "resume-hanoi")
            self.assertTrue(
                any(
                    event.get("event") == "resume"
                    for event in manifest.get("lineage_events", [])
                    if isinstance(event, dict)
                )
            )

    def test_strict_resume_rejects_job_plan_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, run_id="resume-strict")
            args.runs_per_variant = 1
            run_dirs = hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertEqual(len(run_dirs), 1)

            args.resume = True
            args.strict_resume = True
            args.cases = ["3x2"]
            with self.assertRaises(SystemExit) as ctx:
                hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertIn("job plan mismatch", str(ctx.exception).lower())

    def test_strict_resume_rejects_checkpoint_ahead_of_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, run_id="resume-checkpoint-ahead-strict")
            run_dirs = hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertEqual(len(run_dirs), 1)

            run_dir = Path(tmp) / "hanoi" / "cli" / "default" / args.run_id
            episodes_path = run_dir / "episodes.jsonl"
            lines = episodes_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            episodes_path.write_text(lines[0] + "\n")

            args.resume = True
            args.strict_resume = True
            with self.assertRaises(SystemExit) as ctx:
                hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertIn("missing from episodes.jsonl", str(ctx.exception))

    def test_non_strict_resume_rewinds_checkpoint_ahead_of_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, run_id="resume-checkpoint-ahead")
            run_dirs = hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertEqual(len(run_dirs), 1)

            run_dir = Path(tmp) / "hanoi" / "cli" / "default" / args.run_id
            episodes_path = run_dir / "episodes.jsonl"
            traces_path = run_dir / "traces.jsonl"
            lines = episodes_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            episodes_path.write_text(lines[0] + "\n")
            trace_lines = traces_path.read_text().splitlines()
            traces_path.write_text(trace_lines[0] + "\n")

            args.resume = True
            args.strict_resume = False
            run_dirs = hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertEqual(len(run_dirs), 1)

            resumed_lines = episodes_path.read_text().splitlines()
            self.assertEqual(len(resumed_lines), 2)
            resumed_ids = [json.loads(line)["episode_id"] for line in resumed_lines]
            self.assertEqual(resumed_ids, [0, 1])

            state = json.loads((run_dir / "execution_state.json").read_text())
            self.assertEqual(state["completed_episode_ids"], [0, 1])

    def test_resume_requires_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, run_id=None)
            args.resume = True
            with self.assertRaises(SystemExit) as ctx:
                hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertIn("--resume requires --run-id", str(ctx.exception))

    def test_resume_rejects_missing_hanoi_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, run_id="missing-run")
            args.resume = True
            with self.assertRaises(SystemExit) as ctx:
                hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertIn("run directory does not exist", str(ctx.exception))

    def test_resume_rejects_missing_checkpoint_without_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "resume-empty"
            run_dir = Path(tmp) / "hanoi" / "cli" / "default" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "game": "hanoi",
                        "provider": "cli",
                        "model": "default",
                    }
                )
            )

            args = _base_args(out_dir=tmp, run_id=run_id)
            args.resume = True
            with self.assertRaises(SystemExit) as ctx:
                hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertIn(
                "no existing run artifacts were found",
                str(ctx.exception).lower(),
            )

    def test_sokoban_resume_recovers_after_interrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _sokoban_args(out_dir=tmp, run_id="resume-sokoban")
            real_fn = sokoban_bench._run_sokoban_episode_job
            has_failed = {"value": False}

            def flaky_run(*inner_args, **inner_kwargs):  # noqa: ANN002,ANN003
                job = inner_args[0]
                if job.episode_id == 1 and not has_failed["value"]:
                    has_failed["value"] = True
                    raise RuntimeError("intentional interruption")
                return real_fn(*inner_args, **inner_kwargs)

            with patch("games_bench.bench.sokoban._run_sokoban_episode_job", flaky_run):
                with self.assertRaises(RuntimeError):
                    sokoban_bench.run_batch(args, config={}, game_name="sokoban")

            run_dir = Path(tmp) / "sokoban" / "cli" / "default" / "resume-sokoban"
            self.assertTrue(run_dir.exists())
            episodes_after_interrupt = (
                (run_dir / "episodes.jsonl").read_text().splitlines()
            )
            self.assertEqual(len(episodes_after_interrupt), 1)

            state = json.loads((run_dir / "execution_state.json").read_text())
            self.assertEqual(state["completed_episode_ids"], [0])

            args.resume = True
            run_dirs = sokoban_bench.run_batch(args, config={}, game_name="sokoban")
            self.assertEqual(len(run_dirs), 1)

            lines = (run_dir / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            episode_ids = [json.loads(line)["episode_id"] for line in lines]
            self.assertEqual(episode_ids, [0, 1])
            self.assertTrue((run_dir / "summary.json").exists())
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest.get("parent_run_id"), "resume-sokoban")
            self.assertTrue(
                any(
                    event.get("event") == "resume"
                    for event in manifest.get("lineage_events", [])
                    if isinstance(event, dict)
                )
            )

    def test_sokoban_resume_rejects_missing_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _sokoban_args(out_dir=tmp, run_id="missing-sokoban")
            args.resume = True
            with self.assertRaises(SystemExit) as ctx:
                sokoban_bench.run_batch(args, config={}, game_name="sokoban")
            self.assertIn("run directory does not exist", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
