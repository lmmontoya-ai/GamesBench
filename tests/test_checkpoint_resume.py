from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from games_bench.bench import hanoi as hanoi_bench


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

    def test_resume_requires_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, run_id=None)
            args.resume = True
            with self.assertRaises(SystemExit) as ctx:
                hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertIn("--resume requires --run-id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
