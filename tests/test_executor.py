from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from games_bench.bench.executor import run_episode_jobs


@dataclass(frozen=True, slots=True)
class _Job:
    episode_id: int


def _output_for(job: _Job) -> dict[str, object]:
    return {
        "episode_id": job.episode_id,
        "variant_id": "v",
        "episode": {"episode_id": job.episode_id, "variant_id": "v"},
        "events": [],
        "raw_lines": [],
        "recording": None,
    }


class TestExecutor(unittest.TestCase):
    def test_rejects_non_integer_job_episode_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            jobs = [_Job(True)]  # type: ignore[arg-type]
            with self.assertRaises(SystemExit) as ctx:
                run_episode_jobs(
                    out_dir=out_dir,
                    run_id="run",
                    jobs=jobs,
                    run_job=_output_for,
                    parallelism=1,
                    record=False,
                    record_raw=False,
                    progress_reporter=None,
                    resume=False,
                    strict_resume=False,
                    checkpoint_interval=1,
                )
            self.assertIn("missing integer 'episode_id'", str(ctx.exception))

    def test_rejects_non_integer_output_episode_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            jobs = [_Job(0)]

            def bad_output(_job: _Job) -> dict[str, object]:
                return {
                    "episode_id": 1.5,
                    "variant_id": "v",
                    "episode": {"episode_id": 1.5, "variant_id": "v"},
                    "events": [],
                    "raw_lines": [],
                    "recording": None,
                }

            with self.assertRaises(SystemExit) as ctx:
                run_episode_jobs(
                    out_dir=out_dir,
                    run_id="run",
                    jobs=jobs,
                    run_job=bad_output,
                    parallelism=1,
                    record=False,
                    record_raw=False,
                    progress_reporter=None,
                    resume=False,
                    strict_resume=False,
                    checkpoint_interval=1,
                )
            self.assertIn("missing integer 'episode_id'", str(ctx.exception))

    def test_parallel_non_contiguous_episode_ids_are_committed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            jobs = [_Job(10), _Job(12)]
            episodes = run_episode_jobs(
                out_dir=out_dir,
                run_id="run",
                jobs=jobs,
                run_job=_output_for,
                parallelism=2,
                record=False,
                record_raw=False,
                progress_reporter=None,
                resume=False,
                strict_resume=False,
                checkpoint_interval=1,
            )

            self.assertEqual([ep["episode_id"] for ep in episodes], [10, 12])
            lines = (out_dir / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                [json.loads(line)["episode_id"] for line in lines], [10, 12]
            )

    def test_duplicate_job_episode_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            jobs = [_Job(0), _Job(0)]
            with self.assertRaises(SystemExit) as ctx:
                run_episode_jobs(
                    out_dir=out_dir,
                    run_id="run",
                    jobs=jobs,
                    run_job=_output_for,
                    parallelism=2,
                    record=False,
                    record_raw=False,
                    progress_reporter=None,
                    resume=False,
                    strict_resume=False,
                    checkpoint_interval=1,
                )
            self.assertIn(
                "Duplicate episode_id detected in job plan", str(ctx.exception)
            )

    def test_parallel_output_episode_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            jobs = [_Job(0), _Job(1)]

            def bad_output(_job: _Job) -> dict[str, object]:
                return {
                    "episode_id": 0,
                    "variant_id": "v",
                    "episode": {"episode_id": 0, "variant_id": "v"},
                    "events": [],
                    "raw_lines": [],
                    "recording": None,
                }

            with self.assertRaises(SystemExit) as ctx:
                run_episode_jobs(
                    out_dir=out_dir,
                    run_id="run",
                    jobs=jobs,
                    run_job=bad_output,
                    parallelism=2,
                    record=False,
                    record_raw=False,
                    progress_reporter=None,
                    resume=False,
                    strict_resume=False,
                    checkpoint_interval=1,
                )
            self.assertIn(
                "episode_id mismatch for parallel execution", str(ctx.exception)
            )

    def test_checkpoint_save_observes_flushed_jsonl_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            jobs = [_Job(0), _Job(1)]

            from games_bench.bench import executor as executor_module

            real_save_execution_state = executor_module.save_execution_state

            def validating_save(path: Path, state: dict[str, object]) -> None:
                completed_ids = state.get("completed_episode_ids", [])
                completed_count = (
                    len(completed_ids) if isinstance(completed_ids, list) else 0
                )
                episodes_path = out_dir / "episodes.jsonl"
                traces_path = out_dir / "traces.jsonl"
                if episodes_path.exists():
                    episode_lines = len(episodes_path.read_text().splitlines())
                else:
                    episode_lines = 0
                if traces_path.exists():
                    trace_lines = len(traces_path.read_text().splitlines())
                else:
                    trace_lines = 0
                if completed_count > episode_lines or completed_count > trace_lines:
                    raise AssertionError("Checkpoint advanced past durable JSONL rows.")
                real_save_execution_state(path, state)

            with mock.patch.object(
                executor_module,
                "save_execution_state",
                side_effect=validating_save,
            ):
                episodes = run_episode_jobs(
                    out_dir=out_dir,
                    run_id="run",
                    jobs=jobs,
                    run_job=_output_for,
                    parallelism=1,
                    record=False,
                    record_raw=False,
                    progress_reporter=None,
                    resume=False,
                    strict_resume=False,
                    checkpoint_interval=1,
                )

            self.assertEqual([ep["episode_id"] for ep in episodes], [0, 1])


if __name__ == "__main__":
    unittest.main()
