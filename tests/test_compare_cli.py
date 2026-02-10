from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from games_bench.bench import compare
from games_bench.bench.registry import BenchSpec


def _write_run(
    run_dir: Path,
    *,
    game: str,
    spec: str,
    interaction_mode: str,
    provider: str,
    model: str,
    overall: dict[str, float | int | None],
    episodes: list[dict[str, object]] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_id": run_dir.name,
        "game": game,
        "spec": spec,
        "interaction_mode": interaction_mode,
        "provider": provider,
        "model": model,
        "stateless": interaction_mode == "stateless",
    }
    summary = {
        "spec": spec,
        "interaction_mode": interaction_mode,
        "overall": overall,
        "variants": {},
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if episodes is not None:
        lines = [json.dumps(dict(row)) for row in episodes]
        (run_dir / "episodes.jsonl").write_text("\n".join(lines) + "\n")


def _run_compare(argv: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return compare.main(argv)


class TestCompareCli(unittest.TestCase):
    def test_compare_single_pair_passes_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline_run"
            candidate = tmp_path / "candidate_run"
            report_path = tmp_path / "report.json"
            thresholds_path = tmp_path / "thresholds.json"

            _write_run(
                baseline,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={
                    "solve_rate": 0.50,
                    "avg_illegal_moves": 2.0,
                },
            )
            _write_run(
                candidate,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={
                    "solve_rate": 0.52,
                    "avg_illegal_moves": 1.9,
                },
            )
            thresholds_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "solve_rate": {
                                "direction": "higher_better",
                                "max_drop": 0.02,
                            },
                            "avg_illegal_moves": {
                                "direction": "lower_better",
                                "max_increase": 0.20,
                            },
                        },
                        "gating": {
                            "require_same_spec": True,
                            "require_same_interaction_mode": True,
                        },
                    }
                )
            )

            rc = _run_compare(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--thresholds",
                    str(thresholds_path),
                    "--report-file",
                    str(report_path),
                    "--fail-on-regression",
                ]
            )
            self.assertEqual(rc, 0)

            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"]["pairs"], 1)
            self.assertEqual(report["summary"]["regressions"], 0)

    def test_compare_single_pair_fails_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline_run"
            candidate = tmp_path / "candidate_run"
            report_path = tmp_path / "report.json"
            thresholds_path = tmp_path / "thresholds.json"

            _write_run(
                baseline,
                game="sokoban",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={
                    "solve_rate": 0.40,
                    "avg_illegal_moves": 1.0,
                },
            )
            _write_run(
                candidate,
                game="sokoban",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={
                    "solve_rate": 0.10,
                    "avg_illegal_moves": 1.5,
                },
            )
            thresholds_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "solve_rate": {
                                "direction": "higher_better",
                                "max_drop": 0.05,
                            },
                            "avg_illegal_moves": {
                                "direction": "lower_better",
                                "max_increase": 0.10,
                            },
                        }
                    }
                )
            )

            rc = _run_compare(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--thresholds",
                    str(thresholds_path),
                    "--report-file",
                    str(report_path),
                    "--fail-on-regression",
                ]
            )
            self.assertEqual(rc, 1)

            report = json.loads(report_path.read_text())
            self.assertGreaterEqual(report["summary"]["regressions"], 1)

    def test_compare_directory_mode_matches_on_full_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_root = tmp_path / "baseline"
            candidate_root = tmp_path / "candidate"
            report_path = tmp_path / "report.json"

            _write_run(
                baseline_root / "r1",
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.8},
            )
            _write_run(
                baseline_root / "r2",
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m2",
                overall={"solve_rate": 0.6},
            )

            _write_run(
                candidate_root / "r1",
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.9},
            )
            _write_run(
                candidate_root / "r3",
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m3",
                overall={"solve_rate": 0.5},
            )

            rc = _run_compare(
                [
                    "--baseline",
                    str(baseline_root),
                    "--candidate",
                    str(candidate_root),
                    "--report-file",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)

            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"]["pairs"], 1)
            self.assertEqual(len(report["summary"]["unmatched_baseline"]), 1)
            self.assertEqual(len(report["summary"]["unmatched_candidate"]), 1)

    def test_compare_report_file_parent_directory_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline_run"
            candidate = tmp_path / "candidate_run"
            report_path = tmp_path / "nested" / "reports" / "report.json"

            _write_run(
                baseline,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.50},
            )
            _write_run(
                candidate,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.51},
            )

            rc = _run_compare(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--report-file",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(report_path.exists())

    def test_compare_rejects_non_boolean_gating_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline_run"
            candidate = tmp_path / "candidate_run"
            thresholds_path = tmp_path / "thresholds.json"

            _write_run(
                baseline,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.50},
            )
            _write_run(
                candidate,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.51},
            )

            thresholds_path.write_text(
                json.dumps(
                    {
                        "gating": {
                            "require_same_spec": "false",
                        }
                    }
                )
            )

            with self.assertRaises(SystemExit) as ctx:
                _run_compare(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--thresholds",
                        str(thresholds_path),
                    ]
                )
            self.assertIn("thresholds.gating['require_same_spec']", str(ctx.exception))

    def test_compare_rejects_duplicate_match_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_root = tmp_path / "baseline"
            candidate_root = tmp_path / "candidate"

            _write_run(
                baseline_root / "r1",
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.8},
            )
            _write_run(
                baseline_root / "r2",
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.7},
            )
            _write_run(
                candidate_root / "r1",
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.9},
            )

            with self.assertRaises(SystemExit) as ctx:
                _run_compare(
                    [
                        "--baseline",
                        str(baseline_root),
                        "--candidate",
                        str(candidate_root),
                    ]
                )
            self.assertIn("Duplicate baseline run key detected", str(ctx.exception))

    def test_compare_legacy_run_config_fields_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline"
            candidate = tmp_path / "candidate"
            report_path = tmp_path / "report.json"

            baseline.mkdir(parents=True, exist_ok=True)
            candidate.mkdir(parents=True, exist_ok=True)
            (baseline / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "baseline",
                        "game": "hanoi",
                        "provider": "openrouter",
                        "model": "m1",
                        "spec_base": "easy-v1",
                        "stateless": False,
                    },
                    indent=2,
                )
            )
            (candidate / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "candidate",
                        "game": "hanoi",
                        "provider": "openrouter",
                        "model": "m1",
                        "spec_base": "easy-v1",
                        "stateless": False,
                    },
                    indent=2,
                )
            )
            (baseline / "summary.json").write_text(
                json.dumps(
                    {
                        "spec": "easy-v1-stateful",
                        "interaction_mode": "stateful",
                        "overall": {"solve_rate": 0.5},
                        "variants": {},
                    },
                    indent=2,
                )
            )
            (candidate / "summary.json").write_text(
                json.dumps(
                    {
                        "spec": "easy-v1-stateful",
                        "interaction_mode": "stateful",
                        "overall": {"solve_rate": 0.6},
                        "variants": {},
                    },
                    indent=2,
                )
            )

            rc = _run_compare(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--report-file",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"]["pairs"], 1)
            self.assertEqual(
                report["comparisons"][0]["match_key"]["spec"], "easy-v1-stateful"
            )
            self.assertEqual(
                report["comparisons"][0]["match_key"]["interaction_mode"], "stateful"
            )

    def test_compare_tracks_missing_metric_values_for_thresholded_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline"
            candidate = tmp_path / "candidate"
            report_path = tmp_path / "report.json"
            thresholds_path = tmp_path / "thresholds.json"

            _write_run(
                baseline,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.6},
            )
            _write_run(
                candidate,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.5},
            )
            thresholds_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "solve_rate": {
                                "direction": "higher_better",
                                "max_drop": 0.20,
                            },
                            "avg_turns": {
                                "direction": "lower_better",
                                "max_increase": 0.10,
                            },
                        }
                    }
                )
            )

            rc = _run_compare(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--thresholds",
                    str(thresholds_path),
                    "--report-file",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"]["metric_checks"], 1)
            self.assertEqual(report["summary"]["missing_metric_values"], 1)
            self.assertEqual(
                report["comparisons"][0]["metrics"]["avg_turns"]["status"], "missing"
            )

    def test_compare_uses_registry_compare_metrics_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline"
            candidate = tmp_path / "candidate"
            report_path = tmp_path / "report.json"
            thresholds_path = tmp_path / "thresholds.json"

            baseline.mkdir(parents=True, exist_ok=True)
            candidate.mkdir(parents=True, exist_ok=True)
            (baseline / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "b",
                        "game": "custom-game",
                        "spec": "easy-v1-stateful",
                        "interaction_mode": "stateful",
                        "provider": "openrouter",
                        "model": "m1",
                    },
                    indent=2,
                )
            )
            (candidate / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "c",
                        "game": "custom-game",
                        "spec": "easy-v1-stateful",
                        "interaction_mode": "stateful",
                        "provider": "openrouter",
                        "model": "m1",
                    },
                    indent=2,
                )
            )
            (baseline / "summary.json").write_text(
                json.dumps({"stats": {"custom_score": 0.90}}, indent=2)
            )
            (candidate / "summary.json").write_text(
                json.dumps({"stats": {"custom_score": 0.80}}, indent=2)
            )
            thresholds_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "custom_score": {
                                "direction": "higher_better",
                                "max_drop": 0.01,
                            }
                        }
                    }
                )
            )

            spec = BenchSpec(
                name="custom-game",
                description="custom",
                batch_runner=lambda _args, _cfg: [],
                compare_metrics=lambda summary: {
                    "custom_score": float(
                        summary.get("stats", {}).get("custom_score", 0.0)
                    )
                },
            )

            with (
                patch(
                    "games_bench.bench.registry.load_builtin_benchmarks",
                    return_value=None,
                ),
                patch("games_bench.bench.registry.get_benchmark", return_value=spec),
            ):
                rc = _run_compare(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--thresholds",
                        str(thresholds_path),
                        "--report-file",
                        str(report_path),
                        "--fail-on-regression",
                    ]
                )
            self.assertEqual(rc, 1)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"]["regressions"], 1)
            self.assertEqual(
                report["comparisons"][0]["metrics"]["custom_score"]["status"],
                "regression",
            )

    def test_compare_reports_bootstrap_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline_run"
            candidate = tmp_path / "candidate_run"
            report_path = tmp_path / "report.json"

            _write_run(
                baseline,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 0.0},
                episodes=[
                    {"episode_id": i, "variant_id": "v", "solved": False}
                    for i in range(20)
                ],
            )
            _write_run(
                candidate,
                game="hanoi",
                spec="easy-v1-stateful",
                interaction_mode="stateful",
                provider="openrouter",
                model="m1",
                overall={"solve_rate": 1.0},
                episodes=[
                    {"episode_id": i, "variant_id": "v", "solved": True}
                    for i in range(20)
                ],
            )

            rc = _run_compare(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--bootstrap-samples",
                    "300",
                    "--bootstrap-metric",
                    "solve_rate",
                    "--report-file",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)

            report = json.loads(report_path.read_text())
            self.assertTrue(report["bootstrap"]["enabled"])
            self.assertEqual(report["bootstrap"]["samples"], 300)
            solve_rate_bootstrap = report["comparisons"][0]["bootstrap"]["solve_rate"]
            self.assertEqual(solve_rate_bootstrap["status"], "ok")
            self.assertTrue(solve_rate_bootstrap["significant"])
            self.assertGreater(solve_rate_bootstrap["ci_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
