from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from games_bench.bench import compare


class TestCompareFixtures(unittest.TestCase):
    def test_compare_simple_fixture_threshold_regression(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures" / "compare" / "simple"
        baseline = fixture_root / "baseline"
        candidate = fixture_root / "candidate"
        thresholds = fixture_root / "thresholds.json"

        self.assertTrue((baseline / "run_config.json").exists())
        self.assertTrue((candidate / "run_config.json").exists())
        self.assertTrue(thresholds.exists())

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = compare.main(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--thresholds",
                        str(thresholds),
                        "--report-file",
                        str(report_path),
                        "--fail-on-regression",
                    ]
                )
            self.assertEqual(rc, 1)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"]["pairs"], 1)
            self.assertEqual(report["summary"]["regressions"], 1)
            self.assertEqual(report["summary"]["gating_violations"], 0)
            metric = report["comparisons"][0]["metrics"]["solve_rate"]
            self.assertEqual(metric["status"], "regression")


if __name__ == "__main__":
    unittest.main()
