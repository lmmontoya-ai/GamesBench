from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from games_bench.bench.checkpoint import recover_jsonl_records, recover_text_log


class TestCheckpointRecovery(unittest.TestCase):
    def test_recover_jsonl_records_truncates_partial_last_line_non_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            first = json.dumps({"episode_id": 0, "variant_id": "v1"})
            path.write_text(first + "\n" + '{"episode_id": 1')

            rows = recover_jsonl_records(path, strict=False)
            self.assertEqual(rows, [{"episode_id": 0, "variant_id": "v1"}])
            self.assertEqual(path.read_text(), first + "\n")

    def test_recover_jsonl_records_strict_raises_on_partial_last_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text('{"episode_id": 0}\n{"episode_id": 1')

            with self.assertRaises(SystemExit) as ctx:
                recover_jsonl_records(path, strict=True)
            self.assertIn("Invalid JSONL while resuming", str(ctx.exception))

    def test_recover_jsonl_records_non_strict_raises_on_mid_file_corruption(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text('{"episode_id": 0}\n{"episode_id": 1\n{"episode_id": 2}\n')

            with self.assertRaises(SystemExit) as ctx:
                recover_jsonl_records(path, strict=False)
            self.assertIn("Invalid JSONL while resuming", str(ctx.exception))

    def test_recover_text_log_truncates_to_last_newline_non_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.log"
            path.write_bytes(b"line-1\nline-2\npartial")

            recover_text_log(path, strict=False)
            self.assertEqual(path.read_bytes(), b"line-1\nline-2\n")

    def test_recover_text_log_strict_raises_without_terminal_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.log"
            path.write_bytes(b"line-1\nline-2\npartial")

            with self.assertRaises(SystemExit) as ctx:
                recover_text_log(path, strict=True)
            self.assertIn("Non-newline-terminated log", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
