from __future__ import annotations

import argparse
import unittest

from games_bench.bench.hanoi import run_batch


class TestHanoiBatch(unittest.TestCase):
    def test_image_state_format_preflight_for_unsupported_provider(self) -> None:
        args = argparse.Namespace(
            provider="cli",
            model=None,
            config=None,
            max_turns=1,
            out_dir="artifacts/test_runs",
            timeout_s=1,
            provider_retries=None,
            provider_backoff=None,
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
            n_disks=["1"],
            start_peg=None,
            goal_peg=None,
            runs_per_variant=1,
            prompt_variants=["minimal"],
            prompt_file=None,
            tool_variants=["move_only"],
            allowed_tools=None,
            state_format="image",
            image_size="64x64",
            image_background="white",
            image_labels=False,
            no_image_labels=False,
        )
        with self.assertRaises(SystemExit) as ctx:
            run_batch(args, config={}, game_name="hanoi")
        self.assertIn("does not support state_format", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
