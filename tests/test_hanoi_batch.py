from __future__ import annotations

import argparse
import tempfile
import unittest

from games_bench.bench import hanoi as hanoi_bench


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
            hanoi_bench.run_batch(args, config={}, game_name="hanoi")
        self.assertIn("does not support state_format", str(ctx.exception))

    def test_run_batch_does_not_mutate_retry_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                provider="cli",
                model=None,
                config=None,
                max_turns=1,
                out_dir=tmp,
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
                state_format="text",
                image_size="64x64",
                image_background="white",
                image_labels=False,
                no_image_labels=False,
            )
            hanoi_bench.run_batch(args, config={}, game_name="hanoi")
            self.assertIsNone(args.provider_retries)
            self.assertIsNone(args.provider_backoff)

    def test_resolve_models_accepts_scalar_provider_value(self) -> None:
        models = hanoi_bench._resolve_models(
            "cli",
            config={"models": {"cli": "custom-cli-model"}},
            fallback=None,
        )
        self.assertEqual(models, ["custom-cli-model"])

    def test_merge_config_for_game_applies_defaults_and_overrides(self) -> None:
        merged = hanoi_bench._merge_config_for_game(
            {"max_turns": 12, "games": {"hanoi": {"runs_per_variant": 4}}},
            game_name="hanoi",
            defaults=hanoi_bench.default_hanoi_config(),
        )
        self.assertEqual(merged["max_turns"], 12)
        self.assertEqual(merged["runs_per_variant"], 4)
        self.assertEqual(merged["tool_variants"], ["move_only"])


if __name__ == "__main__":
    unittest.main()
