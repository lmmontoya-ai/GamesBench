from __future__ import annotations

import argparse
import json
import threading
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from games_bench.bench import hanoi as hanoi_bench
from games_bench.llm.providers import ProviderResult, ToolCall


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
            cases=None,
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
                cases=None,
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

    def test_estimate_episodes_matches_run_artifacts(self) -> None:
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
            estimated = hanoi_bench.estimate_episodes(args, config={})
            run_dir = hanoi_bench.run_batch(args, config={}, game_name="hanoi")[0]
            actual = len((Path(run_dir) / "episodes.jsonl").read_text().splitlines())
            self.assertEqual(actual, estimated)

    def test_unknown_prompt_variant_raises(self) -> None:
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
            cases=["3x1"],
            n_pegs=None,
            n_disks=None,
            start_peg=None,
            goal_peg=None,
            runs_per_variant=1,
            prompt_variants=["not-a-variant"],
            prompt_file=None,
            tool_variants=["move_only"],
            allowed_tools=None,
            state_format="text",
            image_size="64x64",
            image_background="white",
            image_labels=False,
            no_image_labels=False,
        )
        with self.assertRaises(SystemExit) as ctx:
            hanoi_bench.run_batch(args, config={}, game_name="hanoi")
        self.assertIn("Unknown Hanoi prompt variant(s)", str(ctx.exception))

    def test_unknown_tool_variant_raises(self) -> None:
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
            cases=["3x1"],
            n_pegs=None,
            n_disks=None,
            start_peg=None,
            goal_peg=None,
            runs_per_variant=1,
            prompt_variants=["minimal"],
            prompt_file=None,
            tool_variants=["not-a-variant"],
            allowed_tools=None,
            state_format="text",
            image_size="64x64",
            image_background="white",
            image_labels=False,
            no_image_labels=False,
        )
        with self.assertRaises(SystemExit) as ctx:
            hanoi_bench.run_batch(args, config={}, game_name="hanoi")
        self.assertIn("Unknown Hanoi tool variant(s)", str(ctx.exception))

    def test_allowed_tools_override_accepts_list_from_config(self) -> None:
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
                cases=["3x1"],
                n_pegs=None,
                n_disks=None,
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
            run_dir = hanoi_bench.run_batch(
                args,
                config={"allowed_tools": ["hanoi_move", "hanoi_get_state"]},
                game_name="hanoi",
            )[0]
            run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
            self.assertEqual(run_config["tool_variants"][0]["name"], "custom")
            self.assertEqual(
                run_config["tool_variants"][0]["allowed_tools"],
                ["hanoi_move", "hanoi_get_state"],
            )

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

    def test_run_batch_supports_n_pegs_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                provider="cli",
                model=None,
                config=None,
                max_turns=2,
                out_dir=tmp,
                timeout_s=1,
                provider_retries=None,
                provider_backoff=None,
                cli_cmd='python -c "print(\'{\\"name\\":\\"hanoi_move\\",\\"arguments\\":{\\"from_peg\\":0,\\"to_peg\\":3}}\')"',
                no_stdin=False,
                codex_path="codex",
                codex_args=[],
                record_provider_raw=False,
                no_record_provider_raw=False,
                record=False,
                no_record=False,
                record_raw=False,
                no_record_raw=False,
                cases=None,
                n_pegs=["4"],
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
            run_dir = hanoi_bench.run_batch(args, config={}, game_name="hanoi")[0]
            run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
            self.assertEqual(run_config["n_pegs"], [4])
            self.assertEqual(run_config["start_goal_by_n_pegs"]["4"]["goal_peg"], 3)
            episode = json.loads(
                (Path(run_dir) / "episodes.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(episode["n_pegs"], 4)
            self.assertIn("p4_n1", episode["variant_id"])
            self.assertIn("turn_count", episode)
            self.assertIn("outcome_code", episode)
            self.assertIn("failure_tags", episode)
            self.assertEqual(episode["taxonomy_version"], "taxonomy-v1")

    def test_run_batch_supports_exact_cases(self) -> None:
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
                cases=None,
                n_pegs=None,
                n_disks=None,
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
            run_dir = hanoi_bench.run_batch(
                args,
                config={
                    "cases": [
                        {"n_pegs": 3, "n_disks": 1},
                        {"n_pegs": 4, "n_disks": 2},
                    ],
                    "runs_per_variant": 1,
                    "max_turns": 1,
                    "prompt_variants": ["minimal"],
                    "tool_variants": ["move_only"],
                },
                game_name="hanoi",
            )[0]
            run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
            self.assertEqual(len(run_config["cases"]), 2)
            self.assertEqual(run_config["cases"][0]["n_pegs"], 3)
            self.assertEqual(run_config["cases"][1]["n_pegs"], 4)
            episode_lines = (Path(run_dir) / "episodes.jsonl").read_text().splitlines()
            self.assertEqual(len(episode_lines), 2)
            variant_ids = {json.loads(line)["variant_id"] for line in episode_lines}
            self.assertEqual(
                variant_ids,
                {
                    "p3_n1__prompt=minimal__tools=move_only",
                    "p4_n2__prompt=minimal__tools=move_only",
                },
            )

    def test_case_flag_rejects_n_pegs_mix(self) -> None:
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
            cases=["3x1"],
            n_pegs=["3"],
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
        with self.assertRaises(SystemExit) as ctx:
            hanoi_bench.run_batch(args, config={}, game_name="hanoi")
        self.assertIn("Do not mix --case with --n-pegs/--n-disks", str(ctx.exception))

    def test_optimal_turn_cap_multiplier_limits_effective_max_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                provider="cli",
                model=None,
                config=None,
                max_turns=100,
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
                cases=["3x3"],
                n_pegs=None,
                n_disks=None,
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
                optimal_turn_cap_multiplier=2.0,
                parallelism=1,
                max_inflight_provider=None,
                stagnation_patience=None,
            )

            seen_max_turns: list[int] = []

            def fake_episode(*_args, **kwargs):
                seen_max_turns.append(int(kwargs["max_turns"]))
                return SimpleNamespace(
                    solved=False,
                    game_metrics={
                        "n_pegs": 3,
                        "n_disks": 3,
                        "move_count": 0,
                        "optimal_steps": 7,
                    },
                    move_count=0,
                    optimal_steps=7,
                    illegal_moves=0,
                    tool_calls=0,
                    usage=None,
                    cost=None,
                    events=[],
                    terminated_early=False,
                    termination_reason=None,
                )

            with patch(
                "games_bench.bench.hanoi.run_tool_calling_episode", fake_episode
            ):
                run_dir = hanoi_bench.run_batch(
                    args,
                    config={"optimal_turn_cap_multiplier": 2.0},
                    game_name="hanoi",
                )[0]

            self.assertEqual(seen_max_turns, [14])
            episode = json.loads(
                (Path(run_dir) / "episodes.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(episode["max_turns_effective"], 14)

    def test_parallel_provider_throttle_caps_inflight_calls(self) -> None:
        class FakeProvider:
            supports_images = False

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._inflight = 0
                self.max_seen = 0

            def next_tool_calls(self, **_kwargs):
                with self._lock:
                    self._inflight += 1
                    self.max_seen = max(self.max_seen, self._inflight)
                time.sleep(0.02)
                with self._lock:
                    self._inflight -= 1
                return ProviderResult(
                    tool_calls=[ToolCall("hanoi_move", {"from_peg": 0, "to_peg": 2})],
                    raw={},
                )

        fake_provider = FakeProvider()

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                provider="openrouter",
                model="fake/model",
                config=None,
                max_turns=1,
                out_dir=tmp,
                timeout_s=1,
                provider_retries=None,
                provider_backoff=None,
                cli_cmd=None,
                no_stdin=False,
                codex_path="codex",
                codex_args=[],
                record_provider_raw=False,
                no_record_provider_raw=False,
                record=False,
                no_record=False,
                record_raw=False,
                no_record_raw=False,
                cases=[
                    {"n_pegs": 3, "n_disks": 1},
                    {"n_pegs": 3, "n_disks": 2},
                    {"n_pegs": 3, "n_disks": 3},
                    {"n_pegs": 3, "n_disks": 4},
                ],
                n_pegs=None,
                n_disks=None,
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
                optimal_turn_cap_multiplier=None,
                parallelism=4,
                max_inflight_provider=2,
                stagnation_patience=None,
            )
            with patch(
                "games_bench.bench.hanoi._build_provider",
                side_effect=lambda *_a, **_k: fake_provider,
            ):
                hanoi_bench.run_batch(args, config={}, game_name="hanoi")

        self.assertLessEqual(fake_provider.max_seen, 2)


if __name__ == "__main__":
    unittest.main()
