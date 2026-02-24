from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from games_bench.bench import sokoban as sokoban_bench


def _base_args(*, out_dir: str, state_format: str = "text") -> argparse.Namespace:
    return argparse.Namespace(
        provider="cli",
        model=None,
        config=None,
        max_turns=1,
        out_dir=out_dir,
        timeout_s=1,
        provider_retries=None,
        provider_backoff=None,
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
        level_sets=None,
        level_ids=["starter-authored-v1:1"],
        procgen_grid_sizes=None,
        procgen_box_counts=None,
        procgen_levels_per_combo=None,
        procgen_seed=None,
        procgen_seed_sweep=None,
        procgen_wall_density=None,
        procgen_scramble_steps=None,
        max_levels=None,
        max_optimal_moves=None,
        runs_per_level=1,
        prompt_variants=["minimal"],
        tool_variants=["move_only"],
        allowed_tools=None,
        state_format=state_format,
        image_tile_size=24,
        image_background="white",
        image_labels=False,
        no_image_labels=False,
        detect_deadlocks=None,
        terminal_on_deadlock=None,
    )


class TestSokobanBatch(unittest.TestCase):
    def test_image_state_format_preflight_for_unsupported_provider(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="image")
        with self.assertRaises(SystemExit) as ctx:
            sokoban_bench.run_batch(args, config={}, game_name="sokoban")
        self.assertIn("does not support state_format", str(ctx.exception))

    def test_run_batch_does_not_mutate_retry_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            sokoban_bench.run_batch(args, config={}, game_name="sokoban")
            self.assertIsNone(args.provider_retries)
            self.assertIsNone(args.provider_backoff)

    def test_estimate_episodes_matches_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            estimated = sokoban_bench.estimate_episodes(args, config={})
            run_dir = sokoban_bench.run_batch(args, config={}, game_name="sokoban")[0]
            actual = len((Path(run_dir) / "episodes.jsonl").read_text().splitlines())
            self.assertEqual(actual, estimated)

    def test_summary_includes_denominator_aware_optimal_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            run_dirs = sokoban_bench.run_batch(args, config={}, game_name="sokoban")
            self.assertEqual(len(run_dirs), 1)

            summary_path = Path(run_dirs[0]) / "summary.json"
            summary = json.loads(summary_path.read_text())
            overall = summary["overall"]
            self.assertIn("avg_move_ratio", overall)
            self.assertIn("n_with_optimal_moves", overall)
            self.assertIn("avg_push_ratio", overall)
            self.assertIn("n_with_optimal_pushes", overall)
            self.assertGreaterEqual(overall["n_with_optimal_moves"], 1)
            self.assertGreaterEqual(overall["n_with_optimal_pushes"], 1)

    def test_run_batch_writes_expected_artifacts_and_episode_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            run_dirs = sokoban_bench.run_batch(args, config={}, game_name="sokoban")
            self.assertEqual(len(run_dirs), 1)
            run_dir = Path(run_dirs[0])

            required_files = [
                "run_config.json",
                "run_manifest.json",
                "episodes.jsonl",
                "traces.jsonl",
                "summary.json",
            ]
            for name in required_files:
                self.assertTrue((run_dir / name).exists(), name)

            first_episode = json.loads(
                (run_dir / "episodes.jsonl").read_text().splitlines()[0]
            )
            required_fields = {
                "episode_id",
                "variant_id",
                "run_idx",
                "provider",
                "model",
                "level_id",
                "level_set",
                "prompt_variant",
                "tools_variant",
                "n_boxes",
                "grid_size",
                "solved",
                "deadlocked",
                "turn_count",
                "move_count",
                "push_count",
                "illegal_moves",
                "tool_calls",
                "boxes_on_goals",
                "boxes_on_goals_ratio",
                "optimal_moves",
                "optimal_pushes",
                "known_optimal",
                "move_ratio",
                "push_ratio",
                "usage",
                "cost",
                "outcome_code",
                "failure_tags",
                "taxonomy_version",
            }
            self.assertTrue(required_fields.issubset(first_episode.keys()))

    def test_parallel_execution_preserves_episode_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            args.runs_per_level = 2
            args.parallelism = 2
            args.max_inflight_provider = 1
            run_dir = sokoban_bench.run_batch(args, config={}, game_name="sokoban")[0]
            run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
            self.assertEqual(run_config["parallelism"], 2)
            self.assertEqual(run_config["max_inflight_provider"], 1)
            episode_ids = [
                json.loads(line)["episode_id"]
                for line in (Path(run_dir) / "episodes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(episode_ids, [0, 1])

    def test_prompt_tool_incompatibility_raises(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.prompt_variants = ["full"]
        args.tool_variants = ["move_only"]
        with self.assertRaises(SystemExit) as ctx:
            sokoban_bench.run_batch(args, config={}, game_name="sokoban")
        self.assertIn("requires tool 'sokoban_get_legal_moves'", str(ctx.exception))

    def test_allowed_tools_override_accepts_list_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            run_dir = sokoban_bench.run_batch(
                args,
                config={"allowed_tools": ["sokoban_move", "sokoban_get_state"]},
                game_name="sokoban",
            )[0]
            run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
            self.assertEqual(run_config["tool_variants"][0]["name"], "custom")
            self.assertEqual(
                run_config["tool_variants"][0]["allowed_tools"],
                ["sokoban_move", "sokoban_get_state"],
            )

    def test_move_only_variant_overrides_terminal_on_deadlock(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        seen_terminal_on_deadlock: list[bool] = []

        def fake_episode(adapter, provider, **kwargs):
            seen_terminal_on_deadlock.append(adapter.env.terminal_on_deadlock)
            return SimpleNamespace(
                solved=False,
                game_metrics={
                    "level_id": adapter.env.level.level_id,
                    "n_boxes": adapter.env.level.n_boxes,
                    "grid_size": (adapter.env.level.height, adapter.env.level.width),
                    "deadlocked": False,
                    "move_count": 0,
                    "push_count": 0,
                    "boxes_on_goals": 0,
                    "optimal_moves": adapter.env.level.optimal_moves,
                    "optimal_pushes": adapter.env.level.optimal_pushes,
                    "known_optimal": adapter.env.level.known_optimal,
                },
                move_count=0,
                illegal_moves=0,
                tool_calls=0,
                usage=None,
                cost=None,
                events=[],
            )

        with tempfile.TemporaryDirectory() as tmp:
            args.out_dir = tmp
            with patch(
                "games_bench.bench.sokoban.run_tool_calling_episode", fake_episode
            ):
                run_dirs = sokoban_bench.run_batch(
                    args,
                    config={"terminal_on_deadlock": False},
                    game_name="sokoban",
                )
            first_episode = json.loads(
                (Path(run_dirs[0]) / "episodes.jsonl").read_text().splitlines()[0]
            )
            self.assertIn("terminated_early", first_episode)
            self.assertFalse(first_episode["terminated_early"])
            self.assertIsNone(first_episode["termination_reason"])

        self.assertEqual(seen_terminal_on_deadlock, [True])

    def test_select_levels_by_level_ids_dedupes(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.level_ids = ["starter-authored-v1:2", "starter-authored-v1:2"]
        args.level_sets = ["starter-authored-v1"]
        levels = sokoban_bench._select_levels(args, config={})
        self.assertEqual(
            [level.level_id for level in levels], ["starter-authored-v1:2"]
        )

    def test_select_levels_filters_by_optimal_moves(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.level_ids = None
        args.level_sets = None
        levels = sokoban_bench._select_levels(
            args,
            config={
                "level_sets": ["starter-authored-v1"],
                "max_optimal_moves": 1,
            },
        )
        self.assertEqual(
            [level.level_id for level in levels], ["starter-authored-v1:1"]
        )

    def test_select_levels_applies_max_levels(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.level_ids = None
        args.level_sets = ["starter-authored-v1"]
        args.max_levels = 1
        levels = sokoban_bench._select_levels(args, config={})
        self.assertEqual(len(levels), 1)
        self.assertEqual(levels[0].level_id, "starter-authored-v1:1")

    def test_select_levels_raises_when_filter_removes_all(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.level_ids = None
        args.level_sets = ["starter-authored-v1"]
        with self.assertRaises(SystemExit):
            sokoban_bench._select_levels(args, config={"max_optimal_moves": 0})

    def test_select_levels_supports_procgen(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.level_ids = None
        args.level_sets = None
        args.procgen_grid_sizes = ["8x8"]
        args.procgen_box_counts = ["2"]
        args.procgen_levels_per_combo = 2
        args.procgen_seed = 9
        args.procgen_wall_density = 0.0
        args.procgen_scramble_steps = 12
        levels = sokoban_bench._select_levels(args, config={})
        self.assertEqual(len(levels), 2)
        self.assertTrue(
            all(level.level_id.startswith("procgen:8x8:b2") for level in levels)
        )

    def test_select_levels_supports_procgen_cases(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.level_ids = None
        args.level_sets = None
        levels = sokoban_bench._select_levels(
            args,
            config={
                "procgen_cases": [
                    {
                        "grid_size": "8x8",
                        "box_count": 6,
                        "scramble_steps": [140, 180],
                        "levels_per_combo": 1,
                    },
                    {
                        "grid_size": "10x10",
                        "box_count": 7,
                        "scramble_steps": "220-260",
                        "levels_per_combo": 1,
                    },
                    {
                        "grid_size": "12x12",
                        "box_count": 8,
                        "scramble_steps": "300+",
                        "levels_per_combo": 1,
                    },
                ],
                "procgen_seed": 11,
                "procgen_wall_density": 0.08,
            },
        )
        self.assertEqual(len(levels), 3)
        self.assertTrue(all(level.level_id.startswith("procgen:") for level in levels))
        self.assertTrue(any(":b6:" in level.level_id for level in levels))
        self.assertTrue(any(":b7:" in level.level_id for level in levels))
        self.assertTrue(any(":b8:" in level.level_id for level in levels))

    def test_resolve_procgen_spec_rejects_mixed_case_and_grid_modes(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.procgen_grid_sizes = ["8x8"]
        args.procgen_box_counts = ["2"]
        with self.assertRaises(SystemExit):
            sokoban_bench._resolve_procgen_spec(
                args,
                config={
                    "procgen_cases": [
                        {"grid_size": "8x8", "box_count": 6, "levels_per_combo": 1}
                    ]
                },
            )

    def test_resolve_procgen_spec_rejects_seed_and_seed_sweep_mix(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.procgen_grid_sizes = ["8x8"]
        args.procgen_box_counts = ["2"]
        args.procgen_seed = 7
        args.procgen_seed_sweep = ["7,8"]
        with self.assertRaises(SystemExit):
            sokoban_bench._resolve_procgen_spec(args, config={})

    def test_resolve_procgen_spec_supports_seed_sweep(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.procgen_grid_sizes = ["8x8"]
        args.procgen_box_counts = ["2"]
        args.procgen_levels_per_combo = 1
        args.procgen_seed = None
        args.procgen_seed_sweep = ["7,8,8"]

        spec = sokoban_bench._resolve_procgen_spec(args, config={})
        self.assertIsNotNone(spec)
        self.assertEqual(spec["seeds"], [7, 8])

    def test_select_levels_rejects_mixed_static_and_procgen_flags(self) -> None:
        args = _base_args(out_dir="artifacts/test_runs", state_format="text")
        args.procgen_grid_sizes = ["8x8"]
        args.procgen_box_counts = ["2"]
        args.level_ids = ["starter-authored-v1:1"]
        with self.assertRaises(SystemExit):
            sokoban_bench._select_levels(args, config={})

    def test_run_batch_procgen_records_procgen_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            args.level_ids = None
            args.level_sets = None
            args.procgen_grid_sizes = ["8x8"]
            args.procgen_box_counts = ["2"]
            args.procgen_levels_per_combo = 1
            args.procgen_seed = 17
            args.procgen_wall_density = 0.0
            args.procgen_scramble_steps = 12

            run_dirs = sokoban_bench.run_batch(args, config={}, game_name="sokoban")
            run_config = json.loads((Path(run_dirs[0]) / "run_config.json").read_text())
            self.assertEqual(run_config["level_source"], "procgen")
            self.assertTrue(run_config["procgen"]["enabled"])
            self.assertEqual(run_config["procgen"]["mode"], "grid_box_product")
            self.assertEqual(run_config["procgen"]["grid_sizes"], ["8x8"])
            self.assertEqual(run_config["procgen"]["seed"], 17)
            self.assertEqual(run_config["procgen"]["seed_sweep"], [17])
            self.assertEqual(len(run_config["procgen"]["cases"]), 1)
            self.assertEqual(run_config["procgen"]["cases"][0]["box_count"], 2)

    def test_compute_metrics_handles_empty_input(self) -> None:
        metrics = sokoban_bench._compute_metrics([])
        self.assertEqual(metrics["episodes"], 0)
        self.assertEqual(metrics["solve_rate"], 0.0)
        self.assertIsNone(metrics["avg_moves"])
        self.assertEqual(metrics["n_with_optimal_moves"], 0)

    def test_compute_metrics_uses_solved_episodes_for_optimal_ratios(self) -> None:
        episodes = [
            {
                "solved": True,
                "deadlocked": False,
                "move_count": 10,
                "push_count": 5,
                "illegal_moves": 1,
                "tool_calls": 7,
                "boxes_on_goals_ratio": 1.0,
                "move_ratio": 2.0,
                "push_ratio": 1.5,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "cost": 0.1,
            },
            {
                "solved": False,
                "deadlocked": True,
                "move_count": 20,
                "push_count": 9,
                "illegal_moves": 3,
                "tool_calls": 8,
                "boxes_on_goals_ratio": 0.5,
                "move_ratio": 99.0,
                "push_ratio": 99.0,
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                "cost": 0.2,
            },
            {
                "solved": True,
                "deadlocked": False,
                "move_count": 12,
                "push_count": 6,
                "illegal_moves": 2,
                "tool_calls": 4,
                "boxes_on_goals_ratio": 0.75,
                "move_ratio": 1.5,
                "push_ratio": None,
                "usage": None,
                "cost": None,
            },
        ]
        metrics = sokoban_bench._compute_metrics(episodes)
        self.assertEqual(metrics["episodes"], 3)
        self.assertEqual(metrics["solved"], 2)
        self.assertAlmostEqual(metrics["solve_rate"], 2 / 3)
        self.assertEqual(metrics["deadlocked"], 1)
        self.assertAlmostEqual(metrics["deadlock_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["avg_moves"], 14.0)
        self.assertAlmostEqual(metrics["avg_pushes"], 20.0 / 3.0)
        self.assertAlmostEqual(metrics["avg_illegal_moves"], 2.0)
        self.assertAlmostEqual(metrics["avg_tool_calls"], 19.0 / 3.0)
        self.assertAlmostEqual(metrics["avg_boxes_on_goals_ratio"], 0.75)
        self.assertAlmostEqual(metrics["avg_move_ratio"], 1.75)
        self.assertEqual(metrics["n_with_optimal_moves"], 2)
        self.assertAlmostEqual(metrics["avg_push_ratio"], 1.5)
        self.assertEqual(metrics["n_with_optimal_pushes"], 1)
        self.assertEqual(
            metrics["token_totals"],
            {"prompt_tokens": 30.0, "completion_tokens": 15.0, "total_tokens": 45.0},
        )
        self.assertEqual(
            metrics["token_avgs"],
            {"prompt_tokens": 15.0, "completion_tokens": 7.5, "total_tokens": 22.5},
        )
        self.assertAlmostEqual(metrics["cost_total"], 0.3)
        self.assertAlmostEqual(metrics["cost_avg"], 0.15)

    def test_resolve_models_accepts_scalar_provider_value(self) -> None:
        models = sokoban_bench._resolve_models(
            "cli",
            config={"models": {"cli": "custom-cli-model"}},
            fallback=None,
        )
        self.assertEqual(models, ["custom-cli-model"])

    def test_merge_config_for_game_applies_defaults_and_overrides(self) -> None:
        merged = sokoban_bench._merge_config_for_game(
            {
                "max_turns": 77,
                "games": {"sokoban": {"runs_per_level": 2}},
            },
            game_name="sokoban",
            defaults=sokoban_bench.default_sokoban_config(),
        )
        self.assertEqual(merged["max_turns"], 77)
        self.assertEqual(merged["runs_per_level"], 2)
        self.assertEqual(merged["tool_variants"], ["move_only"])

    def test_max_actions_per_turn_config_alias_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _base_args(out_dir=tmp, state_format="text")
            run_dir = sokoban_bench.run_batch(
                args,
                config={"max_actions_per_turn": 2},
                game_name="sokoban",
            )[0]
            run_config = json.loads((Path(run_dir) / "run_config.json").read_text())
            self.assertEqual(run_config["max_tool_calls_per_turn"], 2)
            self.assertTrue(run_config["parallel_tool_calls"])

    def test_action_budget_instructions_reflect_tool_call_cap(self) -> None:
        single = sokoban_bench._with_action_budget_instructions(
            "Base instructions.", max_tool_calls_per_turn=1
        )
        self.assertIn("Action budget per turn:", single)
        self.assertIn("at most 1 tool call", single)
        self.assertIn("Choose exactly one move/query", single)

        multi = sokoban_bench._with_action_budget_instructions(
            "Base instructions.", max_tool_calls_per_turn=2
        )
        self.assertIn("Action budget per turn:", multi)
        self.assertIn("up to 2 tool calls", multi)
        self.assertIn("chain actions", multi)


if __name__ == "__main__":
    unittest.main()
