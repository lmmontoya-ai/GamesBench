from __future__ import annotations

import unittest

from games_bench.bench.lineage import build_run_manifest


class TestLineage(unittest.TestCase):
    def test_manifest_contains_required_fields(self) -> None:
        run_config = {
            "run_id": "r1",
            "game": "hanoi",
            "provider": "openrouter",
            "model": "m1",
            "spec_base": "easy-v1",
            "spec": "easy-v1-stateful",
            "interaction_mode": "stateful",
            "stateless": False,
            "prompt_variants": ["minimal"],
            "tool_schemas": [{"name": "hanoi_move"}],
            "procgen_seed": 7,
        }

        manifest = build_run_manifest(run_config=run_config, game_config={"x": 1})
        self.assertEqual(manifest["run_manifest_version"], "v1")
        self.assertEqual(manifest["run_id"], "r1")
        self.assertEqual(manifest["game"], "hanoi")
        self.assertEqual(manifest["provider"], "openrouter")
        self.assertEqual(manifest["model"], "m1")
        self.assertIn("hashes", manifest)
        self.assertIn("run_config_hash", manifest["hashes"])
        self.assertIn("suite_hash", manifest["hashes"])
        self.assertIn("prompt_hash", manifest["hashes"])
        self.assertIn("tool_schema_hash", manifest["hashes"])
        self.assertEqual(manifest["seed_lineage"]["procgen_seed"], 7)

    def test_hashes_are_stable_for_identical_inputs(self) -> None:
        run_config = {
            "run_id": "r1",
            "game": "hanoi",
            "provider": "openrouter",
            "model": "m1",
            "spec_base": "easy-v1",
            "spec": "easy-v1-stateful",
            "interaction_mode": "stateful",
            "stateless": False,
            "prompt_variants": ["minimal"],
            "tool_schemas": [{"name": "hanoi_move"}],
        }
        game_config = {"cases": ["3x3"], "runs_per_variant": 1}

        m1 = build_run_manifest(run_config=run_config, game_config=game_config)
        m2 = build_run_manifest(run_config=run_config, game_config=game_config)
        self.assertEqual(m1["hashes"], m2["hashes"])


if __name__ == "__main__":
    unittest.main()
