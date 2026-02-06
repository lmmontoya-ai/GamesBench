from __future__ import annotations

import json
import os
import tempfile
import unittest

from games_bench.config import load_config, merge_dicts, normalize_games_config


class TestConfig(unittest.TestCase):
    def test_load_config_expands_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            os.environ["GB_TEST_MODEL"] = "openai/gpt-4.1-mini"
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"models": ["$GB_TEST_MODEL"]}, f)
            loaded = load_config(path)
            self.assertEqual(loaded["models"], ["openai/gpt-4.1-mini"])

    def test_merge_dicts_nested(self) -> None:
        base = {"a": 1, "nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 3, "z": 4}}
        merged = merge_dicts(base, override)
        self.assertEqual(merged, {"a": 1, "nested": {"x": 1, "y": 3, "z": 4}})

    def test_normalize_games_config_object(self) -> None:
        config = {"models": ["m1"], "games": {"hanoi": {"n_disks": [3]}}}
        global_defaults, games_map = normalize_games_config(config)
        self.assertEqual(global_defaults, {"models": ["m1"]})
        self.assertEqual(games_map, {"hanoi": {"n_disks": [3]}})

    def test_normalize_games_config_list(self) -> None:
        config = {
            "models": ["m1"],
            "games": [
                {"name": "hanoi", "config": {"n_disks": [3]}},
                "other",
            ],
        }
        global_defaults, games_map = normalize_games_config(config)
        self.assertEqual(global_defaults, {"models": ["m1"]})
        self.assertEqual(games_map["hanoi"], {"n_disks": [3]})
        self.assertEqual(games_map["other"], {})


if __name__ == "__main__":
    unittest.main()
