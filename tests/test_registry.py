from __future__ import annotations

import unittest
from unittest.mock import patch

from games_bench.bench.registry import (
    BenchSpec,
    get_benchmark,
    list_benchmarks,
    load_builtin_benchmarks,
)
from games_bench.bench.suites import (
    SuiteSpec,
    get_suite,
    list_suites,
    load_builtin_suites,
)
from games_bench.games.registry import get_game, list_games, load_builtin_games


class TestRegistry(unittest.TestCase):
    def test_game_registry_loads_hanoi(self) -> None:
        load_builtin_games()
        self.assertIn("hanoi", list_games())
        game = get_game("hanoi")
        self.assertEqual(game.name, "hanoi")

    def test_benchmark_registry_loads_hanoi(self) -> None:
        load_builtin_benchmarks()
        self.assertIn("hanoi", list_benchmarks())
        benchmark = get_benchmark("hanoi")
        self.assertEqual(benchmark.name, "hanoi")
        self.assertIsNotNone(benchmark.add_arguments)
        self.assertIsNotNone(benchmark.default_config)
        defaults_a = benchmark.default_config()
        defaults_b = benchmark.default_config()
        self.assertEqual(defaults_a, defaults_b)
        self.assertIsNot(defaults_a, defaults_b)
        self.assertIsNotNone(benchmark.adapter_factory)
        self.assertIsNotNone(benchmark.render_main)
        self.assertIsNotNone(benchmark.review_main)

    def test_game_registry_loads_sokoban(self) -> None:
        load_builtin_games()
        self.assertIn("sokoban", list_games())
        game = get_game("sokoban")
        self.assertEqual(game.name, "sokoban")
        env = game.env_factory()
        self.assertEqual(type(env).__name__, "SokobanEnv")

    def test_benchmark_registry_loads_sokoban(self) -> None:
        load_builtin_benchmarks()
        self.assertIn("sokoban", list_benchmarks())
        benchmark = get_benchmark("sokoban")
        self.assertEqual(benchmark.name, "sokoban")
        self.assertIsNotNone(benchmark.add_arguments)
        self.assertIsNotNone(benchmark.default_config)
        defaults_a = benchmark.default_config()
        defaults_b = benchmark.default_config()
        self.assertEqual(defaults_a, defaults_b)
        self.assertIsNot(defaults_a, defaults_b)
        self.assertIsNotNone(benchmark.adapter_factory)
        self.assertIsNotNone(benchmark.render_main)
        self.assertIsNotNone(benchmark.review_main)

    def test_unknown_entries_raise(self) -> None:
        load_builtin_games()
        load_builtin_benchmarks()
        load_builtin_suites()
        with self.assertRaises(KeyError):
            get_game("__unknown_game__")
        with self.assertRaises(KeyError):
            get_benchmark("__unknown_benchmark__")
        with self.assertRaises(KeyError):
            get_suite("__unknown_suite__")

    def test_suite_registry_loads_builtin_suites(self) -> None:
        load_builtin_suites()
        self.assertIn("easy-v1", list_suites())
        self.assertIn("standard-v1", list_suites())
        for name in ("easy-v1", "standard-v1"):
            suite = get_suite(name)
            self.assertEqual(suite.name, name)
            config_a = suite.config_factory()
            config_b = suite.config_factory()
            self.assertEqual(config_a, config_b)
            self.assertIsNot(config_a, config_b)

    def test_builtin_benchmarks_load_when_registry_has_custom_entries(self) -> None:
        from games_bench.bench import registry as bench_registry_module

        custom = BenchSpec(
            name="custom",
            description="Custom benchmark",
            batch_runner=lambda _args, _cfg: [],
        )
        with patch.dict(bench_registry_module._REGISTRY, {}, clear=True):
            bench_registry_module.register_benchmark(custom)
            bench_registry_module.load_builtin_benchmarks()
            names = set(bench_registry_module.list_benchmarks())
            self.assertIn("custom", names)
            self.assertIn("hanoi", names)
            self.assertIn("sokoban", names)

    def test_builtin_suites_load_when_registry_has_custom_entries(self) -> None:
        from games_bench.bench import suites as suites_module

        custom = SuiteSpec(
            name="custom-suite",
            description="Custom suite",
            config_factory=lambda: {"spec": "custom-suite"},
        )
        with patch.dict(suites_module._REGISTRY, {}, clear=True):
            suites_module.register_suite(custom)
            suites_module.load_builtin_suites()
            names = set(suites_module.list_suites())
            self.assertIn("custom-suite", names)
            self.assertIn("easy-v1", names)
            self.assertIn("standard-v1", names)


if __name__ == "__main__":
    unittest.main()
