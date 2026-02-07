from __future__ import annotations

import unittest

from games_bench.bench.registry import (
    get_benchmark,
    list_benchmarks,
    load_builtin_benchmarks,
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

    def test_unknown_entries_raise(self) -> None:
        load_builtin_games()
        load_builtin_benchmarks()
        with self.assertRaises(KeyError):
            get_game("__unknown_game__")
        with self.assertRaises(KeyError):
            get_benchmark("__unknown_benchmark__")


if __name__ == "__main__":
    unittest.main()
