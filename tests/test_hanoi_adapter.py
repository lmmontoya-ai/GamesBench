from __future__ import annotations

import unittest

from games_bench.games.hanoi.adapter import HanoiGameAdapter
from games_bench.games.hanoi.env import TowerOfHanoiEnv


class TestHanoiAdapterMeta(unittest.TestCase):
    def test_move_meta_fields(self) -> None:
        env = TowerOfHanoiEnv(n_disks=1)
        adapter = HanoiGameAdapter(env)
        execution = adapter.execute_tool("hanoi_move", {"from_peg": 0, "to_peg": 2})
        self.assertTrue(execution.result["ok"])
        self.assertEqual(execution.meta["action_kind"], "move")
        self.assertTrue(execution.meta["state_mutating"])
        self.assertFalse(execution.meta["illegal_action"])
        self.assertTrue(execution.meta["counts_as_move"])

    def test_illegal_move_meta_fields(self) -> None:
        env = TowerOfHanoiEnv(n_disks=1)
        adapter = HanoiGameAdapter(env)
        execution = adapter.execute_tool("hanoi_move", {"from_peg": 1, "to_peg": 2})
        self.assertFalse(execution.result["ok"])
        self.assertEqual(execution.meta["action_kind"], "move")
        self.assertFalse(execution.meta["state_mutating"])
        self.assertTrue(execution.meta["illegal_action"])
        self.assertFalse(execution.meta["counts_as_move"])

    def test_query_and_reset_meta_fields(self) -> None:
        env = TowerOfHanoiEnv(n_disks=2)
        adapter = HanoiGameAdapter(env)

        query = adapter.execute_tool("hanoi_get_state", {})
        self.assertTrue(query.result["ok"])
        self.assertEqual(query.meta["action_kind"], "query")
        self.assertFalse(query.meta["state_mutating"])
        self.assertFalse(query.meta["illegal_action"])
        self.assertFalse(query.meta["counts_as_move"])

        reset = adapter.execute_tool("hanoi_reset", {"n_disks": 3})
        self.assertTrue(reset.result["ok"])
        self.assertEqual(reset.meta["action_kind"], "query")
        self.assertTrue(reset.meta["state_mutating"])
        self.assertFalse(reset.meta["illegal_action"])
        self.assertFalse(reset.meta["counts_as_move"])

    def test_unknown_tool_meta_fields(self) -> None:
        env = TowerOfHanoiEnv(n_disks=1)
        adapter = HanoiGameAdapter(env)
        execution = adapter.execute_tool("hanoi_missing", {})
        self.assertFalse(execution.result["ok"])
        self.assertEqual(execution.meta["action_kind"], "query")
        self.assertFalse(execution.meta["state_mutating"])
        self.assertTrue(execution.meta["illegal_action"])
        self.assertFalse(execution.meta["counts_as_move"])

    def test_schemas_and_instructions_reflect_peg_count(self) -> None:
        env = TowerOfHanoiEnv(n_disks=2, n_pegs=4, goal_peg=3)
        adapter = HanoiGameAdapter(env)
        move_schema = next(
            schema
            for schema in adapter.tool_schemas()
            if schema["name"] == "hanoi_move"
        )
        self.assertEqual(
            move_schema["parameters"]["properties"]["from_peg"]["maximum"], 3
        )
        self.assertEqual(
            move_schema["parameters"]["properties"]["to_peg"]["maximum"], 3
        )
        self.assertIn("0 to 3", adapter.default_instructions())


if __name__ == "__main__":
    unittest.main()
