from __future__ import annotations

import unittest

from games_bench.bench.taxonomy import annotate_episode_with_taxonomy, classify_episode


class TestFailureTaxonomy(unittest.TestCase):
    def test_solved_episode_classification(self) -> None:
        outcome, tags = classify_episode(
            {"solved": True, "termination_reason": "solved"}
        )
        self.assertEqual(outcome, "solved")
        self.assertEqual(tags, [])

    def test_stagnation_episode_classification(self) -> None:
        outcome, tags = classify_episode(
            {
                "solved": False,
                "termination_reason": "stagnation:5",
                "turn_count": 10,
                "max_turns": 20,
            }
        )
        self.assertEqual(outcome, "failed_stagnation")
        self.assertIn("stagnation_stop", tags)
        self.assertIn("unsolved_final", tags)

    def test_query_loop_tagging(self) -> None:
        outcome, tags = classify_episode(
            {
                "solved": False,
                "tool_calls": 5,
                "move_count": 0,
                "turn_count": 5,
                "max_turns": 12,
            }
        )
        self.assertEqual(outcome, "failed_unknown")
        self.assertIn("query_loop", tags)
        self.assertIn("unsolved_final", tags)

    def test_loop_patience_stop_classification(self) -> None:
        outcome, tags = classify_episode(
            {
                "solved": False,
                "termination_reason": "loop:8",
                "turn_count": 20,
                "max_turns": 100,
            }
        )
        self.assertEqual(outcome, "failed_loop")
        self.assertIn("loop_stop", tags)
        self.assertIn("unsolved_final", tags)

    def test_sokoban_deadlock_tags(self) -> None:
        episode = annotate_episode_with_taxonomy(
            {
                "solved": False,
                "deadlocked": True,
                "termination_reason": "deadlock_terminal",
                "turn_count": 8,
                "max_turns": 20,
            },
            game_name="sokoban",
        )
        self.assertEqual(episode["outcome_code"], "failed_deadlock_terminal")
        self.assertIn("deadlock_terminal", episode["failure_tags"])
        self.assertIn("deadlocked_final", episode["failure_tags"])
        self.assertIn("taxonomy_version", episode)


if __name__ == "__main__":
    unittest.main()
