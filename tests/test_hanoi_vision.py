from __future__ import annotations

import unittest
from unittest import mock

from games_bench.games.hanoi import StateImage as HanoiStateImage
from games_bench.games.hanoi import render_hanoi_state_image
from games_bench.games.sokoban import StateImage as SokobanStateImage

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None


class TestHanoiVision(unittest.TestCase):
    def test_state_image_type_is_shared_across_games(self) -> None:
        self.assertIs(HanoiStateImage, SokobanStateImage)

    def test_missing_pillow_error_has_install_guidance(self) -> None:
        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("No module named PIL")
            return real_import(name, *args, **kwargs)

        state = {"pegs": [[3, 2, 1], [], []], "n_disks": 3}
        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError) as ctx:
                render_hanoi_state_image(state)
        msg = str(ctx.exception)
        self.assertIn("games-bench[viz]", msg)
        self.assertIn("uv sync --group viz", msg)

    @unittest.skipUnless(PILImage is not None, "pillow required for vision rendering")
    def test_render_supports_four_pegs(self) -> None:
        image = render_hanoi_state_image(
            {"n_disks": 4, "n_pegs": 4, "pegs": [[4, 3, 2, 1], [], [], []]}
        )
        self.assertEqual(image.mime_type, "image/png")
        self.assertEqual(image.width, 640)
        self.assertEqual(image.height, 360)


if __name__ == "__main__":
    unittest.main()
