from __future__ import annotations

import base64
import io
import unittest
from unittest import mock

from games_bench.games.sokoban import (
    SokobanEnv,
    StateImage,
    parse_xsb_levels,
    render_sokoban_env_image,
    render_sokoban_image,
    render_sokoban_state_image,
)

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None


def _level_from_xsb(xsb: str):
    return parse_xsb_levels(xsb, set_name="vision")[0]


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _tile_bounds(row: int, col: int, tile_size: int) -> tuple[int, int, int, int]:
    outer_pad = max(2, tile_size // 12)
    x0 = outer_pad + col * tile_size
    y0 = outer_pad + row * tile_size
    return (x0, y0, x0 + tile_size, y0 + tile_size)


def _count_color_in_tile(
    image: "PILImage.Image",
    row: int,
    col: int,
    tile_size: int,
    color: tuple[int, int, int],
) -> int:
    x0, y0, x1, y1 = _tile_bounds(row, col, tile_size)
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if image.getpixel((x, y)) == color:
                count += 1
    return count


class TestSokobanVision(unittest.TestCase):
    def setUp(self) -> None:
        level = _level_from_xsb(
            """#######
#  .  #
#  $@ #
#     #
#######
"""
        )
        self.env = SokobanEnv(level)
        self.state = self.env.get_state()

    def test_missing_pillow_error_has_install_guidance(self) -> None:
        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("No module named PIL")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError) as ctx:
                render_sokoban_state_image(self.state)
        msg = str(ctx.exception)
        self.assertIn("games-bench[viz]", msg)
        self.assertIn("uv sync --group viz", msg)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_render_png_payload_and_dimensions(self) -> None:
        image = render_sokoban_state_image(self.state, tile_size=32, label_grid=True)
        self.assertIsInstance(image, StateImage)
        self.assertEqual(image.mime_type, "image/png")
        self.assertTrue(image.data_base64)
        self.assertTrue(image.data_url.startswith("data:image/png;base64,"))

        decoded = base64.b64decode(image.data_base64)
        with PILImage.open(io.BytesIO(decoded)) as decoded_image:
            self.assertEqual(decoded_image.size, (image.width, image.height))

        tile_size = 32
        outer_pad = max(2, tile_size // 12)
        expected_w = tile_size + self.state.width * tile_size + outer_pad * 2
        expected_h = tile_size + self.state.height * tile_size + outer_pad * 2
        self.assertEqual(image.width, expected_w)
        self.assertEqual(image.height, expected_h)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_render_core_function_directly(self) -> None:
        image = render_sokoban_image(self.state, tile_size=26, label_grid=False)
        self.assertIsInstance(image, StateImage)
        self.assertEqual(image.mime_type, "image/png")

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_render_is_deterministic(self) -> None:
        image_a = render_sokoban_state_image(
            self.state, tile_size=36, label_grid=True, background="white"
        )
        image_b = render_sokoban_state_image(
            self.state, tile_size=36, label_grid=True, background="white"
        )
        self.assertEqual(image_a.data_base64, image_b.data_base64)
        self.assertEqual(image_a.width, image_b.width)
        self.assertEqual(image_a.height, image_b.height)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_different_states_produce_different_images(self) -> None:
        start = render_sokoban_state_image(self.state, tile_size=24, label_grid=False)
        moved_env = SokobanEnv(self.env.level)
        moved_env.move("left")
        moved = render_sokoban_state_image(
            moved_env.get_state(), tile_size=24, label_grid=False
        )
        self.assertNotEqual(start.data_base64, moved.data_base64)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_label_grid_toggle_changes_render(self) -> None:
        with_labels = render_sokoban_state_image(
            self.state, tile_size=28, label_grid=True
        )
        without_labels = render_sokoban_state_image(
            self.state, tile_size=28, label_grid=False
        )
        self.assertNotEqual(with_labels.data_base64, without_labels.data_base64)
        self.assertNotEqual(with_labels.width, without_labels.width)
        self.assertNotEqual(with_labels.height, without_labels.height)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_env_wrapper_matches_state_wrapper(self) -> None:
        from_state = render_sokoban_state_image(
            self.state, tile_size=30, label_grid=False, background="#ffffff"
        )
        from_env = render_sokoban_env_image(
            self.env, tile_size=30, label_grid=False, background="#ffffff"
        )
        self.assertEqual(from_state.data_base64, from_env.data_base64)
        self.assertEqual(from_state.width, from_env.width)
        self.assertEqual(from_state.height, from_env.height)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_mapping_state_wrapper_matches_state_object(self) -> None:
        from_object = render_sokoban_state_image(
            self.state, tile_size=24, label_grid=True
        )
        from_mapping = render_sokoban_state_image(
            self.state.to_dict(), tile_size=24, label_grid=True
        )
        self.assertEqual(from_object.data_base64, from_mapping.data_base64)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_minimum_tile_size_renders_without_error(self) -> None:
        image = render_sokoban_state_image(self.state, tile_size=8, label_grid=False)
        outer_pad = max(2, 8 // 12)
        self.assertEqual(image.width, self.state.width * 8 + outer_pad * 2)
        self.assertEqual(image.height, self.state.height * 8 + outer_pad * 2)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_box_on_goal_uses_teal_color(self) -> None:
        level = _level_from_xsb(
            """#####
#*@ #
#####
"""
        )
        state = SokobanEnv(level).get_state()
        tile_size = 24
        image = render_sokoban_state_image(state, tile_size=tile_size, label_grid=False)
        decoded = base64.b64decode(image.data_base64)
        with PILImage.open(io.BytesIO(decoded)) as decoded_image:
            teal_pixels = _count_color_in_tile(
                decoded_image,
                row=1,
                col=1,
                tile_size=tile_size,
                color=_rgb("#2a9d8f"),
            )
        self.assertGreater(teal_pixels, 0)

    @unittest.skipUnless(PILImage is not None, "Pillow not installed")
    def test_player_on_goal_uses_expected_color(self) -> None:
        level = _level_from_xsb(
            """#####
# +$#
#####
"""
        )
        state = SokobanEnv(level).get_state()
        tile_size = 24
        image = render_sokoban_state_image(state, tile_size=tile_size, label_grid=False)
        decoded = base64.b64decode(image.data_base64)
        with PILImage.open(io.BytesIO(decoded)) as decoded_image:
            player_goal_pixels = _count_color_in_tile(
                decoded_image,
                row=1,
                col=2,
                tile_size=tile_size,
                color=_rgb("#5e60ce"),
            )
        self.assertGreater(player_goal_pixels, 0)

    def test_invalid_tile_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            render_sokoban_state_image(self.state, tile_size=4)

    def test_invalid_tile_size_checked_before_pillow_import(self) -> None:
        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("No module named PIL")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ValueError):
                render_sokoban_state_image(self.state, tile_size=4)

    def test_invalid_mapping_state_raises(self) -> None:
        with self.assertRaises(ValueError):
            render_sokoban_state_image({"width": 0, "height": 1}, tile_size=16)

        with self.assertRaises(ValueError):
            render_sokoban_state_image(
                {
                    "width": 2,
                    "height": 2,
                    "walls": [],
                    "boxes": [],
                    "goals": [],
                    "player": [0, 0, 0],
                    "n_boxes": 1,
                },
                tile_size=16,
            )

        with self.assertRaises(ValueError):
            render_sokoban_state_image({}, tile_size=16)

        with self.assertRaises(ValueError):
            render_sokoban_state_image(
                {
                    "width": 2,
                    "height": 2,
                    "walls": "not-a-position-list",
                    "boxes": [[0, 0]],
                    "goals": [[0, 1]],
                    "player": [0, 0],
                    "n_boxes": 1,
                },
                tile_size=16,
            )

        with self.assertRaises(ValueError):
            render_sokoban_state_image(
                {
                    "width": 2,
                    "height": 2,
                    "walls": [],
                    "boxes": [[0, 0]],
                    "player": [0, 0],
                    "n_boxes": 1,
                },
                tile_size=16,
            )


if __name__ == "__main__":
    unittest.main()
