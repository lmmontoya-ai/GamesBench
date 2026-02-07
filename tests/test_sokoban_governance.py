from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from games_bench.games.sokoban.level_loader import (
    assert_level_governance_valid,
    default_levels_dir,
    validate_level_governance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_levels_bundle(
    levels_dir: Path,
    *,
    xsb_text: str,
    redistribution_allowed: bool,
    metadata_levels: dict[str, dict[str, object]],
) -> None:
    levels_dir.mkdir(parents=True, exist_ok=True)
    xsb_path = levels_dir / "bundle.xsb"
    xsb_path.write_text(xsb_text)

    manifest = {
        "version": 1,
        "sets": [
            {
                "set_name": "test-set",
                "file": "bundle.xsb",
                "source_name": "test",
                "source_url": "https://example.com/test",
                "license": "MIT",
                "license_url": "https://opensource.org/licenses/MIT",
                "redistribution_allowed": redistribution_allowed,
                "copyright_notice": "Copyright (c) 2026 Test",
                "downloaded_at": "2026-02-07",
                "sha256": _sha256(xsb_path),
            }
        ],
    }
    (levels_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    metadata = {
        "version": 1,
        "levels": metadata_levels,
    }
    (levels_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


class TestSokobanGovernance(unittest.TestCase):
    def test_repository_bundle_is_governance_valid(self) -> None:
        levels_dir = default_levels_dir()
        errors = validate_level_governance(levels_dir)
        self.assertEqual(errors, [])
        assert_level_governance_valid(levels_dir)

    def test_rejects_vendored_set_without_redistribution_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            levels_dir = Path(tmp)
            _write_levels_bundle(
                levels_dir,
                xsb_text="""#####\n#@$.#\n#####\n""",
                redistribution_allowed=False,
                metadata_levels={
                    "test-set:1": {
                        "known_optimal": True,
                        "optimal_moves": 1,
                        "optimal_pushes": 1,
                        "optimal_source": "test",
                    }
                },
            )
            errors = validate_level_governance(levels_dir)
            self.assertTrue(
                any("redistribution_allowed is false" in error for error in errors),
                msg=f"Expected redistribution error, got: {errors}",
            )

    def test_known_optimal_requires_complete_optimal_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            levels_dir = Path(tmp)
            _write_levels_bundle(
                levels_dir,
                xsb_text="""#####\n#@$.#\n#####\n""",
                redistribution_allowed=True,
                metadata_levels={
                    "test-set:1": {
                        "known_optimal": True,
                        "optimal_moves": 1,
                        "optimal_pushes": None,
                        "optimal_source": None,
                    }
                },
            )
            errors = validate_level_governance(levels_dir)
            self.assertTrue(
                any(
                    "known_optimal=true requires positive int optimal_pushes" in e
                    for e in errors
                ),
                msg=f"Expected optimal_pushes validation error, got: {errors}",
            )
            self.assertTrue(
                any(
                    "known_optimal=true requires non-empty optimal_source" in e
                    for e in errors
                ),
                msg=f"Expected optimal_source validation error, got: {errors}",
            )

    def test_metadata_must_cover_all_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            levels_dir = Path(tmp)
            _write_levels_bundle(
                levels_dir,
                xsb_text="""#####\n#@$.#\n#####\n\n#####\n#@$.#\n#####\n""",
                redistribution_allowed=True,
                metadata_levels={
                    "test-set:1": {
                        "known_optimal": False,
                        "optimal_moves": None,
                        "optimal_pushes": None,
                        "optimal_source": None,
                    }
                },
            )
            errors = validate_level_governance(levels_dir)
            self.assertTrue(
                any(
                    "metadata missing required level entry: test-set:2" in e
                    for e in errors
                ),
                msg=f"Expected missing-level metadata error, got: {errors}",
            )

    def test_rejects_undeclared_xsb_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            levels_dir = Path(tmp)
            _write_levels_bundle(
                levels_dir,
                xsb_text="""#####\n#@$.#\n#####\n""",
                redistribution_allowed=True,
                metadata_levels={
                    "test-set:1": {
                        "known_optimal": False,
                        "optimal_moves": None,
                        "optimal_pushes": None,
                        "optimal_source": None,
                    }
                },
            )
            (levels_dir / "orphan.xsb").write_text("#####\n#@$.#\n#####\n")
            errors = validate_level_governance(levels_dir)
            self.assertTrue(
                any(
                    "xsb file is present but not declared in manifest.json: orphan.xsb"
                    in e
                    for e in errors
                ),
                msg=f"Expected undeclared-file validation error, got: {errors}",
            )


if __name__ == "__main__":
    unittest.main()
