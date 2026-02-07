from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LevelSetManifest:
    set_name: str
    file: str
    source_name: str
    source_url: str
    license: str
    license_url: str
    redistribution_allowed: bool
    copyright_notice: str
    downloaded_at: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LevelMetadata:
    level_id: str
    known_optimal: bool
    optimal_moves: int | None
    optimal_pushes: int | None
    optimal_source: str | None


_REQUIRED_MANIFEST_FIELDS = {
    "set_name",
    "file",
    "source_name",
    "source_url",
    "license",
    "license_url",
    "redistribution_allowed",
    "copyright_notice",
    "downloaded_at",
    "sha256",
}


def default_levels_dir() -> Path:
    return Path(__file__).resolve().parent / "levels"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_safe_relative_file(file_name: str) -> bool:
    path = Path(file_name)
    if path.is_absolute():
        return False
    return all(part not in {"", ".", ".."} for part in path.parts)


def count_levels_in_xsb(text: str) -> int:
    count = 0
    in_level = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(";"):
            continue
        if stripped == "":
            if in_level:
                count += 1
                in_level = False
            continue
        in_level = True
    if in_level:
        count += 1
    return count


def validate_level_manifest(
    levels_dir: Path,
) -> tuple[list[LevelSetManifest], list[str]]:
    errors: list[str] = []
    manifests: list[LevelSetManifest] = []
    referenced_files: set[str] = set()

    manifest_path = levels_dir / "manifest.json"
    if not manifest_path.exists():
        return manifests, [f"Missing manifest file: {manifest_path}"]

    try:
        raw = _load_json(manifest_path)
    except json.JSONDecodeError as exc:
        return manifests, [f"Invalid JSON in {manifest_path}: {exc}"]

    if not isinstance(raw, dict):
        return manifests, ["manifest.json must be a JSON object"]

    version = raw.get("version")
    if version != 1:
        errors.append(f"manifest.json version must be 1, got {version!r}")

    sets = raw.get("sets")
    if not isinstance(sets, list) or not sets:
        errors.append("manifest.json must include a non-empty 'sets' array")
        return manifests, errors

    seen_set_names: set[str] = set()
    for index, item in enumerate(sets, start=1):
        if not isinstance(item, dict):
            errors.append(f"manifest sets[{index}] must be an object")
            continue

        missing = sorted(_REQUIRED_MANIFEST_FIELDS - set(item.keys()))
        if missing:
            errors.append(
                f"manifest sets[{index}] missing fields: {', '.join(missing)}"
            )
            continue

        set_name = item["set_name"]
        if not isinstance(set_name, str) or not set_name.strip():
            errors.append(f"manifest sets[{index}].set_name must be a non-empty string")
            continue
        if set_name in seen_set_names:
            errors.append(f"duplicate set_name in manifest: {set_name}")
            continue
        seen_set_names.add(set_name)

        file_name = item["file"]
        if not isinstance(file_name, str) or not _is_safe_relative_file(file_name):
            errors.append(
                f"manifest sets[{index}].file must be a safe relative path, got {file_name!r}"
            )
            continue
        if not file_name.lower().endswith(".xsb"):
            errors.append(
                f"manifest set '{set_name}' file must end with .xsb, got {file_name!r}"
            )
            continue

        xsb_path = levels_dir / file_name
        if not xsb_path.exists():
            errors.append(
                f"manifest set '{set_name}' references missing file: {xsb_path}"
            )
            continue
        referenced_files.add(file_name)

        redistribution_allowed = item["redistribution_allowed"]
        if not isinstance(redistribution_allowed, bool):
            errors.append(
                f"manifest set '{set_name}' redistribution_allowed must be bool"
            )
            continue
        if not redistribution_allowed:
            errors.append(
                f"manifest set '{set_name}' has vendored file but redistribution_allowed is false"
            )

        for text_field in (
            "source_name",
            "source_url",
            "license",
            "license_url",
            "copyright_notice",
            "downloaded_at",
            "sha256",
        ):
            value = item[text_field]
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"manifest set '{set_name}' field '{text_field}' must be a non-empty string"
                )

        sha256 = item["sha256"]
        if isinstance(sha256, str) and sha256.strip():
            actual_sha = _sha256_file(xsb_path)
            if sha256 != actual_sha:
                errors.append(
                    f"manifest set '{set_name}' sha256 mismatch: expected {sha256}, got {actual_sha}"
                )

        manifests.append(
            LevelSetManifest(
                set_name=set_name,
                file=file_name,
                source_name=item["source_name"],
                source_url=item["source_url"],
                license=item["license"],
                license_url=item["license_url"],
                redistribution_allowed=redistribution_allowed,
                copyright_notice=item["copyright_notice"],
                downloaded_at=item["downloaded_at"],
                sha256=item["sha256"],
            )
        )

    for xsb_path in sorted(levels_dir.glob("*.xsb")):
        if xsb_path.name not in referenced_files:
            errors.append(
                f"xsb file is present but not declared in manifest.json: {xsb_path.name}"
            )

    return manifests, errors


def _expected_level_ids(
    levels_dir: Path, manifests: list[LevelSetManifest]
) -> set[str]:
    ids: set[str] = set()
    for entry in manifests:
        level_count = count_levels_in_xsb((levels_dir / entry.file).read_text())
        for level_idx in range(1, level_count + 1):
            ids.add(f"{entry.set_name}:{level_idx}")
    return ids


def validate_optimal_metadata(
    levels_dir: Path,
    manifests: list[LevelSetManifest],
) -> tuple[dict[str, LevelMetadata], list[str]]:
    errors: list[str] = []
    parsed: dict[str, LevelMetadata] = {}

    metadata_path = levels_dir / "metadata.json"
    if not metadata_path.exists():
        return parsed, [f"Missing metadata file: {metadata_path}"]

    try:
        raw = _load_json(metadata_path)
    except json.JSONDecodeError as exc:
        return parsed, [f"Invalid JSON in {metadata_path}: {exc}"]

    if not isinstance(raw, dict):
        return parsed, ["metadata.json must be a JSON object"]

    if raw.get("version") != 1:
        errors.append(f"metadata.json version must be 1, got {raw.get('version')!r}")

    levels = raw.get("levels")
    if not isinstance(levels, dict):
        errors.append("metadata.json must include a 'levels' object")
        return parsed, errors

    expected_ids = _expected_level_ids(levels_dir, manifests)
    manifest_set_names = {entry.set_name for entry in manifests}

    for level_id, payload in levels.items():
        if not isinstance(level_id, str) or ":" not in level_id:
            errors.append(
                f"metadata level id must look like '<set>:<index>', got {level_id!r}"
            )
            continue
        set_name, level_idx_str = level_id.split(":", 1)
        if set_name not in manifest_set_names:
            errors.append(
                f"metadata level id '{level_id}' uses unknown set '{set_name}'"
            )
            continue
        if not level_idx_str.isdigit() or int(level_idx_str) < 1:
            errors.append(f"metadata level id '{level_id}' has invalid index")
            continue

        if not isinstance(payload, dict):
            errors.append(f"metadata level '{level_id}' must be an object")
            continue
        if "known_optimal" not in payload or not isinstance(
            payload["known_optimal"], bool
        ):
            errors.append(
                f"metadata level '{level_id}' must include boolean 'known_optimal'"
            )
            continue

        known_optimal = payload["known_optimal"]
        optimal_moves = payload.get("optimal_moves")
        optimal_pushes = payload.get("optimal_pushes")
        optimal_source = payload.get("optimal_source")

        if known_optimal:
            if not isinstance(optimal_moves, int) or optimal_moves < 1:
                errors.append(
                    f"metadata level '{level_id}' known_optimal=true requires positive int optimal_moves"
                )
            if not isinstance(optimal_pushes, int) or optimal_pushes < 1:
                errors.append(
                    f"metadata level '{level_id}' known_optimal=true requires positive int optimal_pushes"
                )
            if not isinstance(optimal_source, str) or not optimal_source.strip():
                errors.append(
                    f"metadata level '{level_id}' known_optimal=true requires non-empty optimal_source"
                )
        else:
            if optimal_moves is not None or optimal_pushes is not None:
                errors.append(
                    f"metadata level '{level_id}' known_optimal=false must set optimal_moves and optimal_pushes to null"
                )

        parsed[level_id] = LevelMetadata(
            level_id=level_id,
            known_optimal=known_optimal,
            optimal_moves=optimal_moves if isinstance(optimal_moves, int) else None,
            optimal_pushes=optimal_pushes if isinstance(optimal_pushes, int) else None,
            optimal_source=optimal_source if isinstance(optimal_source, str) else None,
        )

    missing_ids = sorted(expected_ids - set(levels.keys()))
    for level_id in missing_ids:
        errors.append(f"metadata missing required level entry: {level_id}")

    extra_ids = sorted(set(levels.keys()) - expected_ids)
    for level_id in extra_ids:
        errors.append(f"metadata has unknown level entry: {level_id}")

    return parsed, errors


def validate_level_governance(levels_dir: Path | None = None) -> list[str]:
    resolved_dir = levels_dir or default_levels_dir()
    manifests, manifest_errors = validate_level_manifest(resolved_dir)
    if manifest_errors:
        return manifest_errors

    _metadata, metadata_errors = validate_optimal_metadata(resolved_dir, manifests)
    return metadata_errors


def assert_level_governance_valid(levels_dir: Path | None = None) -> None:
    errors = validate_level_governance(levels_dir)
    if errors:
        bullet_list = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Sokoban level governance validation failed:\n{bullet_list}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Sokoban level governance files."
    )
    parser.add_argument(
        "--levels-dir",
        default=str(default_levels_dir()),
        help="Directory containing manifest.json, metadata.json, and .xsb files.",
    )
    args = parser.parse_args(argv)

    errors = validate_level_governance(Path(args.levels_dir))
    if errors:
        print("Sokoban level governance validation failed:")
        for error in errors:
            print(f"- {error}")
        return 2

    print("Sokoban level governance validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
