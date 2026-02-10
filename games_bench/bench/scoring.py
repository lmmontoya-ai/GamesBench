from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from games_bench.bench.taxonomy import TAXONOMY_VERSION, annotate_episode_with_taxonomy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"Invalid JSONL row at {path}:{lineno}: expected object")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def enrich_episodes_with_taxonomy(
    episodes: list[dict[str, Any]],
    *,
    game_name: str,
) -> list[dict[str, Any]]:
    return [annotate_episode_with_taxonomy(ep, game_name=game_name) for ep in episodes]


def build_summary_document(
    *,
    run_config: dict[str, Any],
    episodes: list[dict[str, Any]],
    score_episodes: Callable[[list[dict[str, Any]]], dict[str, Any]],
    score_version: str,
    game_name: str,
    scoring_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched_episodes = enrich_episodes_with_taxonomy(episodes, game_name=game_name)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for episode in enriched_episodes:
        variant_id = str(episode.get("variant_id", "unknown"))
        grouped.setdefault(variant_id, []).append(episode)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "spec_base": run_config.get("spec_base"),
        "spec": run_config.get("spec"),
        "interaction_mode": run_config.get("interaction_mode"),
        "stateless": bool(run_config.get("stateless", False)),
        "score_version": score_version,
        "scored_at": now,
        "taxonomy_version": TAXONOMY_VERSION,
        "scoring_input": scoring_input or {},
        "overall": score_episodes(enriched_episodes),
        "variants": {
            variant_id: score_episodes(items) for variant_id, items in grouped.items()
        },
    }


def score_run_dir(
    run_dir: Path,
    *,
    game_name: str | None = None,
    score_version: str = "score-v1",
    overwrite: bool = False,
    write_taxonomy: bool = False,
) -> Path:
    run_dir = run_dir.resolve()
    run_config_path = run_dir / "run_config.json"
    episodes_path = run_dir / "episodes.jsonl"
    summary_path = run_dir / "summary.json"

    if summary_path.exists() and not overwrite:
        raise SystemExit(
            f"Summary already exists: {summary_path}. Use --overwrite to replace it."
        )

    if not run_config_path.exists():
        raise SystemExit(f"Missing required file: {run_config_path}")

    run_config = json.loads(run_config_path.read_text())
    detected_game = game_name or run_config.get("game")
    if not detected_game:
        raise SystemExit(
            "Could not determine game. Provide --game or ensure run_config.json has 'game'."
        )
    detected_game = str(detected_game)

    episodes = read_jsonl(episodes_path)
    enriched_episodes = enrich_episodes_with_taxonomy(episodes, game_name=detected_game)

    if write_taxonomy:
        _write_jsonl(episodes_path, enriched_episodes)

    from games_bench.bench.registry import get_benchmark, load_builtin_benchmarks

    load_builtin_benchmarks()
    benchmark = get_benchmark(detected_game)
    score_fn = benchmark.score_episodes
    if score_fn is None:
        raise SystemExit(
            f"Benchmark '{detected_game}' does not expose a scoring function."
        )

    summary = build_summary_document(
        run_config=run_config,
        episodes=enriched_episodes,
        score_episodes=score_fn,
        score_version=score_version,
        game_name=detected_game,
        scoring_input={
            "source": "offline",
            "run_dir": str(run_dir),
            "run_config_file": str(run_config_path),
            "episodes_file": str(episodes_path),
            "run_config_sha256": _sha256_file(run_config_path),
            "episodes_sha256": _sha256_file(episodes_path),
            "episodes_count": len(enriched_episodes),
        },
    )

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score benchmark run artifacts from episodes.jsonl."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        required=True,
        help="Run directory to score (repeatable).",
    )
    parser.add_argument(
        "--game",
        default=None,
        help="Override game name if run_config is missing or incorrect.",
    )
    parser.add_argument(
        "--score-version",
        default="score-v1",
        help="Score version label to stamp into summary.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite summary.json if it already exists.",
    )
    parser.add_argument(
        "--write-taxonomy",
        action="store_true",
        help="Persist taxonomy fields back into episodes.jsonl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    summary_paths: list[str] = []
    for raw_path in args.run_dirs:
        run_dir = Path(raw_path)
        summary_path = score_run_dir(
            run_dir,
            game_name=args.game,
            score_version=str(args.score_version),
            overwrite=bool(args.overwrite),
            write_taxonomy=bool(args.write_taxonomy),
        )
        summary_paths.append(str(summary_path))

    print(json.dumps({"summary_paths": summary_paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
