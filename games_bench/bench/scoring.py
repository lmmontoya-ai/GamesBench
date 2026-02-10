from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from games_bench.bench.lineage import ensure_run_manifest, make_lineage_event
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


def _coerce_failure_tags(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise SystemExit(
        "episode_taxonomy hook must return failure tags as a list/tuple, "
        "a (outcome_code, tags) tuple, or a dict payload."
    )


def enrich_episodes_with_taxonomy(
    episodes: list[dict[str, Any]],
    *,
    game_name: str,
    run_config: dict[str, Any],
    episode_taxonomy: (
        Callable[
            [dict[str, Any], dict[str, Any]],
            dict[str, Any] | tuple[str, list[str]] | list[str],
        ]
        | None
    ) = None,
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    run_config_payload = dict(run_config)
    for episode in episodes:
        base = annotate_episode_with_taxonomy(episode, game_name=game_name)
        if episode_taxonomy is None:
            enriched_rows.append(base)
            continue

        hook_value = episode_taxonomy(dict(episode), run_config_payload)
        if isinstance(hook_value, dict):
            merged = dict(base)
            merged.update(hook_value)
            merged["taxonomy_version"] = str(
                merged.get("taxonomy_version") or TAXONOMY_VERSION
            )
            merged["outcome_code"] = str(
                merged.get("outcome_code") or base["outcome_code"]
            )
            merged["failure_tags"] = _coerce_failure_tags(
                merged.get("failure_tags", base["failure_tags"])
            )
            enriched_rows.append(merged)
            continue

        if isinstance(hook_value, tuple) and len(hook_value) == 2:
            outcome_code, tags = hook_value
            merged = dict(base)
            merged["outcome_code"] = str(outcome_code)
            merged["failure_tags"] = _coerce_failure_tags(tags)
            enriched_rows.append(merged)
            continue

        if isinstance(hook_value, list):
            merged = dict(base)
            merged["failure_tags"] = _coerce_failure_tags(hook_value)
            enriched_rows.append(merged)
            continue

        raise SystemExit(
            "episode_taxonomy hook must return dict, (outcome_code, tags), or tags list."
        )

    return enriched_rows


def build_summary_document(
    *,
    run_config: dict[str, Any],
    episodes: list[dict[str, Any]],
    score_episodes: Callable[[list[dict[str, Any]]], dict[str, Any]],
    score_version: str,
    game_name: str,
    scoring_input: dict[str, Any] | None = None,
    episode_taxonomy: (
        Callable[
            [dict[str, Any], dict[str, Any]],
            dict[str, Any] | tuple[str, list[str]] | list[str],
        ]
        | None
    ) = None,
    episodes_are_enriched: bool = False,
) -> dict[str, Any]:
    if episodes_are_enriched:
        enriched_episodes = [dict(episode) for episode in episodes]
    else:
        enriched_episodes = enrich_episodes_with_taxonomy(
            episodes,
            game_name=game_name,
            run_config=run_config,
            episode_taxonomy=episode_taxonomy,
        )

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

    from games_bench.bench.registry import get_benchmark, load_builtin_benchmarks

    load_builtin_benchmarks()
    benchmark = get_benchmark(detected_game)
    score_fn = benchmark.episode_scorer or benchmark.score_episodes
    if score_fn is None:
        raise SystemExit(
            f"Benchmark '{detected_game}' does not expose a scoring function."
        )
    episode_taxonomy = benchmark.episode_taxonomy

    episodes = read_jsonl(episodes_path)
    summary_episodes = episodes
    if write_taxonomy:
        summary_episodes = enrich_episodes_with_taxonomy(
            episodes,
            game_name=detected_game,
            run_config=run_config,
            episode_taxonomy=episode_taxonomy,
        )
        _write_jsonl(episodes_path, summary_episodes)

    parent_run_id_raw = run_config.get("run_id")
    parent_run_id = (
        str(parent_run_id_raw).strip() if parent_run_id_raw is not None else None
    )
    if parent_run_id == "":
        parent_run_id = None

    summary = build_summary_document(
        run_config=run_config,
        episodes=summary_episodes,
        score_episodes=score_fn,
        score_version=score_version,
        game_name=detected_game,
        episode_taxonomy=episode_taxonomy,
        episodes_are_enriched=bool(write_taxonomy),
        scoring_input={
            "source": "offline",
            "run_dir": str(run_dir),
            "run_config_file": str(run_config_path),
            "episodes_file": str(episodes_path),
            "run_config_sha256": _sha256_file(run_config_path),
            "episodes_sha256": _sha256_file(episodes_path),
            "episodes_count": len(summary_episodes),
            "parent_run_id": parent_run_id,
        },
    )

    summary_path.write_text(json.dumps(summary, indent=2))
    ensure_run_manifest(
        run_dir,
        run_config=run_config,
        game_config=None,
        parent_run_id=parent_run_id,
        lineage_event=make_lineage_event(
            "rescored",
            payload={
                "score_version": score_version,
                "write_taxonomy": bool(write_taxonomy),
            },
        ),
    )
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
