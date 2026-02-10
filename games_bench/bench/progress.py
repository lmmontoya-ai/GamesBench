from __future__ import annotations

import importlib
import sys
import threading
from typing import Any


class EpisodeProgressReporter:
    def on_episode_complete(self, episode: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class NoopEpisodeProgressReporter(EpisodeProgressReporter):
    def on_episode_complete(self, episode: dict[str, Any]) -> None:  # noqa: ARG002
        return

    def close(self) -> None:
        return


class TqdmEpisodeProgressReporter(EpisodeProgressReporter):
    def __init__(
        self,
        *,
        total_episodes: int,
        refresh_s: float,
        tqdm_cls: Any,
    ) -> None:
        self._lock = threading.Lock()
        self._completed = 0
        self._solved = 0
        self._bar = tqdm_cls(
            total=max(0, int(total_episodes)),
            desc="Episodes",
            unit="ep",
            dynamic_ncols=True,
            mininterval=float(refresh_s),
            file=sys.stderr,
            leave=True,
        )

    def on_episode_complete(self, episode: dict[str, Any]) -> None:
        with self._lock:
            self._completed += 1
            if bool(episode.get("solved")):
                self._solved += 1

            game = str(episode.get("game", "?"))
            episode_id = _safe_int(episode.get("episode_id"))
            turns = _safe_int(episode.get("turn_count"))
            moves = _safe_int(episode.get("move_count"))
            illegal = _safe_int(episode.get("illegal_moves"))

            postfix = {
                "game": game,
                "ep": "-" if episode_id is None else str(episode_id),
                "turns": "-" if turns is None else str(turns),
                "moves": "-" if moves is None else str(moves),
                "illegal": "-" if illegal is None else str(illegal),
                "solved": f"{self._solved}/{self._completed}",
            }
            self._bar.set_postfix(postfix, refresh=False)
            self._bar.update(1)

    def close(self) -> None:
        with self._lock:
            self._bar.close()


def build_episode_progress_reporter(
    *,
    enabled: bool,
    total_episodes: int,
    refresh_s: float,
    explicit_request: bool,
) -> EpisodeProgressReporter:
    if not enabled or total_episodes <= 0:
        return NoopEpisodeProgressReporter()

    try:
        tqdm_module = importlib.import_module("tqdm")
    except ImportError:
        if explicit_request:
            print(
                "Progress requested but missing dependency: tqdm. Install with "
                "pip install 'games-bench[bench]' or uv sync --group bench.",
                file=sys.stderr,
                flush=True,
            )
        return NoopEpisodeProgressReporter()

    tqdm_cls = getattr(tqdm_module, "tqdm", None)
    if tqdm_cls is None:
        if explicit_request:
            print(
                "Progress requested but tqdm could not be loaded. Install with "
                "pip install 'games-bench[bench]' or uv sync --group bench.",
                file=sys.stderr,
                flush=True,
            )
        return NoopEpisodeProgressReporter()
    return TqdmEpisodeProgressReporter(
        total_episodes=total_episodes,
        refresh_s=refresh_s,
        tqdm_cls=tqdm_cls,
    )


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
