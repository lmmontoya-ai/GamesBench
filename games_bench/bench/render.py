from __future__ import annotations

import argparse
import sys

from games_bench.bench.registry import get_benchmark, load_builtin_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--game", default="hanoi")
    args, remaining = parser.parse_known_args()

    load_builtin_benchmarks()
    try:
        benchmark = get_benchmark(args.game)
    except KeyError as exc:
        raise SystemExit(f"Unknown game: {args.game}") from exc
    if benchmark.render_main is None:
        raise SystemExit(f"Benchmark '{args.game}' does not define a render command.")

    sys.argv = [sys.argv[0], *remaining]
    return benchmark.render_main()


if __name__ == "__main__":
    raise SystemExit(main())
