"""
Gather native DreamRetroer JSON into the raw artifact contract consumed by RetroCast.

Examples:
    uv run --directory runtime/dreamretroer 1-gather-results.py --benchmark uspto-190
    uv run --directory runtime/dreamretroer 1-gather-results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    RAW_DIR,
    configure_logging,
    gather_benchmark,
    logger,
    resolve_additional_sources,
    safe_path_component,
)


def discover_benchmarks(run_name: str) -> list[str]:
    run_dir = RAW_DIR / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"DreamRetroer raw run directory not found: {run_dir}")
    return sorted(path.name for path in run_dir.iterdir() if path.is_dir() and (path / "results.json").is_file())


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Gather native DreamRetroer results")
    parser.add_argument(
        "--benchmark",
        action="append",
        type=safe_path_component,
        help="Benchmark to gather; repeat for multiple benchmarks. Defaults to every native results directory.",
    )
    parser.add_argument("--run-name", type=safe_path_component, default="dream-retroer")
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[],
        help="Additional planner input to hash, relative to RETROCAST_DATA_DIR. Repeat as needed.",
    )
    args = parser.parse_args()

    additional_sources = resolve_additional_sources(args.source)
    benchmark_names = args.benchmark or discover_benchmarks(args.run_name)
    if not benchmark_names:
        raise ValueError(f"no native DreamRetroer results found under {RAW_DIR / args.run_name}")

    for benchmark_name in benchmark_names:
        manifest_path = gather_benchmark(
            run_name=args.run_name,
            benchmark_name=benchmark_name,
            additional_sources=additional_sources,
        )
        logger.info("Gathered %s into %s", benchmark_name, manifest_path.parent)


if __name__ == "__main__":
    main()
