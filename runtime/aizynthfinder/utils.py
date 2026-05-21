"""Shared utilities for AiZynthFinder runtime scripts."""

from __future__ import annotations

import argparse
import copy
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from aizynthfinder.aizynthfinder import AiZynthFinder
from retrocast.cli.progress import create_cli_progress
from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import logger
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "retrocast"
AIZYNTHFINDER_DIR = DATA_DIR / "0-assets" / "model-configs" / "aizynthfinder"
BENCHMARKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"


def create_benchmark_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark set (e.g. random-n5-50)",
    )
    parser.add_argument(
        "--effort",
        type=str,
        default="normal",
        choices=["normal", "high"],
        help="Search effort level: normal or high",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of targets to process. Useful for smoke tests.",
    )
    return parser


def load_aizynthfinder_benchmark(
    benchmark_name: str,
) -> tuple[BenchmarkSet, Path]:
    bench_path = BENCHMARKS_DIR / f"{benchmark_name}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {benchmark_name}"
    return benchmark, bench_path


def load_config(config_path: Path, stock_name: str, effort: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as fileobj:
        config = yaml.safe_load(fileobj)

    stock_config = config.get("stock", {})
    if stock_name not in stock_config:
        raise KeyError(f"Stock {stock_name!r} not found in {config_path}")

    stock_path = PROJECT_ROOT / stock_config[stock_name]
    if not stock_path.exists():
        raise FileNotFoundError(
            f"Required stock file {stock_path} does not exist. Create it with runtime/aizynthfinder/2-prepare-stock.py."
        )

    config["stock"] = {stock_name: stock_config[stock_name]}
    if effort == "high":
        config.setdefault("search", {})["iteration_limit"] = 500
    return config


@contextmanager
def quiet_progress_info_logs() -> Iterator[None]:
    """Hide info/debug logs while progress owns the terminal."""
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(logging.DEBUG)


def run_aizynthfinder_predictions(
    benchmark: BenchmarkSet,
    config_path: Path,
    *,
    effort: str,
    limit: int | None,
) -> tuple[dict[str, dict[str, Any]], int, ExecutionStats]:
    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    timer = ExecutionTimer()
    targets = list(benchmark.targets.values())
    if limit is not None:
        targets = targets[:limit]
    os.chdir(PROJECT_ROOT)

    assert benchmark.stock_name is not None
    config = load_config(config_path, benchmark.stock_name, effort)

    with create_cli_progress(console=Console(), unit="target") as progress:
        progress_task = progress.add_task("Finding retrosynthetic paths", total=len(targets))
        with quiet_progress_info_logs():
            for target in targets:
                with timer.measure(target.id):
                    try:
                        finder = AiZynthFinder(configdict=copy.deepcopy(config))
                        finder.stock.select(benchmark.stock_name)
                        finder.expansion_policy.select("uspto")
                        finder.filter_policy.select("uspto")

                        finder.target_smiles = target.smiles
                        finder.tree_search()
                        finder.build_routes()
                        stats = finder.extract_statistics()

                        if stats.get("is_solved", False):
                            solved_count += 1

                        results[target.id] = (
                            finder.routes.dict_with_extra(include_metadata=False, include_scores=True)
                            if finder.routes
                            else {}
                        )
                    except Exception as e:
                        logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                        results[target.id] = {}
                    finally:
                        progress.advance(progress_task)

    for target in targets:
        results.setdefault(target.id, {})

    if not targets:
        logger.warning("No targets selected for processing.")

    return results, solved_count, timer.to_model()


def save_aizynthfinder_results(
    results: dict[str, dict[str, Any]],
    runtime: ExecutionStats,
    save_dir: Path,
    bench_path: Path,
    config_path: Path,
    script_name: str,
    benchmark: BenchmarkSet,
    planner_version: str,
    solved_count: int,
) -> None:
    summary = {
        "solved_count": solved_count,
        "total_targets": len(results),
        "benchmark_total_targets": len(benchmark.targets),
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest = create_manifest(
        action=script_name,
        sources=[bench_path, config_path],
        root_dir=save_dir.parents[2],
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        parameters={
            "adapter": "aizynthfinder",
            "planner_version": planner_version,
            "raw_results_filename": "results.json.gz",
        },
        statistics=summary,
    )

    with open(save_dir / "manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(results)} targets")
    logger.info(f"Solved: {summary['solved_count']}")


def quiet_aizynthfinder_debug_logs() -> None:
    logging.disable(logging.DEBUG)
    logging.getLogger("aizynthfinder").setLevel(logging.INFO)
