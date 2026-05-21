"""Shared utilities for AiZynthFinder runtime scripts."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from aizynthfinder.aizynthfinder import AiZynthFinder
from tqdm import tqdm

from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.paths import get_paths
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import logger


@dataclass
class AizynthfinderPaths:
    """Standard paths for AiZynthFinder resources."""

    project_root: Path
    data_dir: Path
    aizynthfinder_dir: Path
    benchmarks_dir: Path
    raw_dir: Path


def get_aizynthfinder_paths() -> AizynthfinderPaths:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "retrocast"
    paths = get_paths(data_dir)
    return AizynthfinderPaths(
        project_root=project_root,
        data_dir=data_dir,
        aizynthfinder_dir=data_dir / "0-assets" / "model-configs" / "aizynthfinder",
        benchmarks_dir=paths["benchmarks"],
        raw_dir=paths["raw"],
    )


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
    paths: AizynthfinderPaths,
) -> tuple[BenchmarkSet, Path]:
    bench_path = paths.benchmarks_dir / f"{benchmark_name}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {benchmark_name}"
    return benchmark, bench_path


def iter_targets(benchmark: BenchmarkSet, limit: int | None) -> Iterable[Any]:
    targets = benchmark.targets.values()
    if limit is None:
        yield from targets
        return

    for index, target in enumerate(targets):
        if index >= limit:
            break
        yield target


@contextmanager
def pruned_stock_config(config_path: Path, stock_name: str, project_root: Path) -> Iterable[Path]:
    """Write a temporary AiZynthFinder config containing only the selected stock.

    AiZynthFinder loads every stock declared in the config before selection, so
    a benchmark using buyables-stock can fail if unrelated hdf5 stocks are absent.
    """
    with open(config_path, encoding="utf-8") as fileobj:
        config = yaml.safe_load(fileobj)

    stock_config = config.get("stock", {})
    if stock_name not in stock_config:
        raise KeyError(f"Stock {stock_name!r} not found in {config_path}")

    stock_path = project_root / stock_config[stock_name]
    if not stock_path.exists():
        raise FileNotFoundError(
            f"Required stock file {stock_path} does not exist. "
            "Create it with runtime/aizynthfinder/2-prepare-stock.py."
        )

    config["stock"] = {stock_name: stock_config[stock_name]}

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fileobj:
        yaml.safe_dump(config, fileobj, sort_keys=False)
        temp_config_path = Path(fileobj.name)

    try:
        yield temp_config_path
    finally:
        temp_config_path.unlink(missing_ok=True)


def run_aizynthfinder_predictions(
    benchmark: BenchmarkSet,
    config_path: Path,
    *,
    project_root: Path,
    limit: int | None,
) -> tuple[dict[str, dict[str, Any]], int, ExecutionStats]:
    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    timer = ExecutionTimer()
    targets = list(iter_targets(benchmark, limit))
    os.chdir(project_root)

    assert benchmark.stock_name is not None
    with pruned_stock_config(config_path, benchmark.stock_name, project_root) as run_config_path:
        for target in tqdm(targets, desc="Finding retrosynthetic paths"):
            with timer.measure(target.id):
                try:
                    finder = AiZynthFinder(configfile=str(run_config_path))
                    finder.stock.select(benchmark.stock_name)
                    finder.expansion_policy.select("uspto")
                    finder.filter_policy.select("uspto")

                    finder.target_smiles = target.smiles
                    finder.tree_search()
                    finder.build_routes()
                    stats = finder.extract_statistics()

                    if finder.routes:
                        results[target.id] = finder.routes.dict_with_extra(include_metadata=False, include_scores=True)
                        if stats.get("is_solved", False):
                            solved_count += 1
                    else:
                        results[target.id] = {}
                except Exception as e:
                    logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                    results[target.id] = {}

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
