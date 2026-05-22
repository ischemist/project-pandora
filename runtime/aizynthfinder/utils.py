"""Shared utilities for AiZynthFinder runtime scripts."""

from __future__ import annotations

import argparse
import logging
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
        "--iteration-limit",
        type=int,
        default=100,
        choices=[100, 500],
        help="Maximum tree search iterations.",
    )
    parser.add_argument(
        "--max-transforms",
        type=int,
        default=6,
        choices=[6, 10],
        help="Maximum route depth to search.",
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


def load_config(config_path: Path, stock_name: str, iteration_limit: int, max_transforms: int) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as fileobj:
        config: dict[str, Any] = yaml.safe_load(fileobj)

    def project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    stock_config = config["stock"]
    if stock_name not in stock_config:
        raise KeyError(f"Stock {stock_name!r} not found in {config_path}")

    stock_path = project_path(stock_config[stock_name])
    if not stock_path.exists():
        raise FileNotFoundError(
            f"Required stock file {stock_path} does not exist. "
            "Create it with runtime/aizynthfinder/dev/prepare-stock.py."
        )

    config["stock"] = {stock_name: str(stock_path)}
    config["expansion"] = {
        policy_name: [str(project_path(path)) for path in paths]
        for policy_name, paths in config.get("expansion", {}).items()
    }
    config["filter"] = {policy_name: str(project_path(path)) for policy_name, path in config.get("filter", {}).items()}

    search_config = config.setdefault("search", {})
    search_config["iteration_limit"] = iteration_limit
    search_config["max_transforms"] = max_transforms

    molecule_cost = search_config.get("algorithm_config", {}).get("molecule_cost", {})
    if "model_path" in molecule_cost:
        molecule_cost["model_path"] = str(project_path(molecule_cost["model_path"]))

    return config


@contextmanager
def quiet_progress_info_logs() -> Iterator[None]:
    """Hide info/debug logs while progress owns the terminal."""
    previous_disable = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous_disable)


def run_aizynthfinder_predictions(
    benchmark: BenchmarkSet,
    config_path: Path,
    *,
    iteration_limit: int,
    max_transforms: int,
    limit: int | None,
) -> tuple[dict[str, dict[str, Any]], int, ExecutionStats]:
    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    timer = ExecutionTimer()
    targets = list(benchmark.targets.values())
    if limit is not None:
        targets = targets[:limit]

    assert benchmark.stock_name is not None
    config = load_config(config_path, benchmark.stock_name, iteration_limit, max_transforms)
    finder = AiZynthFinder(configdict=config)
    finder.stock.select(benchmark.stock_name)
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")

    with create_cli_progress(console=Console(), unit="target") as progress:
        progress_task = progress.add_task("Finding retrosynthetic paths", total=len(targets))
        with quiet_progress_info_logs():
            for target in targets:
                with timer.measure(target.id):
                    try:
                        finder.target_smiles = target.smiles
                        finder.prepare_tree()
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
    iteration_limit: int,
    max_transforms: int,
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
            "iteration_limit": iteration_limit,
            "max_transforms": max_transforms,
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
