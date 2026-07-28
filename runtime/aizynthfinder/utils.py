"""Shared utilities for AiZynthFinder runtime scripts."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from aizynthfinder.aizynthfinder import AiZynthFinder
from retrocast import (
    create_planner_manifest,
    load_task,
    resolve_stock_bindings,
    verify_planner_manifest,
    write_execution_stats,
    write_json,
    write_json_gz,
)
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
AIZYNTHFINDER_DIR = DATA_DIR / "0-assets" / "model-configs" / "aizynthfinder"
BENCHMARKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"
logger = logging.getLogger("pandora.aizynthfinder")

Task = dict[str, Any]
ExecutionStats = dict[str, dict[str, float]]


def configure_script_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class ExecutionTimer:
    def __init__(self) -> None:
        self.wall_time: dict[str, float] = {}
        self.cpu_time: dict[str, float] = {}

    @contextmanager
    def measure(self, target_id: str) -> Iterator[None]:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            yield
        finally:
            self.wall_time[target_id] = time.perf_counter() - wall_start
            self.cpu_time[target_id] = time.process_time() - cpu_start

    def to_dict(self) -> ExecutionStats:
        return {"wall_time": self.wall_time, "cpu_time": self.cpu_time}


@contextmanager
def create_cli_progress(*, console: Console, unit: str) -> Iterator[Progress]:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(unit),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        yield progress


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


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
        type=positive_int,
        default=None,
        help="Maximum number of targets to process. Useful for smoke tests.",
    )
    parser.add_argument(
        "--shard-count",
        type=positive_int,
        default=1,
        help="Number of round-robin target shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=nonnegative_int,
        default=0,
        help="Zero-based round-robin shard index.",
    )
    return parser


def load_aizynthfinder_benchmark(
    benchmark_name: str,
) -> tuple[Task, Path]:
    bench_path = BENCHMARKS_DIR / f"{benchmark_name}.json.gz"
    benchmark = load_task(bench_path)
    benchmark_stock_name(benchmark)
    return benchmark, bench_path


def benchmark_stock_name(benchmark: Task) -> str:
    bindings = resolve_stock_bindings(benchmark)
    stock_names = set(bindings.values())
    if None in stock_names:
        missing = sorted(target_id for target_id, stock_name in bindings.items() if stock_name is None)
        raise ValueError(f"Targets have no effective stock binding: {missing}")
    if len(stock_names) != 1:
        raise ValueError(f"AiZynthFinder runtime requires one effective stock, got {sorted(stock_names)}")
    return next(iter(stock_names))


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


def write_effective_config(config: dict[str, Any], save_dir: Path) -> Path:
    effective_config_path = save_dir / "config.effective.yaml"
    with open(effective_config_path, "w", encoding="utf-8") as fileobj:
        yaml.safe_dump(config, fileobj, sort_keys=True)
    return effective_config_path


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
    benchmark: Task,
    config: dict[str, Any],
    *,
    limit: int | None,
    shard_count: int = 1,
    shard_index: int = 0,
    expansion_policy_name: str = "uspto",
) -> tuple[dict[str, dict[str, Any]], int, ExecutionStats]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(f"Invalid shard {shard_index} of {shard_count}")

    results: dict[str, dict[str, Any]] = {}
    solved_count = 0
    timer = ExecutionTimer()
    targets = list(benchmark["targets"].values())[shard_index::shard_count]
    if limit is not None:
        targets = targets[:limit]

    stock_name = benchmark_stock_name(benchmark)
    finder = AiZynthFinder(configdict=config)
    finder.stock.select(stock_name)
    finder.expansion_policy.select(expansion_policy_name)
    # Keep the filter fixed so checkpoint comparisons isolate the expansion policy.
    finder.filter_policy.select("uspto")

    with create_cli_progress(console=Console(), unit="target") as progress:
        progress_task = progress.add_task("Finding retrosynthetic paths", total=len(targets))
        with quiet_progress_info_logs():
            for target in targets:
                target_id = target["id"]
                target_smiles = target["smiles"]
                with timer.measure(target_id):
                    try:
                        finder.target_smiles = target_smiles
                        finder.prepare_tree()
                        finder.tree_search()
                        finder.build_routes()
                        stats = finder.extract_statistics()

                        if stats.get("is_solved", False):
                            solved_count += 1

                        results[target_id] = (
                            finder.routes.dict_with_extra(include_metadata=False, include_scores=True)
                            if finder.routes
                            else {}
                        )
                    except Exception as e:
                        logger.error("Failed to process target %s (%s): %s", target_id, target_smiles, e, exc_info=True)
                        results[target_id] = {}
                    finally:
                        progress.advance(progress_task)

    for target in targets:
        results.setdefault(target["id"], {})

    if not targets:
        logger.warning("No targets selected for processing.")

    return results, solved_count, timer.to_dict()


def shard_save_dir(base_dir: Path, *, shard_count: int, shard_index: int) -> Path:
    if shard_count == 1:
        return base_dir
    return base_dir / "shards" / f"part-{shard_index:02d}-of-{shard_count:02d}"


def save_aizynthfinder_results(
    results: dict[str, dict[str, Any]],
    runtime: ExecutionStats,
    save_dir: Path,
    bench_path: Path,
    effective_config_path: Path,
    config_template_path: Path,
    script_name: str,
    benchmark: Task,
    parameters: dict[str, Any],
    solved_count: int,
    additional_sources: list[Path] | None = None,
) -> None:
    summary = {
        "solved_count": solved_count,
        "total_targets": len(results),
        "benchmark_total_targets": len(benchmark["targets"]),
    }

    results_path = save_dir / "results.json.gz"
    manifest_path = save_dir / "manifest.json"
    write_json_gz(results, results_path)
    write_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest_parameters = {
        **parameters,
        "config_template_path": str(config_template_path.relative_to(DATA_DIR)),
        "effective_config_path": str(effective_config_path.relative_to(DATA_DIR)),
    }
    manifest = create_planner_manifest(
        action=script_name,
        adapter="aizynthfinder",
        raw_results_path=results_path,
        sources=[bench_path, effective_config_path, *(additional_sources or [])],
        root_dir=DATA_DIR,
        parameters=manifest_parameters,
        statistics=summary,
    )
    write_json(manifest, manifest_path)
    verification = verify_planner_manifest(manifest_path, DATA_DIR)
    if not verification["is_valid"]:
        raise ValueError(f"Planner manifest verification failed: {verification['issues']}")

    logger.info(f"Completed processing {len(results)} targets")
    logger.info(f"Solved: {summary['solved_count']}")


def quiet_aizynthfinder_debug_logs() -> None:
    logging.disable(logging.DEBUG)
    logging.getLogger("aizynthfinder").setLevel(logging.INFO)
