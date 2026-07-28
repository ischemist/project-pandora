"""Shared execution and artifact helpers for Syntheseus runtimes."""

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
from retrocast import (
    create_planner_manifest,
    load_stock,
    load_task,
    resolve_stock_bindings,
    verify_planner_manifest,
    write_execution_stats,
    write_json,
    write_json_gz,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
BENCHMARKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
STOCKS_DIR = DATA_DIR / "1-benchmarks" / "stocks"
RAW_DIR = DATA_DIR / "2-raw"
logger = logging.getLogger("pandora.syntheseus")

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
        cpu_start = time.thread_time()
        try:
            yield
        finally:
            self.wall_time[target_id] = time.perf_counter() - wall_start
            self.cpu_time[target_id] = time.thread_time() - cpu_start

    def to_dict(self) -> ExecutionStats:
        return {"wall_time": self.wall_time, "cpu_time": self.cpu_time}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def create_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--benchmark", required=True, help="Task name under data/retrocast/1-benchmarks/definitions")
    parser.add_argument("--effort", choices=["normal", "high"], default="normal")
    parser.add_argument("--limit", type=positive_int)
    return parser


def load_task_and_stock(benchmark_name: str) -> tuple[Task, list[str], Path, Path, str]:
    task_path = BENCHMARKS_DIR / f"{benchmark_name}.json.gz"
    task = load_task(task_path)
    bindings = resolve_stock_bindings(task)
    stock_names = set(bindings.values())
    if None in stock_names:
        missing = sorted(target_id for target_id, stock_name in bindings.items() if stock_name is None)
        raise ValueError(f"Targets have no effective stock binding: {missing}")
    if len(stock_names) != 1:
        raise ValueError(f"Syntheseus runtime requires one effective stock, got {sorted(stock_names)}")
    stock_name = next(iter(stock_names))
    stock_path = STOCKS_DIR / f"{stock_name}.csv.gz"
    return task, load_stock(stock_path), task_path, stock_path, stock_name


def write_effective_config(config: dict[str, Any], save_dir: Path) -> Path:
    path = save_dir / "config.effective.yaml"
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=True)
    return path


def publish_results(
    *,
    results: dict[str, list[dict[str, Any]]],
    runtime: ExecutionStats,
    save_dir: Path,
    task_path: Path,
    stock_path: Path,
    effective_config_path: Path,
    action: str,
    parameters: dict[str, Any],
    task: Task,
) -> None:
    results_path = save_dir / "results.json.gz"
    manifest_path = save_dir / "manifest.json"
    summary = {
        "solved_count": sum(bool(routes) for routes in results.values()),
        "total_targets": len(results),
        "benchmark_total_targets": len(task["targets"]),
    }

    write_json_gz(results, results_path)
    write_execution_stats(runtime, save_dir / "execution_stats.json.gz")
    manifest = create_planner_manifest(
        action,
        "syntheseus",
        results_path,
        [task_path, stock_path, effective_config_path],
        DATA_DIR,
        parameters=parameters,
        statistics=summary,
    )
    write_json(manifest, manifest_path)
    verification = verify_planner_manifest(manifest_path, DATA_DIR)
    if not verification["is_valid"]:
        raise ValueError(f"Planner manifest verification failed: {verification['issues']}")

    logger.info("Completed processing %s targets", len(results))
    logger.info("Solved: %s", summary["solved_count"])
