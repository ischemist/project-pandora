"""Pandora-owned runtime support for the original Retro* planner."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import retrocast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
ASSET_DIR = Path(
    os.environ.get(
        "RETROSTAR_ASSETS_DIR",
        DATA_DIR / "0-assets" / "model-configs" / "retro-star",
    )
)
TASKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
STOCKS_DIR = DATA_DIR / "1-benchmarks" / "stocks"
RAW_DIR = DATA_DIR / "2-raw"

logger = logging.getLogger("pandora.retrostar")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run original Retro* retrosynthesis planning")
    parser.add_argument("--benchmark", required=True, help="Schema-v2 task name.")
    parser.add_argument(
        "--effort",
        default="normal",
        choices=("normal", "high"),
        help="Search effort: normal uses 100 iterations; high uses 500.",
    )
    parser.add_argument("--limit", type=positive_int, help="Maximum number of targets to process.")
    return parser


def load_task_and_stock(benchmark_name: str) -> tuple[dict[str, Any], Path, Path, str]:
    task_path = TASKS_DIR / f"{benchmark_name}.json.gz"
    task = retrocast.load_task(task_path)
    bindings = retrocast.resolve_stock_bindings(task)
    stock_names = {name for name in bindings.values() if name is not None}

    if len(stock_names) != 1 or any(name is None for name in bindings.values()):
        raise ValueError(
            "original Retro* requires exactly one stock shared by every target; "
            f"resolved bindings were {sorted(stock_names)!r}"
        )

    stock_name = stock_names.pop()
    stock_path = STOCKS_DIR / f"{stock_name}.csv.gz"
    if not stock_path.is_file():
        raise FileNotFoundError(f"Retro* stock not found: {stock_path}")

    return task, task_path, stock_path, stock_name


def write_retrostar_stock(stock_path: Path, destination: Path) -> None:
    """Project a shared stock into Retro*'s required uncompressed `mol` CSV."""
    smiles = retrocast.load_stock(stock_path, representation="smiles")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as fileobj:
        fileobj.write("mol\n")
        for value in smiles:
            fileobj.write(f"{value}\n")


def convert_numpy(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: convert_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_numpy(item) for item in value]
    if isinstance(value, tuple):
        return [convert_numpy(item) for item in value]
    return value


class ExecutionTimer:
    """Measure the external planner call for each target."""

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

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"wall_time": self.wall_time, "cpu_time": self.cpu_time}


def save_results(
    *,
    results: dict[str, dict[str, Any]],
    execution_stats: dict[str, dict[str, float]],
    save_dir: Path,
    task_path: Path,
    stock_path: Path,
    effective_stock_path: Path,
    effective_config_path: Path,
    parameters: dict[str, Any],
) -> None:
    results_path = save_dir / "results.json.gz"
    execution_stats_path = save_dir / "execution_stats.json.gz"
    manifest_path = save_dir / "manifest.json"
    solved_count = sum(bool(result) for result in results.values())
    statistics = {
        "solved_count": solved_count,
        "total_targets": len(results),
    }

    retrocast.write_json_gz(results, results_path)
    retrocast.write_execution_stats(execution_stats, execution_stats_path)
    manifest = retrocast.create_planner_manifest(
        "runtime/retrostar/2-run-retrostar.py",
        "retrostar",
        results_path,
        [task_path, stock_path, effective_stock_path, effective_config_path],
        DATA_DIR,
        parameters=parameters,
        statistics=statistics,
    )
    retrocast.write_json(manifest, manifest_path)
    verification = retrocast.verify_planner_manifest(manifest_path, DATA_DIR, deep=True)
    if not verification["is_valid"]:
        raise RuntimeError(f"Retro* planner manifest verification failed: {verification['issues']}")

    logger.info("Completed processing %d targets", len(results))
    logger.info("Solved: %d", solved_count)
