"""Pandora-owned execution and artifact helpers for DirectMultiStep."""

from __future__ import annotations

import ast
import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from retrocast import (
    create_planner_manifest,
    load_task,
    read_json,
    verify_planner_manifest,
    write_execution_stats,
    write_json,
    write_json_gz,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
DMS_DIR = DATA_DIR / "0-assets" / "model-configs" / "dms"
TASKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"

logger = logging.getLogger("pandora.directmultistep")

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
RouteTree = dict[str, Any]
ExecutionStats = dict[str, dict[str, float]]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class ExecutionTimer:
    """Measure planner wall and CPU time per target without owning execution policy."""

    def __init__(self) -> None:
        self._wall_time: dict[str, float] = {}
        self._cpu_time: dict[str, float] = {}

    @contextmanager
    def measure(self, target_id: str) -> Iterator[None]:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            yield
        finally:
            self._wall_time[target_id] = time.perf_counter() - wall_started
            self._cpu_time[target_id] = time.process_time() - cpu_started

    def to_dict(self) -> ExecutionStats:
        return {
            "wall_time": dict(self._wall_time),
            "cpu_time": dict(self._cpu_time),
        }


def load_dms_task(task_name: str) -> tuple[dict[str, Any], Path]:
    task_path = TASKS_DIR / f"{task_name}.json.gz"
    return load_task(task_path), task_path


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_run_name(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
    if not value or any(character not in allowed for character in value):
        raise ValueError("run name must contain only letters, digits, '-', '_', or '.'")
    return value


def write_effective_config(config: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.effective.yaml"
    with open(path, "w", encoding="utf-8") as fileobj:
        yaml.safe_dump(dict(config), fileobj, sort_keys=True)
    return path


def parse_route_strings(raw_paths: Sequence[str]) -> list[RouteTree]:
    """Convert trusted DMS string output into its documented raw tree shape."""
    routes: list[RouteTree] = []
    for raw_path in raw_paths:
        route = ast.literal_eval(raw_path)
        if not isinstance(route, dict):
            raise ValueError("DirectMultiStep emitted a route that is not an object")
        routes.append(route)
    return routes


def write_planner_artifacts(
    *,
    results: Mapping[str, list[RouteTree]],
    execution_stats: ExecutionStats,
    output_dir: Path,
    sources: Sequence[Path],
    parameters: Mapping[str, JSONValue],
    statistics: Mapping[str, JSONValue],
    action: str,
) -> None:
    results_path = output_dir / "results.json.gz"
    execution_stats_path = output_dir / "execution_stats.json.gz"
    manifest_path = output_dir / "manifest.json"

    write_json_gz(dict(results), results_path)
    write_execution_stats(execution_stats, execution_stats_path)
    manifest = create_planner_manifest(
        action,
        "dms",
        results_path,
        sources,
        DATA_DIR,
        parameters=parameters,
        statistics=statistics,
    )
    write_json(manifest, manifest_path)

    report = verify_planner_manifest(manifest_path, DATA_DIR)
    if not report["is_valid"]:
        messages = "; ".join(str(issue["message"]) for issue in report["issues"])
        raise ValueError(f"created an invalid DirectMultiStep planner manifest: {messages}")


def _require_mapping(value: Any, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def gather_dms_parts(
    *,
    part_dirs: Sequence[Path],
    expected_target_ids: set[str],
) -> tuple[dict[str, list[RouteTree]], ExecutionStats, list[Path]]:
    """Verify and merge partitioned DMS raw runs without silently dropping data."""
    results: dict[str, list[RouteTree]] = {}
    wall_time: dict[str, float] = {}
    cpu_time: dict[str, float] = {}
    sources: list[Path] = []

    for part_dir in part_dirs:
        manifest_path = part_dir / "manifest.json"
        results_path = part_dir / "results.json.gz"
        execution_stats_path = part_dir / "execution_stats.json.gz"

        report = verify_planner_manifest(manifest_path, DATA_DIR)
        if not report["is_valid"]:
            messages = "; ".join(str(issue["message"]) for issue in report["issues"])
            raise ValueError(f"invalid part manifest {manifest_path}: {messages}")

        manifest = _require_mapping(read_json(manifest_path), manifest_path)
        directives = _require_mapping(manifest.get("directives"), manifest_path)
        if directives.get("adapter") != "dms" or directives.get("raw_results_filename") != results_path.name:
            raise ValueError(f"{manifest_path} is not a DirectMultiStep results manifest")

        part_results = _require_mapping(read_json(results_path), results_path)
        part_stats = _require_mapping(read_json(execution_stats_path), execution_stats_path)
        part_wall = _require_mapping(part_stats.get("wall_time"), execution_stats_path)
        part_cpu = _require_mapping(part_stats.get("cpu_time"), execution_stats_path)

        duplicate_ids = set(results).intersection(part_results)
        if duplicate_ids:
            raise ValueError(f"duplicate target IDs across DMS parts: {sorted(duplicate_ids)}")

        results.update(part_results)
        wall_time.update({str(key): float(value) for key, value in part_wall.items()})
        cpu_time.update({str(key): float(value) for key, value in part_cpu.items()})
        sources.extend([manifest_path, execution_stats_path])

    actual_target_ids = set(results)
    if actual_target_ids != expected_target_ids:
        missing = sorted(expected_target_ids - actual_target_ids)
        unexpected = sorted(actual_target_ids - expected_target_ids)
        raise ValueError(f"combined DMS target coverage mismatch: missing={missing}, unexpected={unexpected}")

    if set(wall_time) != actual_target_ids or set(cpu_time) != actual_target_ids:
        raise ValueError("combined DMS execution statistics do not cover every target exactly once")

    return results, {"wall_time": wall_time, "cpu_time": cpu_time}, sources
