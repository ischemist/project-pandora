"""Pandora-owned execution and gather utilities for the ASKCOS runtime."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import retrocast
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
TASKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"

logger = logging.getLogger("pandora.askcos")


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, str],
        timeout: int,
    ) -> HttpResponse: ...


@dataclass
class ExecutionTimer:
    """Measure planner wall and CPU time per target."""

    wall_time: dict[str, float] = field(default_factory=dict)
    cpu_time: dict[str, float] = field(default_factory=dict)

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


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def resolve_path_within_root(value: str, root: Path) -> Path:
    resolved_root = root.resolve()
    path = Path(value)
    candidate = (path if path.is_absolute() else resolved_root / path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"path must be inside {resolved_root}") from error
    return candidate


def task_path(task_name: str) -> Path:
    return TASKS_DIR / f"{task_name}.json.gz"


def load_task(task_name: str) -> tuple[dict[str, Any], Path, str | None]:
    path = task_path(task_name)
    task = retrocast.load_task(path)
    stock_name = effective_stock_name(task)
    return task, path, stock_name


def effective_stock_name(task: Mapping[str, Any]) -> str | None:
    bindings = retrocast.resolve_stock_bindings(task)
    stock_names = {name for name in bindings.values() if name is not None}
    if len(stock_names) > 1:
        raise ValueError(
            "ASKCOS uses one server-side stock per run, but the task resolves to multiple stocks: "
            f"{sorted(stock_names)}"
        )
    return next(iter(stock_names), None)


def write_effective_config(config: Mapping[str, Any], save_dir: Path) -> Path:
    path = save_dir / "config.effective.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fileobj:
        yaml.safe_dump(dict(config), fileobj, sort_keys=True)
    return path


def write_run_artifacts(
    *,
    results: Mapping[str, Any],
    execution_stats: Mapping[str, Mapping[str, float]],
    save_dir: Path,
    task: Mapping[str, Any],
    task_source: Path,
    effective_config_path: Path,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    results_path = save_dir / "results.json.gz"
    execution_stats_path = save_dir / "execution_stats.json.gz"
    manifest_path = save_dir / "manifest.json"

    retrocast.write_json_gz(dict(results), results_path)
    retrocast.write_execution_stats(execution_stats, execution_stats_path)

    successful_requests, failed_requests = request_counts(results)
    statistics = {
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "total_targets": len(results),
        "task_total_targets": len(task["targets"]),
    }
    manifest = retrocast.create_planner_manifest(
        "runtime/askcos/1-run-askcos.py",
        "askcos",
        results_path,
        [task_source, effective_config_path],
        DATA_DIR,
        parameters=parameters,
        statistics=statistics,
    )
    retrocast.write_json(manifest, manifest_path)
    report = retrocast.verify_planner_manifest(manifest_path, DATA_DIR)
    if not report["is_valid"]:
        raise RuntimeError(f"ASKCOS planner manifest failed verification: {report['issues']}")
    return manifest


def request_counts(results: Mapping[str, Any]) -> tuple[int, int]:
    successful_requests = sum(result is not None for result in results.values())
    return successful_requests, len(results) - successful_requests


def normalize_target_id(target_id: str) -> str:
    normalized = target_id.replace(" ", "_").replace("/", "_").replace("(", "_").replace(")", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _select_unambiguous(candidates: list[Path], description: str) -> Path | None:
    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"multiple ASKCOS files match {description}: {names}")
    return candidates[0] if candidates else None


def find_result_file(position: int, target_id: str, eval_dir: Path) -> Path | None:
    json_files = sorted(eval_dir.glob("*.json"))
    by_position = [path for path in json_files if path.name.startswith(f"{position:04d}_")]
    match = _select_unambiguous(by_position, f"position {position}")
    if match is not None:
        return match

    normalized_id = normalize_target_id(target_id)
    by_name = [path for path in json_files if path.stem.endswith(target_id) or path.stem.endswith(normalized_id)]
    match = _select_unambiguous(by_name, f"target {target_id!r}")
    if match is not None:
        return match

    if "(" in target_id and ")" in target_id:
        main_name = target_id.split(")")[-1].strip("-").strip()
        partial = [path for path in json_files if main_name in path.stem]
        return _select_unambiguous(partial, f"target {target_id!r}")
    return None


def gather_results(
    task: Mapping[str, Any],
    eval_dir: Path,
) -> tuple[dict[str, Any], list[Path], list[str]]:
    if not eval_dir.is_dir():
        raise OSError(f"ASKCOS evaluation directory does not exist: {eval_dir}")

    matches: list[tuple[str, Path | None]] = []
    for position, (target_id, _target) in enumerate(task["targets"].items(), start=1):
        result_path = find_result_file(position, target_id, eval_dir)
        matches.append((target_id, result_path))

    target_ids_by_result: dict[Path, list[str]] = {}
    for target_id, result_path in matches:
        if result_path is not None:
            target_ids_by_result.setdefault(result_path.resolve(), []).append(target_id)
    collisions = {
        result_path: target_ids
        for result_path, target_ids in target_ids_by_result.items()
        if len(target_ids) > 1
    }
    if collisions:
        details = "; ".join(
            f"{result_path.name}: {', '.join(repr(target_id) for target_id in target_ids)}"
            for result_path, target_ids in sorted(collisions.items())
        )
        raise ValueError(f"ASKCOS result files match multiple task targets: {details}")

    results: dict[str, Any] = {}
    sources: list[Path] = []
    missing: list[str] = []
    for target_id, result_path in matches:
        if result_path is None:
            missing.append(target_id)
            continue
        sources.append(result_path)
        try:
            result = retrocast.read_json(result_path)
        except (OSError, ValueError) as error:
            logger.error("Could not load %s for target %s: %s", result_path, target_id, error)
            missing.append(target_id)
            continue
        if not isinstance(result, dict):
            logger.error("ASKCOS result %s is not a JSON object", result_path)
            missing.append(target_id)
            continue
        results[target_id] = result
    return results, sources, missing


def write_gathered_artifacts(
    *,
    results: Mapping[str, Any],
    source_files: list[Path],
    missing_targets: list[str],
    task: Mapping[str, Any],
    task_source: Path,
    output_dir: Path,
    eval_dir: Path,
) -> dict[str, Any]:
    results_path = output_dir / "results.json.gz"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    retrocast.write_json_gz(dict(results), results_path)

    manifest = retrocast.create_planner_manifest(
        "runtime/askcos/2-gather-askcos-results.py",
        "askcos",
        results_path,
        [task_source, *source_files],
        DATA_DIR,
        parameters={
            "gather_mode": "position-then-target-name",
            "input_directory": str(eval_dir.relative_to(DATA_DIR.resolve())),
        },
        statistics={
            "gathered_targets": len(results),
            "missing_targets": len(missing_targets),
            "task_total_targets": len(task["targets"]),
        },
    )
    retrocast.write_json(manifest, manifest_path)
    report = retrocast.verify_planner_manifest(manifest_path, DATA_DIR)
    if not report["is_valid"]:
        raise RuntimeError(f"ASKCOS gather manifest failed verification: {report['issues']}")
    return manifest
