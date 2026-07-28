"""Pandora-owned gathering and serialization for native DreamRetroer output."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
TASKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"

logger = logging.getLogger("pandora.dreamretroer")

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
DreamRetroResults = dict[str, dict[str, JSONValue]]
ExecutionStats = dict[str, dict[str, float]]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def safe_path_component(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
    if not value or any(character not in allowed for character in value) or value in {".", ".."}:
        raise ValueError("path component must contain only letters, digits, '-', '_', or '.'")
    return value


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def normalize_native_results(
    value: Any,
    *,
    expected_target_ids: set[str],
) -> tuple[DreamRetroResults, ExecutionStats]:
    """Normalize upstream RSPlanner output without adapting its route strings."""
    native_results = _require_mapping(value, label="DreamRetroer results")
    actual_target_ids = set(native_results)
    if actual_target_ids != expected_target_ids:
        missing = sorted(expected_target_ids - actual_target_ids)
        unexpected = sorted(actual_target_ids - expected_target_ids)
        raise ValueError(f"DreamRetroer target coverage mismatch: missing={missing}, unexpected={unexpected}")

    results: DreamRetroResults = {}
    wall_time: dict[str, float] = {}
    for target_id, native_payload in native_results.items():
        if native_payload is None:
            results[target_id] = {"succ": False}
            continue

        payload = _require_mapping(native_payload, label=f"DreamRetroer result for {target_id}")
        success = payload.get("succ")
        if not isinstance(success, bool):
            raise ValueError(f"DreamRetroer result for {target_id} requires a boolean 'succ'")
        if success and (not isinstance(payload.get("routes"), str) or not payload["routes"]):
            raise ValueError(f"successful DreamRetroer result for {target_id} requires a non-empty 'routes' string")

        elapsed = payload.get("time")
        if elapsed is not None:
            if isinstance(elapsed, bool) or not isinstance(elapsed, int | float):
                raise ValueError(f"DreamRetroer result for {target_id} has a non-numeric 'time'")
            elapsed = float(elapsed)
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"DreamRetroer result for {target_id} has an invalid 'time'")
            wall_time[target_id] = elapsed

        results[target_id] = payload

    return results, {"wall_time": wall_time, "cpu_time": {}}


def gather_benchmark(
    *,
    run_name: str,
    benchmark_name: str,
    additional_sources: Sequence[Path] = (),
) -> Path:
    task_path = TASKS_DIR / f"{benchmark_name}.json.gz"
    task = load_task(task_path)
    if task["name"] != benchmark_name:
        raise ValueError(f"task name {task['name']!r} does not match raw directory {benchmark_name!r}")

    raw_dir = RAW_DIR / run_name / benchmark_name
    native_results_path = raw_dir / "results.json"
    effective_config_path = raw_dir / "config.effective.yaml"
    results_path = raw_dir / "results.json.gz"
    execution_stats_path = raw_dir / "execution_stats.json.gz"
    manifest_path = raw_dir / "manifest.json"

    if not effective_config_path.is_file():
        raise FileNotFoundError(
            f"missing {effective_config_path}; DreamRetroer execution must record its effective planner configuration"
        )

    results, execution_stats = normalize_native_results(
        read_json(native_results_path),
        expected_target_ids=set(task["targets"]),
    )
    write_json_gz(results, results_path)
    write_execution_stats(execution_stats, execution_stats_path)

    statistics: dict[str, JSONValue] = {
        "solved_count": sum(bool(payload["succ"]) for payload in results.values()),
        "timed_targets": len(execution_stats["wall_time"]),
        "total_targets": len(results),
        "total_wall_time": sum(execution_stats["wall_time"].values()),
    }
    manifest = create_planner_manifest(
        "runtime/dreamretroer/1-gather-results.py",
        "dreamretroer",
        results_path,
        [task_path, native_results_path, effective_config_path, *additional_sources],
        DATA_DIR,
        parameters={
            "effective_config_path": str(effective_config_path.relative_to(DATA_DIR)),
            "native_results_filename": native_results_path.name,
        },
        statistics=statistics,
    )
    write_json(manifest, manifest_path)

    report = verify_planner_manifest(manifest_path, DATA_DIR)
    if not report["is_valid"]:
        messages = "; ".join(str(issue["message"]) for issue in report["issues"])
        raise ValueError(f"created an invalid DreamRetroer planner manifest: {messages}")
    return manifest_path


def resolve_additional_sources(values: Sequence[Path]) -> list[Path]:
    sources: list[Path] = []
    data_root = DATA_DIR.resolve()
    for value in values:
        path = value if value.is_absolute() else DATA_DIR / value
        if not path.is_file():
            raise FileNotFoundError(f"provenance source not found: {path}")
        path = path.resolve()
        try:
            path.relative_to(data_root)
        except ValueError as error:
            raise ValueError(f"provenance source must be inside the active RetroCast data root: {path}") from error
        sources.append(path)
    return sources
