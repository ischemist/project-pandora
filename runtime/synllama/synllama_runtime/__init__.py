"""Pandora-owned conversion of SynLlama CSV output."""

from __future__ import annotations

import csv
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import retrocast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
TASKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"

logger = logging.getLogger("pandora.synllama")


class SynLlamaConversionError(ValueError):
    """A SynLlama CSV cannot be converted to its raw adapter format."""


@dataclass(frozen=True)
class ConversionResult:
    routes_by_target: dict[str, list[dict[str, str]]]
    reported_wall_times: dict[str, float]
    total_reported_time_seconds: float
    total_rows: int
    skipped_rows: int
    invalid_time_rows: int
    conflicting_time_targets: list[str]

    @property
    def route_count(self) -> int:
        return sum(len(routes) for routes in self.routes_by_target.values())


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def task_path(task_name: str) -> Path:
    return TASKS_DIR / f"{task_name}.json.gz"


def convert_csv(input_path: Path, task: Mapping[str, Any]) -> ConversionResult:
    task_target_ids = set(task["targets"])
    routes_by_target: dict[str, list[dict[str, str]]] = {}
    reported_wall_times: dict[str, float] = {}
    conflicting_time_targets: set[str] = set()
    total_reported_time_seconds = 0.0
    total_rows = 0
    skipped_rows = 0
    invalid_time_rows = 0

    try:
        with input_path.open("r", encoding="utf-8", newline="") as fileobj:
            reader = csv.DictReader(fileobj)
            columns = set(reader.fieldnames or [])
            missing_columns = {"Structure ID", "synthesis"} - columns
            if missing_columns:
                raise SynLlamaConversionError(f"SynLlama CSV is missing required columns: {sorted(missing_columns)}")

            for row in reader:
                total_rows += 1
                target_id = row.get("Structure ID")
                synthesis_string = row.get("synthesis")
                if not target_id or not synthesis_string:
                    skipped_rows += 1
                    continue
                if target_id not in task_target_ids:
                    raise SynLlamaConversionError(
                        f"SynLlama CSV target {target_id!r} is not present in task {task['name']!r}"
                    )

                routes_by_target.setdefault(target_id, []).append({"synthesis_string": synthesis_string})
                time_value = row.get("time, s")
                if not time_value:
                    continue
                try:
                    parsed_time = float(time_value)
                except (TypeError, ValueError):
                    invalid_time_rows += 1
                    logger.warning(
                        "Could not parse reported time %r for target %s",
                        time_value,
                        target_id,
                    )
                    continue
                if not math.isfinite(parsed_time) or parsed_time < 0:
                    invalid_time_rows += 1
                    logger.warning("Ignoring negative reported time %r for target %s", time_value, target_id)
                    continue

                total_reported_time_seconds += parsed_time
                previous_time = reported_wall_times.get(target_id)
                if previous_time is None and target_id not in conflicting_time_targets:
                    reported_wall_times[target_id] = parsed_time
                elif previous_time != parsed_time:
                    reported_wall_times.pop(target_id, None)
                    conflicting_time_targets.add(target_id)
                    logger.warning("Conflicting reported times for target %s; omitting target timing", target_id)
    except OSError:
        raise
    except csv.Error as error:
        raise SynLlamaConversionError(f"could not parse SynLlama CSV {input_path}: {error}") from error

    return ConversionResult(
        routes_by_target=routes_by_target,
        reported_wall_times=reported_wall_times,
        total_reported_time_seconds=total_reported_time_seconds,
        total_rows=total_rows,
        skipped_rows=skipped_rows,
        invalid_time_rows=invalid_time_rows,
        conflicting_time_targets=sorted(conflicting_time_targets),
    )


def write_artifacts(
    *,
    converted: ConversionResult,
    task: Mapping[str, Any],
    task_source: Path,
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json.gz"
    execution_stats_path = output_dir / "execution_stats.json.gz"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    summary = {
        "solved_count": len(converted.routes_by_target),
        "time_elapsed": converted.total_reported_time_seconds,
    }
    retrocast.write_json_gz(converted.routes_by_target, results_path)
    retrocast.write_execution_stats(
        {"wall_time": converted.reported_wall_times, "cpu_time": {}},
        execution_stats_path,
    )
    retrocast.write_json(summary, summary_path)

    manifest = retrocast.create_planner_manifest(
        "runtime/synllama/1-convert-to-json.py",
        "synllama",
        results_path,
        [task_source, input_path],
        DATA_DIR,
        parameters={
            "execution_stats_source": "csv:time, s",
            "input_csv": str(input_path.relative_to(DATA_DIR)),
            "serializer": "pandora-synllama-csv-v1",
        },
        statistics={
            "conflicting_time_targets": len(converted.conflicting_time_targets),
            "invalid_time_rows": converted.invalid_time_rows,
            "route_count": converted.route_count,
            "serialized_targets": len(converted.routes_by_target),
            "skipped_rows": converted.skipped_rows,
            "task_total_targets": len(task["targets"]),
            "total_rows": converted.total_rows,
        },
        summary=summary,
    )
    retrocast.write_json(manifest, manifest_path)
    report = retrocast.verify_planner_manifest(manifest_path, DATA_DIR)
    if not report["is_valid"]:
        raise RuntimeError(f"SynLlama planner manifest failed verification: {report['issues']}")
    return manifest
