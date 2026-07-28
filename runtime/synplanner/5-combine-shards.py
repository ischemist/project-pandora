"""Combine verified SynPlanner shards into one complete raw result."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from retrocast import load_task, read_json, verify_planner_manifest
from utils import (
    BENCHMARKS_DIR,
    DATA_DIR,
    RAW_DIR,
    STOCKS_DIR,
    SYNPLANNER_DIR,
    benchmark_stock_name,
    configure_script_logging,
    logger,
    positive_int,
    save_synplanner_results,
)

ExecutionStats = dict[str, dict[str, float]]


def require_mapping(value: Any, path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def gather_shards(
    *,
    base_dir: Path,
    shard_count: int,
    ordered_target_ids: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], ExecutionStats, dict[str, Any], list[Path], Path]:
    results: dict[str, list[dict[str, Any]]] = {}
    wall_time: dict[str, float] = {}
    cpu_time: dict[str, float] = {}
    sources: list[Path] = []
    first_parameters: dict[str, Any] | None = None
    first_config_path: Path | None = None

    for shard_index in range(shard_count):
        part_dir = base_dir / "shards" / f"part-{shard_index:02d}-of-{shard_count:02d}"
        manifest_path = part_dir / "manifest.json"
        results_path = part_dir / "results.json.gz"
        stats_path = part_dir / "execution_stats.json.gz"
        config_path = part_dir / "config.effective.yaml"

        verification = verify_planner_manifest(manifest_path, DATA_DIR)
        if not verification["is_valid"]:
            raise ValueError(f"Invalid shard manifest {manifest_path}: {verification['issues']}")

        manifest = require_mapping(read_json(manifest_path), manifest_path)
        parameters = require_mapping(manifest.get("parameters"), manifest_path)
        if (
            parameters.get("shard_count") != shard_count
            or parameters.get("shard_index") != shard_index
            or parameters.get("shard_strategy") != "round_robin"
        ):
            raise ValueError(f"Unexpected shard parameters in {manifest_path}")

        comparable_parameters = {
            key: value
            for key, value in parameters.items()
            if key not in {"effective_config_path", "shard_index"}
        }
        if first_parameters is None:
            first_parameters = comparable_parameters
            first_config_path = config_path
        elif comparable_parameters != first_parameters:
            raise ValueError(f"Planner parameter mismatch in {manifest_path}")

        part_results = require_mapping(read_json(results_path), results_path)
        part_stats = require_mapping(read_json(stats_path), stats_path)
        part_wall = require_mapping(part_stats.get("wall_time"), stats_path)
        part_cpu = require_mapping(part_stats.get("cpu_time"), stats_path)
        expected_ids = set(ordered_target_ids[shard_index::shard_count])
        actual_ids = set(part_results)
        if actual_ids != expected_ids:
            raise ValueError(
                f"Shard {shard_index} target coverage mismatch: "
                f"missing={sorted(expected_ids - actual_ids)}, unexpected={sorted(actual_ids - expected_ids)}"
            )
        if set(part_wall) != expected_ids or set(part_cpu) != expected_ids:
            raise ValueError(f"Shard {shard_index} execution statistics do not match its targets")

        duplicate_ids = set(results).intersection(part_results)
        if duplicate_ids:
            raise ValueError(f"Duplicate target IDs across shards: {sorted(duplicate_ids)}")

        for target_id, routes in part_results.items():
            if not isinstance(routes, list):
                raise ValueError(f"{results_path} result for {target_id} must be a list")
            results[str(target_id)] = routes
        wall_time.update({str(key): float(value) for key, value in part_wall.items()})
        cpu_time.update({str(key): float(value) for key, value in part_cpu.items()})
        sources.extend([manifest_path, stats_path])

    expected_ids = set(ordered_target_ids)
    if set(results) != expected_ids:
        raise ValueError("Combined shard output does not cover the full benchmark")
    assert first_parameters is not None and first_config_path is not None
    ordered_results = {target_id: results[target_id] for target_id in ordered_target_ids}
    ordered_runtime = {
        "wall_time": {target_id: wall_time[target_id] for target_id in ordered_target_ids},
        "cpu_time": {target_id: cpu_time[target_id] for target_id in ordered_target_ids},
    }
    return ordered_results, ordered_runtime, first_parameters, sources, first_config_path


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--shard-count", required=True, type=positive_int)
    args = parser.parse_args()

    bench_path = BENCHMARKS_DIR / f"{args.benchmark}.json.gz"
    benchmark = load_task(bench_path)
    stock_path = STOCKS_DIR / f"{benchmark_stock_name(benchmark)}.csv.gz"
    base_dir = RAW_DIR / args.run_name / benchmark["name"]
    results, runtime, parameters, sources, source_config_path = gather_shards(
        base_dir=base_dir,
        shard_count=args.shard_count,
        ordered_target_ids=list(benchmark["targets"]),
    )

    effective_config_path = base_dir / "config.effective.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config_path, effective_config_path)

    planner_version = str(parameters["planner_version"])
    parameters = {
        key: value
        for key, value in parameters.items()
        if key
        not in {
            "config_template_path",
            "effective_config_path",
            "planner_version",
            "shard_count",
            "shard_strategy",
        }
    }
    parameters.update(
        {
            "execution_mode": "sharded",
            "shard_count": args.shard_count,
            "shard_strategy": "round_robin",
        }
    )
    evaluation_kind = parameters.get("evaluation_kind")
    if evaluation_kind == "value_network":
        config_template_path = SYNPLANNER_DIR / "mcts-val-config.yaml"
    elif evaluation_kind == "rollout":
        config_template_path = SYNPLANNER_DIR / "mcts-rollout-config.yaml"
    else:
        raise ValueError(f"Unsupported SynPlanner evaluation kind: {evaluation_kind!r}")

    save_synplanner_results(
        results=results,
        runtime=runtime,
        save_dir=base_dir,
        bench_path=bench_path,
        stock_path=stock_path,
        effective_config_path=effective_config_path,
        config_template_path=config_template_path,
        script_name="runtime/synplanner/5-combine-shards.py",
        benchmark=benchmark,
        planner_version=planner_version,
        parameters=parameters,
        additional_sources=sources,
    )
    logger.info("Combined %d shards into %s", args.shard_count, base_dir)


if __name__ == "__main__":
    main()
