"""Combine verified AiZynthFinder shards into one complete raw result."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from retrocast import read_json, verify_planner_manifest
from utils import (
    AIZYNTHFINDER_DIR,
    DATA_DIR,
    RAW_DIR,
    TargetResult,
    configure_script_logging,
    load_aizynthfinder_benchmark,
    logger,
    positive_int,
    save_aizynthfinder_results,
)

ExecutionStats = dict[str, dict[str, float]]


def require_mapping(value: Any, path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def normalize_target_result(value: Any, path: Path, target_id: str) -> TargetResult:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list) and all(isinstance(route, Mapping) for route in value):
        return [dict(route) for route in value]
    raise ValueError(f"{path} result for {target_id} must be an object or a list of objects")


def gather_shards(
    *,
    base_dir: Path,
    shard_count: int,
    ordered_target_ids: list[str],
) -> tuple[dict[str, TargetResult], ExecutionStats, dict[str, Any], list[Path], Path]:
    results: dict[str, TargetResult] = {}
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

        results.update(
            {
                str(key): normalize_target_result(value, results_path, str(key))
                for key, value in part_results.items()
            }
        )
        wall_time.update({str(key): float(value) for key, value in part_wall.items()})
        cpu_time.update({str(key): float(value) for key, value in part_cpu.items()})
        sources.extend([manifest_path, stats_path])

    expected_ids = set(ordered_target_ids)
    if set(results) != expected_ids:
        raise ValueError("Combined shard output does not cover the full benchmark")
    assert first_parameters is not None and first_config_path is not None
    return results, {"wall_time": wall_time, "cpu_time": cpu_time}, first_parameters, sources, first_config_path


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--shard-count", required=True, type=positive_int)
    args = parser.parse_args()

    benchmark, bench_path = load_aizynthfinder_benchmark(args.benchmark)
    base_dir = RAW_DIR / args.run_name / benchmark["name"]
    results, runtime, parameters, sources, source_config_path = gather_shards(
        base_dir=base_dir,
        shard_count=args.shard_count,
        ordered_target_ids=list(benchmark["targets"]),
    )

    effective_config_path = base_dir / "config.effective.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config_path, effective_config_path)

    parameters = {
        key: value
        for key, value in parameters.items()
        if key not in {"config_template_path", "shard_count", "shard_strategy"}
    }
    parameters.update(
        {
            "execution_mode": "sharded",
            "shard_count": args.shard_count,
            "shard_strategy": "round_robin",
        }
    )
    strategy = parameters.get("search_strategy")
    if strategy == "mcts":
        config_template_path = AIZYNTHFINDER_DIR / "config-mcts.yaml"
    elif strategy == "retro_star":
        config_template_path = AIZYNTHFINDER_DIR / "config-retrostar.yaml"
    else:
        raise ValueError(f"Unsupported AiZynthFinder search strategy: {strategy!r}")

    save_aizynthfinder_results(
        results=results,
        runtime=runtime,
        save_dir=base_dir,
        bench_path=bench_path,
        effective_config_path=effective_config_path,
        config_template_path=config_template_path,
        script_name="runtime/aizynthfinder/4-combine-shards.py",
        benchmark=benchmark,
        parameters=parameters,
        solved_count=sum(bool(result) for result in results.values()),
        additional_sources=sources,
    )
    logger.info("Combined %d shards into %s", args.shard_count, base_dir)


if __name__ == "__main__":
    main()
