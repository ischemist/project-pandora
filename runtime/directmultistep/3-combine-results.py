"""
Combine partitioned DirectMultiStep raw runs into one ingestible task artifact.

Example:
    uv run --directory runtime/directmultistep 3-combine-results.py \
        --run-name dms-wide-fp16 --benchmark uspto-190 --parts pt1 pt2 pt3 pt4
"""

from __future__ import annotations

import argparse

from utils import (
    DATA_DIR,
    RAW_DIR,
    configure_logging,
    gather_dms_parts,
    load_dms_task,
    logger,
    safe_run_name,
    write_effective_config,
    write_planner_artifacts,
)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Combine partitioned DirectMultiStep raw runs")
    parser.add_argument("--run-name", required=True, type=safe_run_name)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--parts", nargs="+", required=True)
    args = parser.parse_args()

    task, task_path = load_dms_task(args.benchmark)
    part_names = [f"{args.benchmark}-{part}" for part in args.parts]
    part_dirs = [RAW_DIR / args.run_name / part_name for part_name in part_names]
    output_dir = RAW_DIR / args.run_name / task["name"]

    results, execution_stats, part_sources = gather_dms_parts(
        part_dirs=part_dirs,
        expected_target_ids=set(task["targets"]),
    )
    effective_config_path = write_effective_config(
        {
            "collision_policy": "reject",
            "coverage_policy": "exact",
            "parts": part_names,
            "source_run_name": args.run_name,
        },
        output_dir,
    )
    statistics = {
        "parts_combined": len(part_dirs),
        "solved_count": sum(bool(routes) for routes in results.values()),
        "total_targets": len(results),
    }
    write_planner_artifacts(
        results=results,
        execution_stats=execution_stats,
        output_dir=output_dir,
        sources=[task_path, effective_config_path, *part_sources],
        parameters={
            "effective_config_path": str(effective_config_path.relative_to(DATA_DIR)),
            "parts": args.parts,
            "source_run_name": args.run_name,
        },
        statistics=statistics,
        action="runtime/directmultistep/3-combine-results.py",
    )
    logger.info("Combined %d parts into %s", len(part_dirs), output_dir)


if __name__ == "__main__":
    main()
