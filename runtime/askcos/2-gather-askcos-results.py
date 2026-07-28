"""Gather individual ASKCOS JSON responses into the documented raw output."""

from __future__ import annotations

import argparse
from pathlib import Path

from askcos_runtime import (
    DATA_DIR,
    RAW_DIR,
    configure_logging,
    gather_results,
    load_task,
    logger,
    resolve_path_within_root,
    write_gathered_artifacts,
)


def path_within_data_root(value: str) -> Path:
    try:
        return resolve_path_within_root(value, DATA_DIR)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--benchmark", dest="task_name", required=True)
    parser.add_argument(
        "--eval-dir",
        required=True,
        type=path_within_data_root,
        help="Directory of individual .json responses, relative to RETROCAST_DATA_DIR by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=path_within_data_root,
        help="Raw output directory. Defaults to 2-raw/askcos/<task>.",
    )
    args = parser.parse_args()

    task, task_source, _stock_name = load_task(args.task_name)
    output_dir = args.output_dir or RAW_DIR / "askcos" / task["name"]
    results, source_files, missing_targets = gather_results(task, args.eval_dir)
    write_gathered_artifacts(
        results=results,
        source_files=source_files,
        missing_targets=missing_targets,
        task=task,
        task_source=task_source,
        output_dir=output_dir,
        eval_dir=args.eval_dir,
    )
    logger.info("Gathered %d/%d ASKCOS results into %s", len(results), len(task["targets"]), output_dir)
    if missing_targets:
        logger.warning("Missing or invalid ASKCOS results for %d targets", len(missing_targets))


if __name__ == "__main__":
    configure_logging()
    main()
