"""Serialize MultiStepTTL target pickle directories to RetroCast raw JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

import retrocast

from multistepttl_runtime import (
    DATA_DIR,
    RAW_DIR,
    configure_logging,
    logger,
    resolve_data_path,
    serialize_task,
    task_path,
    write_artifacts,
)


def path_within_data_root(value: str) -> Path:
    try:
        return resolve_data_path(Path(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"path must be inside {DATA_DIR}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--ds-name", dest="task_name", required=True)
    parser.add_argument(
        "--input-dir",
        type=path_within_data_root,
        help="Directory containing one pickle subdirectory per target.",
    )
    parser.add_argument(
        "--output-dir",
        type=path_within_data_root,
        help="Directory for results.json.gz and manifest.json.",
    )
    args = parser.parse_args()

    try:
        source_path = resolve_data_path(task_path(args.task_name))
    except ValueError as error:
        parser.error(str(error))
    task = retrocast.load_task(source_path)
    output_dir = resolve_data_path(args.output_dir or RAW_DIR / "multistepttl" / task["name"])
    input_dir = resolve_data_path(args.input_dir or output_dir)

    serialized = serialize_task(task, input_dir)
    write_artifacts(
        serialized=serialized,
        task=task,
        task_source=source_path,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    logger.info(
        "Serialized %d routes for %d/%d targets into %s",
        serialized.route_count,
        len(serialized.results),
        len(task["targets"]),
        output_dir,
    )
    if serialized.failed_targets:
        logger.warning("Failed targets: %s", ", ".join(serialized.failed_targets))
    if serialized.missing_pickle_targets:
        logger.warning("Targets missing pickle pairs: %s", ", ".join(serialized.missing_pickle_targets))


if __name__ == "__main__":
    configure_logging()
    main()
