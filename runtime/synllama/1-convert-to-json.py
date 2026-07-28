"""Convert externally produced SynLlama CSV routes to raw adapter JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

import retrocast

from synllama_runtime import (
    DATA_DIR,
    RAW_DIR,
    configure_logging,
    convert_csv,
    logger,
    task_path,
    write_artifacts,
)


def path_within_data_root(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else DATA_DIR / path
    try:
        path.relative_to(DATA_DIR)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"path must be inside {DATA_DIR}") from error
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="RetroCast task name.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=path_within_data_root,
        help="SynLlama results CSV, relative to RETROCAST_DATA_DIR by default.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output_dir",
        type=path_within_data_root,
        help="Raw output directory. Defaults to 2-raw/synllama/<task>.",
    )
    args = parser.parse_args()

    source_path = task_path(args.task)
    task = retrocast.load_task(source_path)
    output_dir = args.output_dir or RAW_DIR / "synllama" / task["name"]
    converted = convert_csv(args.input, task)
    write_artifacts(
        converted=converted,
        task=task,
        task_source=source_path,
        input_path=args.input,
        output_dir=output_dir,
    )

    logger.info(
        "Converted %d routes for %d/%d targets into %s",
        converted.route_count,
        len(converted.routes_by_target),
        len(task["targets"]),
        output_dir,
    )


if __name__ == "__main__":
    configure_logging()
    main()
