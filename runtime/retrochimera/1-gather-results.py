"""Gather per-target RetroChimera JSON files into its raw results artifact."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from gather import write_gathered_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Schema-v2 task name.")
    parser.add_argument(
        "--eval-dir",
        required=True,
        type=project_path,
        help="Directory containing target-id.json planner outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=project_path,
        help="Combined raw-output directory; defaults to 2-raw/retrochimera/<benchmark>.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    data_root = DEFAULT_DATA_ROOT.resolve()
    task_path = data_root / "1-benchmarks" / "definitions" / f"{args.benchmark}.json.gz"
    output_dir = args.output_dir or data_root / "2-raw" / "retrochimera" / args.benchmark
    write_gathered_results(
        task_path=task_path,
        eval_dir=args.eval_dir,
        output_dir=output_dir,
        data_root=data_root,
    )


if __name__ == "__main__":
    main()
