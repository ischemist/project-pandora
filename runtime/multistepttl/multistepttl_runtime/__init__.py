"""Pandora-owned serialization for MultiStepTTL pickle outputs."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import retrocast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
TASKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"

logger = logging.getLogger("pandora.multistepttl")


class MultiStepTTLSerializationError(ValueError):
    """A MultiStepTTL pickle pair cannot be converted to the raw JSON contract."""


@dataclass(frozen=True)
class SerializedTask:
    results: dict[str, list[dict[str, Any]]]
    source_files: list[Path]
    discovered_targets: int
    failed_targets: list[str]
    missing_pickle_targets: list[str]

    @property
    def route_count(self) -> int:
        return sum(len(routes) for routes in self.results.values())


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def task_path(task_name: str) -> Path:
    return TASKS_DIR / f"{task_name}.json.gz"


def resolve_data_path(path: Path) -> Path:
    """Resolve a path and require its canonical location to stay inside the data root."""
    data_root = DATA_DIR.resolve()
    candidate = path if path.is_absolute() else DATA_DIR / path
    candidate = candidate.resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"path must be inside data root {data_root}: {candidate}") from error
    return candidate


def _advanced_scores(tree: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    scores: list[float] = []
    for route in tree["Route"]:
        product = 1.0
        try:
            for reaction_id in route:
                product *= predictions.loc[reaction_id, "Prob_Forward_Prediction_1"]
        except KeyError as error:
            raise MultiStepTTLSerializationError(f"reaction id {error} from route not found in predictions") from error
        scores.append(product)
    tree["fwd_conf_score"] = scores
    return tree


def serialize_target(tree: pd.DataFrame, predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert one target's MultiStepTTL dataframes to the documented route list."""
    if "index" in predictions.columns and predictions.index.name != "index":
        predictions = predictions.set_index("index")

    solved_routes = tree[tree["Solved"] == "Yes"].copy()
    if solved_routes.empty:
        return []
    solved_routes = _advanced_scores(solved_routes, predictions)

    output_routes: list[dict[str, Any]] = []
    for _, route_row in solved_routes.iterrows():
        reactions: list[dict[str, Any]] = []
        reaction_ids = route_row["Route"]
        for reaction_id in reaction_ids:
            try:
                prediction = predictions.loc[reaction_id]
                reactions.append(
                    {
                        "product": prediction["Target"],
                        "reactants": prediction["Retro"].split("."),
                    }
                )
            except KeyError as error:
                raise MultiStepTTLSerializationError(
                    f"reaction id {reaction_id} from route not found in predictions"
                ) from error

        output_routes.append(
            {
                "reactions": reactions,
                "metadata": {
                    "fwd_conf_score": route_row.get("fwd_conf_score"),
                    "score": route_row.get("Score"),
                    "steps": len(reaction_ids),
                },
            }
        )
    return output_routes


def discover_pickle_pair(target_dir: Path) -> tuple[Path, Path] | None:
    tree_pickles = sorted(target_dir.glob("*__tree.pkl"))
    prediction_pickles = sorted(target_dir.glob("*__prediction.pkl"))
    if not tree_pickles or not prediction_pickles:
        return None
    if len(tree_pickles) != 1 or len(prediction_pickles) != 1:
        raise MultiStepTTLSerializationError(
            f"expected exactly one tree and prediction pickle in {target_dir}, "
            f"found {len(tree_pickles)} tree and {len(prediction_pickles)} prediction files"
        )
    return tree_pickles[0], prediction_pickles[0]


def serialize_target_directory(target_dir: Path) -> tuple[list[dict[str, Any]], tuple[Path, Path]] | None:
    pair = discover_pickle_pair(target_dir)
    if pair is None:
        return None
    tree_pickle, prediction_pickle = pair
    try:
        tree = pd.read_pickle(tree_pickle)
        predictions = pd.read_pickle(prediction_pickle)
        if not isinstance(tree, pd.DataFrame) or not isinstance(predictions, pd.DataFrame):
            raise MultiStepTTLSerializationError(
                f"expected dataframe pickles in {target_dir}, got "
                f"{type(tree).__name__} and {type(predictions).__name__}"
            )
        return serialize_target(tree, predictions), pair
    except MultiStepTTLSerializationError:
        raise
    except Exception as error:
        raise MultiStepTTLSerializationError(f"failed to process pickles in {target_dir}: {error}") from error


def target_id_from_directory(directory_name: str) -> str:
    if directory_name.startswith("USPTO"):
        return directory_name.replace("_", "/")
    return directory_name


def serialize_task(task: Mapping[str, Any], input_dir: Path) -> SerializedTask:
    if not input_dir.is_dir():
        raise OSError(f"MultiStepTTL input directory does not exist: {input_dir}")

    target_ids = set(task["targets"])
    results: dict[str, list[dict[str, Any]]] = {}
    source_files: list[Path] = []
    failed_targets: list[str] = []
    missing_pickle_targets: list[str] = []
    discovered_targets = 0

    for target_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        target_id = target_id_from_directory(target_dir.name)
        if target_id not in target_ids:
            logger.warning("Ignoring MultiStepTTL directory for unknown target %s", target_id)
            continue
        discovered_targets += 1
        candidate_sources = sorted([*target_dir.glob("*__tree.pkl"), *target_dir.glob("*__prediction.pkl")])
        try:
            serialized = serialize_target_directory(target_dir)
            if serialized is None:
                logger.warning("No MultiStepTTL pickle pair found for target %s", target_id)
                source_files.extend(candidate_sources)
                missing_pickle_targets.append(target_id)
                continue
            routes, pair = serialized
            source_files.extend(pair)
            results[target_id] = routes
            logger.info("Serialized %d routes for target %s", len(routes), target_id)
        except MultiStepTTLSerializationError as error:
            source_files.extend(candidate_sources)
            failed_targets.append(target_id)
            logger.error("Could not serialize target %s: %s", target_id, error)

    return SerializedTask(
        results=results,
        source_files=source_files,
        discovered_targets=discovered_targets,
        failed_targets=failed_targets,
        missing_pickle_targets=missing_pickle_targets,
    )


def write_artifacts(
    *,
    serialized: SerializedTask,
    task: Mapping[str, Any],
    task_source: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    task_source = resolve_data_path(task_source)
    input_dir = resolve_data_path(input_dir)
    output_dir = resolve_data_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json.gz"
    manifest_path = output_dir / "manifest.json"
    retrocast.write_json_gz(serialized.results, results_path)

    manifest = retrocast.create_planner_manifest(
        "runtime/multistepttl/1-serialize-pickles.py",
        "multistepttl",
        results_path,
        [task_source, *serialized.source_files],
        DATA_DIR,
        parameters={
            "input_directory": str(input_dir.relative_to(DATA_DIR)),
            "serializer": "pandora-multistepttl-pickle-v1",
        },
        statistics={
            "discovered_targets": serialized.discovered_targets,
            "failed_targets": len(serialized.failed_targets),
            "missing_pickle_targets": len(serialized.missing_pickle_targets),
            "missing_target_directories": len(task["targets"]) - serialized.discovered_targets,
            "serialized_routes": serialized.route_count,
            "serialized_targets": len(serialized.results),
            "task_total_targets": len(task["targets"]),
        },
    )
    retrocast.write_json(manifest, manifest_path)
    report = retrocast.verify_planner_manifest(manifest_path, DATA_DIR)
    if not report["is_valid"]:
        raise RuntimeError(f"MultiStepTTL planner manifest failed verification: {report['issues']}")
    return manifest
