"""Pandora-owned RetroChimera result gathering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import retrocast
from tqdm import tqdm

logger = logging.getLogger("pandora.retrochimera")


@dataclass(frozen=True)
class GatherReport:
    results: dict[str, Any]
    matched_files: tuple[Path, ...]
    missing_target_ids: tuple[str, ...]
    malformed_target_ids: tuple[str, ...]


def normalize_target_id_for_filename(target_id: str) -> str:
    """Apply RetroChimera's historical filename normalization."""
    normalized = target_id.replace(" ", "_").replace("/", "_")
    normalized = normalized.replace("(", "_").replace(")", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def find_result_file(target_id: str, eval_dir: Path) -> Path | None:
    """Find the per-target JSON emitted by RetroChimera."""
    exact_path = eval_dir / f"{target_id}.json"
    if exact_path.is_file():
        return exact_path

    normalized_path = eval_dir / f"{normalize_target_id_for_filename(target_id)}.json"
    if normalized_path.is_file():
        return normalized_path

    if "(" in target_id and ")" in target_id:
        main_name = target_id.split(")")[-1].strip("-").strip()
        matches = sorted(eval_dir.glob(f"{main_name}*.json"))
        if matches:
            return matches[0]

    return None


def gather_results(task: dict[str, Any], eval_dir: Path) -> GatherReport:
    """Combine per-target files without changing their raw payloads."""
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"RetroChimera evaluation directory does not exist: {eval_dir}")

    results: dict[str, Any] = {}
    matched_files: list[Path] = []
    missing_target_ids: list[str] = []
    malformed_target_ids: list[str] = []

    targets = task["targets"].values()
    for target in tqdm(targets, desc="Combining RetroChimera results"):
        target_id = target["id"]
        result_path = find_result_file(target_id, eval_dir)
        if result_path is None:
            missing_target_ids.append(target_id)
            continue

        matched_files.append(result_path)
        try:
            results[target_id] = retrocast.read_json(result_path)
        except (OSError, ValueError) as exc:
            logger.error("Failed to read %s: %s", result_path, exc)
            malformed_target_ids.append(target_id)

    return GatherReport(
        results=results,
        matched_files=tuple(matched_files),
        missing_target_ids=tuple(missing_target_ids),
        malformed_target_ids=tuple(malformed_target_ids),
    )


def write_gathered_results(
    *,
    task_path: Path,
    eval_dir: Path,
    output_dir: Path,
    data_root: Path,
) -> GatherReport:
    """Gather raw output, write it, and publish a verified planner manifest."""
    task_path = task_path.resolve()
    eval_dir = eval_dir.resolve()
    output_dir = output_dir.resolve()
    data_root = data_root.resolve()

    for label, path in (("task", task_path), ("evaluation directory", eval_dir), ("output directory", output_dir)):
        try:
            path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"RetroChimera {label} must be inside data root {data_root}: {path}") from exc

    task = retrocast.load_task(task_path)
    report = gather_results(task, eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json.gz"
    manifest_path = output_dir / "manifest.json"

    retrocast.write_json_gz(report.results, results_path)
    statistics = {
        "total_targets": len(task["targets"]),
        "gathered_targets": len(report.results),
        "missing_targets": len(report.missing_target_ids),
        "malformed_targets": len(report.malformed_target_ids),
    }
    parameters = {
        "source_layout": "target-id.json",
        "evaluation_directory": str(eval_dir.relative_to(data_root)),
        "missing_target_ids": list(report.missing_target_ids),
        "malformed_target_ids": list(report.malformed_target_ids),
    }
    manifest = retrocast.create_planner_manifest(
        "runtime/retrochimera/1-gather-results.py",
        "retrochimera",
        results_path,
        [task_path, *report.matched_files],
        data_root,
        parameters=parameters,
        statistics=statistics,
    )
    retrocast.write_json(manifest, manifest_path)
    verification = retrocast.verify_planner_manifest(manifest_path, data_root, deep=True)
    if not verification["is_valid"]:
        raise RuntimeError(f"RetroChimera planner manifest verification failed: {verification['issues']}")

    logger.info("Gathered %d of %d RetroChimera results", len(report.results), len(task["targets"]))
    if report.missing_target_ids:
        logger.warning("Missing files for %d targets: %s", len(report.missing_target_ids), report.missing_target_ids)
    if report.malformed_target_ids:
        logger.warning(
            "Malformed files for %d targets: %s",
            len(report.malformed_target_ids),
            report.malformed_target_ids,
        )
    return report
