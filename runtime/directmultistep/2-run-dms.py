"""
Run DirectMultiStep retrosynthesis predictions on a RetroCast task.

Example:
    uv run --directory runtime/directmultistep 2-run-dms.py \
        --benchmark uspto-190 --model-name "explorer XL" --device cuda --use-fp16
"""

from __future__ import annotations

import argparse
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

from directmultistep.generate import (
    MODEL_CHECKPOINTS,
    create_beam_search,
    load_published_model,
    prepare_input_tensors,
)
from directmultistep.model import ModelFactory
from directmultistep.utils.dataset import RoutesProcessing
from directmultistep.utils.post_process import (
    canonicalize_paths,
    find_valid_paths,
    remove_repetitions_within_beam_result,
)
from directmultistep.utils.pre_process import canonicalize_smiles
from tqdm import tqdm

from utils import (
    DATA_DIR,
    DMS_DIR,
    RAW_DIR,
    ExecutionTimer,
    configure_logging,
    load_dms_task,
    logger,
    parse_route_strings,
    resolve_project_path,
    safe_run_name,
    write_effective_config,
    write_planner_artifacts,
)

MODEL_NAMES = tuple(MODEL_CHECKPOINTS)
AUTOREGRESSIVE_MODELS = {"explorer", "explorer XL"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def default_run_name(model_name: str, use_fp16: bool) -> str:
    slug = model_name.lower().replace("_", "-").replace(" ", "-")
    return f"dms-{slug}{'-fp16' if use_fp16 else ''}"


def load_model(
    model_name: str,
    checkpoint_path: Path,
    *,
    use_fp16: bool,
    device_name: str,
) -> Any:
    expected_checkpoint = DMS_DIR / "checkpoints" / MODEL_CHECKPOINTS[model_name][1]
    if checkpoint_path == expected_checkpoint:
        return load_published_model(model_name, expected_checkpoint.parent, use_fp16, force_device=device_name)

    preset_name = MODEL_CHECKPOINTS[model_name][0]
    device = ModelFactory.determine_device(device_name)
    model = ModelFactory.from_preset(preset_name, compile_model=False).create_model()
    model = ModelFactory.load_checkpoint(model, checkpoint_path, device)
    return model.half() if use_fp16 else model


def raw_paths_for_target(
    *,
    target_smiles: str,
    model_name: str,
    route_processor: RoutesProcessing,
    beam_search: Any,
    device: Any,
    use_fp16: bool,
    max_steps: int,
) -> list[str]:
    step_choices: tuple[int | None, ...] = (
        (None,) if model_name in AUTOREGRESSIVE_MODELS else tuple(range(1, max_steps + 1))
    )
    all_beam_results: list[list[tuple[str, float]]] = []

    for step in step_choices:
        encoder_input, steps_tensor, path_tensor = prepare_input_tensors(
            target_smiles,
            step,
            None,
            route_processor,
            route_processor.product_max_length,
            route_processor.sm_max_length,
            use_fp16,
        )
        beam_results = beam_search.decode(
            src_BC=encoder_input.to(device),
            steps_B1=steps_tensor.to(device) if steps_tensor is not None else None,
            path_start_BL=path_tensor.to(device),
            progress_bar=False,
        )
        all_beam_results.extend(beam_results)

    valid_paths = find_valid_paths(all_beam_results)
    flat_valid_paths = [path for batch in valid_paths for path in batch]
    canonical_paths = canonicalize_paths([flat_valid_paths])
    unique_paths = remove_repetitions_within_beam_result(canonical_paths)
    return [beam_result[0] for beam_result in unique_paths[0]]


def main() -> None:
    configure_logging()
    logging.getLogger("directmultistep").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run DirectMultiStep")
    parser.add_argument("--benchmark", required=True, help="Task name under 1-benchmarks/definitions")
    parser.add_argument("--model-name", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Override the published checkpoint for the selected preset",
    )
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-size", type=positive_int, default=50)
    parser.add_argument("--max-steps", type=positive_int, default=10)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--run-name", type=safe_run_name, help="Override the output model directory name")
    args = parser.parse_args()

    task, task_path = load_dms_task(args.benchmark)
    targets = list(task["targets"].values())
    if args.limit is not None:
        targets = targets[: args.limit]

    checkpoint_path = (
        resolve_project_path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else DMS_DIR / "checkpoints" / MODEL_CHECKPOINTS[args.model_name][1]
    )
    try:
        checkpoint_path.relative_to(DATA_DIR)
    except ValueError as error:
        raise ValueError(f"checkpoint path must be inside the active RetroCast data root: {DATA_DIR}") from error
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DirectMultiStep checkpoint not found at {checkpoint_path}. "
            "Run runtime/directmultistep/1-download-assets.sh first."
        )

    dictionary_path = DMS_DIR / "dms_dictionary.yaml"
    if not dictionary_path.is_file():
        raise FileNotFoundError(f"DirectMultiStep dictionary not found at {dictionary_path}")

    run_name = args.run_name or default_run_name(args.model_name, args.use_fp16)
    output_dir = RAW_DIR / run_name / task["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = ModelFactory.determine_device(args.device)
    effective_config = {
        "beam_size": args.beam_size,
        "checkpoint_path": str(checkpoint_path),
        "device_requested": args.device,
        "device_resolved": str(device),
        "dictionary_path": str(dictionary_path),
        "generation_mode": "autoregressive" if args.model_name in AUTOREGRESSIVE_MODELS else "step_conditioned",
        "limit": args.limit,
        "max_steps": None if args.model_name in AUTOREGRESSIVE_MODELS else args.max_steps,
        "model_name": args.model_name,
        "planner_version": version("directmultistep"),
        "use_fp16": args.use_fp16,
    }
    effective_config_path = write_effective_config(effective_config, output_dir)

    logger.info("Loading DirectMultiStep model %s from %s", args.model_name, checkpoint_path)
    route_processor = RoutesProcessing(metadata_path=dictionary_path)
    model = load_model(
        args.model_name,
        checkpoint_path,
        use_fp16=args.use_fp16,
        device_name=args.device,
    )
    beam_search = create_beam_search(model, args.beam_size, route_processor)

    results: dict[str, list[dict[str, Any]]] = {}
    timer = ExecutionTimer()
    for target in tqdm(targets, desc="Finding retrosynthetic paths"):
        target_id = target["id"]
        with timer.measure(target_id):
            try:
                target_smiles = canonicalize_smiles(target["smiles"])
                raw_paths = raw_paths_for_target(
                    target_smiles=target_smiles,
                    model_name=args.model_name,
                    route_processor=route_processor,
                    beam_search=beam_search,
                    device=device,
                    use_fp16=args.use_fp16,
                    max_steps=args.max_steps,
                )
                results[target_id] = parse_route_strings(raw_paths)
            except Exception:
                logger.exception("Failed to process target %s (%s)", target_id, target["smiles"])
                results[target_id] = []

    statistics = {
        "benchmark_total_targets": len(task["targets"]),
        "solved_count": sum(bool(routes) for routes in results.values()),
        "total_targets": len(results),
    }
    parameters = {
        "beam_size": args.beam_size,
        "effective_config_path": str(effective_config_path.relative_to(DATA_DIR)),
        "model_name": args.model_name,
        "planner_version": version("directmultistep"),
        "use_fp16": args.use_fp16,
    }
    if args.limit is not None:
        parameters["limit"] = args.limit

    write_planner_artifacts(
        results=results,
        execution_stats=timer.to_dict(),
        output_dir=output_dir,
        sources=[task_path, dictionary_path, checkpoint_path, effective_config_path],
        parameters=parameters,
        statistics=statistics,
        action="runtime/directmultistep/2-run-dms.py",
    )
    logger.info("Completed %d targets; %d produced at least one route", len(results), statistics["solved_count"])


if __name__ == "__main__":
    main()
