"""Run the original Retro* planner on a schema-v2 task."""

from __future__ import annotations

from typing import Any

import retrocast
from retro_star.api import RSPlanner
from tqdm import tqdm
from utils import (
    ASSET_DIR,
    RAW_DIR,
    ExecutionTimer,
    configure_logging,
    convert_numpy,
    create_parser,
    load_task_and_stock,
    logger,
    save_results,
    write_retrostar_stock,
)

PLANNER_REVISION = "f392fdc920d12d325ef61b8622a8795ff06fa49f"


def main() -> None:
    configure_logging()
    args = create_parser().parse_args()
    iterations = 500 if args.effort == "high" else 100
    task, task_path, stock_path, stock_name = load_task_and_stock(args.benchmark)

    folder_name = "retro-star" if args.effort == "normal" else f"retro-star-{args.effort}"
    save_dir = RAW_DIR / folder_name / task["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    effective_stock_path = save_dir / "stock.effective.csv"
    effective_config_path = save_dir / "config.effective.json"
    write_retrostar_stock(stock_path, effective_stock_path)

    parameters: dict[str, Any] = {
        "planner_revision": PLANNER_REVISION,
        "stock_name": stock_name,
        "effort": args.effort,
        "iterations": iterations,
        "expansion_topk": 50,
        "use_value_fn": True,
        "effective_stock_path": str(effective_stock_path.relative_to(RAW_DIR.parent)),
        "effective_config_path": str(effective_config_path.relative_to(RAW_DIR.parent)),
    }
    if args.limit is not None:
        parameters["limit"] = args.limit
    retrocast.write_json(parameters, effective_config_path)

    logger.info("stock: %s", stock_name)
    logger.info("effort: %s (iterations=%d)", args.effort, iterations)
    planner = RSPlanner(
        gpu=-1,
        use_value_fn=True,
        iterations=iterations,
        expansion_topk=50,
        starting_molecules=str(effective_stock_path),
        mlp_templates=str(ASSET_DIR / "one_step_model" / "template_rules_1.dat"),
        mlp_model_dump=str(ASSET_DIR / "one_step_model" / "saved_rollout_state_1_2048.ckpt"),
        save_folder=str(ASSET_DIR / "saved_models"),
    )

    targets = list(task["targets"].values())
    if args.limit is not None:
        targets = targets[: args.limit]

    results: dict[str, dict[str, Any]] = {}
    timer = ExecutionTimer()
    for target in tqdm(targets, desc="Finding retrosynthetic paths"):
        target_id = target["id"]
        with timer.measure(target_id):
            try:
                result = planner.plan(target["smiles"])
                results[target_id] = convert_numpy(result) if result and result.get("succ") else {}
            except Exception:
                logger.exception("Failed to process target %s (%s)", target_id, target["smiles"])
                results[target_id] = {}

    save_results(
        results=results,
        execution_stats=timer.to_dict(),
        save_dir=save_dir,
        task_path=task_path,
        stock_path=stock_path,
        effective_stock_path=effective_stock_path,
        effective_config_path=effective_config_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    main()
