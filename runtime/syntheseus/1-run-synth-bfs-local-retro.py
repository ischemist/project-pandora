"""Run Syntheseus breadth-first search with LocalRetro over a RetroCast task."""

from __future__ import annotations

from typing import Any

from serialization import SyntheseusSerializationError, serialize_route
from syntheseus import Molecule
from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
from syntheseus.search.mol_inventory import SmilesListInventory
from template_aware_local_retro import TemplateAwareLocalRetroModel
from tqdm import tqdm
from utils import (
    DATA_DIR,
    RAW_DIR,
    ExecutionTimer,
    configure_script_logging,
    create_parser,
    load_task_and_stock,
    logger,
    publish_results,
    write_effective_config,
)

PLANNER_VERSION = "0.7.2"


if __name__ == "__main__":
    configure_script_logging()
    args = create_parser("Run Syntheseus breadth-first search with LocalRetro").parse_args()
    iterations = 500 if args.effort == "high" else 100
    task, building_blocks, task_path, stock_path, stock_name = load_task_and_stock(args.benchmark)
    folder_name = "syntheseus-bfs-local-retro"
    if args.effort != "normal":
        folder_name = f"{folder_name}-{args.effort}"
    save_dir = RAW_DIR / folder_name / task["name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("stock: %s", stock_name)
    logger.info("effort: %s", args.effort)
    parameters = {
        "planner_version": PLANNER_VERSION,
        "reaction_model": "local-retro",
        "search_strategy": "breadth_first",
        "effort": args.effort,
        "iteration_limit": iterations,
        "reaction_model_call_limit": 100,
        "time_limit_seconds": 60.0,
    }
    if args.limit is not None:
        parameters["limit"] = args.limit
    effective_config_path = write_effective_config({**parameters, "stock": stock_name}, save_dir)
    parameters["effective_config_path"] = str(effective_config_path.relative_to(DATA_DIR))

    inventory = SmilesListInventory(smiles_list=building_blocks)
    model = TemplateAwareLocalRetroModel(use_cache=True, default_num_results=10)
    targets = list(task["targets"].values())
    if args.limit is not None:
        targets = targets[: args.limit]

    results: dict[str, list[dict[str, Any]]] = {}
    timer = ExecutionTimer()
    for target in tqdm(targets, desc="Finding retrosynthetic paths"):
        target_id = target["id"]
        target_smiles = target["smiles"]
        with timer.measure(target_id):
            try:
                search = AndOr_BreadthFirstSearch(
                    reaction_model=model,
                    mol_inventory=inventory,
                    limit_iterations=iterations,
                    limit_reaction_model_calls=100,
                    time_limit_s=60.0,
                )
                search.reset()
                graph, _ = search.run_from_mol(Molecule(target_smiles))
                serialized_routes = []
                for route in iter_routes_time_order(graph, max_routes=10):
                    try:
                        serialized_routes.append(serialize_route(graph, route, target_smiles))
                    except SyntheseusSerializationError as error:
                        logger.warning("Could not serialize route for target %s: %s", target_id, error)
                results[target_id] = serialized_routes
            except Exception as error:
                logger.error("Failed to process target %s (%s): %s", target_id, target_smiles, error, exc_info=True)
                results[target_id] = []

    publish_results(
        results=results,
        runtime=timer.to_dict(),
        save_dir=save_dir,
        task_path=task_path,
        stock_path=stock_path,
        effective_config_path=effective_config_path,
        action="runtime/syntheseus/1-run-synth-bfs-local-retro.py",
        parameters=parameters,
        task=task,
    )
