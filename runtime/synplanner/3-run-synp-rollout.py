"""
Run Synplanner MCTS retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using Synplanner's MCTS algorithm
and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --directory runtime/synplanner 3-run-synp-rollout.py --benchmark uspto-190
    uv run --directory runtime/synplanner 3-run-synp-rollout.py \
        --benchmark random-n5-2-seed=20251030 --iteration-limit 500

The benchmark definition should be located at: data/retrocast/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/retrocast/2-raw/synplanner-{version}-mcts-rollout-iter{iteration_limit}/{benchmark_name}/
"""

from retrocast.utils.logging import configure_script_logging, logger
from synplan.mcts.tree import TreeConfig
from synplan.utils.config import RolloutEvaluationConfig
from synplan.utils.loading import load_evaluation_function, load_reaction_rules
from utils import (
    RAW_DIR,
    SYNPLANNER_DIR,
    create_benchmark_parser,
    load_benchmark_and_stock,
    load_policy_from_config,
    load_synplanner_config,
    run_synplanner_predictions,
    save_synplanner_results,
    write_effective_config,
)

configure_script_logging()
# Synplanner version - update when upgrading the library
PLANNER_VERSION = "1.3.2"

if __name__ == "__main__":
    parser = create_benchmark_parser("Run Synplanner MCTS with rollout evaluation")
    parser.add_argument(
        "--iteration-limit",
        type=int,
        default=100,
        choices=[100, 500],
        help="Maximum tree search iterations.",
    )
    args = parser.parse_args()

    benchmark, building_blocks, bench_path, stock_path = load_benchmark_and_stock(args.benchmark)

    folder_name = f"synplanner-{PLANNER_VERSION}-mcts-rollout-iter{args.iteration_limit}"
    save_dir = RAW_DIR / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"iteration limit: {args.iteration_limit}")

    config_path = SYNPLANNER_DIR / "mcts-rollout-config.yaml"
    config = load_synplanner_config(config_path)
    resources = config["resources"]

    config["tree"]["max_iterations"] = args.iteration_limit
    tree_config = TreeConfig.from_dict(config["tree"])
    tree_config.search_strategy = "expansion_first"
    tree_config.evaluation_agg = config["node_evaluation"].get("evaluation_agg", tree_config.evaluation_agg)
    config["tree"] = tree_config.to_dict()
    config["stock"] = {"name": benchmark.stock_name, "path": str(stock_path)}

    policy_function = load_policy_from_config(
        policy_params=config.get("node_expansion", {}),
        resources=resources,
    )

    reaction_rules = load_reaction_rules(resources["reaction_rules"])

    eval_config = RolloutEvaluationConfig(
        policy_network=policy_function,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=tree_config.min_mol_size,
        max_depth=tree_config.max_depth,
        normalize=tree_config.normalize_scores,
    )
    config["node_evaluation"] = {
        **config.get("node_evaluation", {}),
        "evaluation_type": "rollout",
        "min_mol_size": eval_config.min_mol_size,
        "max_depth": eval_config.max_depth,
        "normalize": eval_config.normalize,
        "stochastic": eval_config.stochastic,
    }
    effective_config_path = write_effective_config(config, save_dir)
    evaluation_function = load_evaluation_function(eval_config)

    logger.info("Retrosynthesis starting")
    results, solved_count, runtime = run_synplanner_predictions(
        benchmark=benchmark,
        tree_config=tree_config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_function,
        evaluation_function=evaluation_function,
        limit=args.limit,
    )
    parameters = {
        "iteration_limit": args.iteration_limit,
        "search_strategy": tree_config.search_strategy,
        "evaluation_kind": "rollout",
    }
    if args.limit is not None:
        parameters["limit"] = args.limit

    save_synplanner_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        stock_path=stock_path,
        effective_config_path=effective_config_path,
        config_template_path=config_path,
        script_name="runtime/synplanner/3-run-synp-rollout.py",
        benchmark=benchmark,
        planner_version=PLANNER_VERSION,
        parameters=parameters,
    )
