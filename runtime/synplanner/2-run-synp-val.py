"""
Run Synplanner MCTS retrosynthesis predictions on a batch of targets using value-network evaluation.

This script processes targets from a benchmark using Synplanner's MCTS algorithm
with evaluation-first search guided by a value network and saves results in a structured
format matching other prediction scripts.

Example usage:
    uv run --directory runtime/synplanner 2-run-synp-val.py --benchmark mkt-cnv-160
    uv run --directory runtime/synplanner 2-run-synp-val.py --benchmark random-n5-2-seed=20251030 --iteration-limit 500

The benchmark definition should be located at: data/retrocast/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/retrocast/2-raw/synplanner-{version}-mcts-val-iter{iteration_limit}/{benchmark_name}/
"""

from synplan.mcts.tree import TreeConfig
from synplan.utils.config import ValueNetworkEvaluationConfig
from synplan.utils.loading import load_evaluation_function, load_reaction_rules
from utils import (
    RAW_DIR,
    SYNPLANNER_DIR,
    benchmark_stock_name,
    configure_script_logging,
    create_benchmark_parser,
    load_benchmark_and_stock,
    load_policy_from_config,
    load_synplanner_config,
    logger,
    run_synplanner_predictions,
    save_synplanner_results,
    shard_save_dir,
    write_effective_config,
)

configure_script_logging()

# Synplanner version - update when upgrading the library
PLANNER_VERSION = "1.3.2"

if __name__ == "__main__":
    parser = create_benchmark_parser(
        "Run Synplanner MCTS with value-network evaluation",
        enable_sharding=True,
    )
    parser.add_argument(
        "--iteration-limit",
        type=int,
        default=100,
        choices=[100, 500],
        help="Maximum tree search iterations.",
    )
    args = parser.parse_args()
    if args.shard_index >= args.shard_count:
        parser.error("--shard-index must be smaller than --shard-count")

    benchmark, building_blocks, bench_path, stock_path = load_benchmark_and_stock(args.benchmark)
    stock_name = benchmark_stock_name(benchmark)

    folder_name = f"synplanner-{PLANNER_VERSION}-mcts-val-iter{args.iteration_limit}"
    save_dir = shard_save_dir(
        RAW_DIR / folder_name / benchmark["name"],
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("stock: %s", stock_name)
    logger.info("iteration limit: %s", args.iteration_limit)

    config_path = SYNPLANNER_DIR / "mcts-val-config.yaml"
    config = load_synplanner_config(config_path)
    resources = config["resources"]

    config["tree"]["max_iterations"] = args.iteration_limit
    tree_config = TreeConfig.from_dict(config["tree"])
    tree_config.search_strategy = "evaluation_first"
    tree_config.evaluation_agg = config["node_evaluation"].get("evaluation_agg", tree_config.evaluation_agg)
    config["tree"] = tree_config.to_dict()
    config["stock"] = {"name": stock_name, "path": str(stock_path)}

    policy_function = load_policy_from_config(
        policy_params=config.get("node_expansion", {}),
        resources=resources,
    )

    reaction_rules = load_reaction_rules(resources["reaction_rules"])

    evaluation_type = str(config["node_evaluation"].get("evaluation_type", "")).lower()
    if evaluation_type and evaluation_type != "gcn":
        logger.warning("Config evaluation_type=%r ignored; using value network evaluation.", evaluation_type)

    eval_config = ValueNetworkEvaluationConfig(weights_path=resources["value_weights"])
    config["node_evaluation"] = {
        **config.get("node_evaluation", {}),
        "evaluation_type": "gcn",
        "value_weights": resources["value_weights"],
        "normalize": eval_config.normalize,
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
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    parameters = {
        "algorithm": tree_config.algorithm,
        "iteration_limit": args.iteration_limit,
        "search_strategy": tree_config.search_strategy,
        "evaluation_kind": "value_network",
    }
    if args.limit is not None:
        parameters["limit"] = args.limit
    if args.shard_count > 1:
        parameters.update(
            {
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
                "shard_strategy": "round_robin",
            }
        )

    save_synplanner_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        stock_path=stock_path,
        effective_config_path=effective_config_path,
        config_template_path=config_path,
        script_name="runtime/synplanner/2-run-synp-val.py",
        benchmark=benchmark,
        planner_version=PLANNER_VERSION,
        parameters=parameters,
    )
