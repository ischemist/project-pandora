"""
Run Synplanner NMCS retrosynthesis predictions on a batch of targets.

This script processes targets from a benchmark using Synplanner's Nested Monte Carlo Search
algorithm and saves results in a structured format matching other prediction scripts.

Example usage:
    uv run --directory runtime/synplanner 4-run-synp-nmcs.py --benchmark uspto-190
    uv run --directory runtime/synplanner 4-run-synp-nmcs.py --benchmark random-n5-2-seed=20251030 --max-time 120

The benchmark definition should be located at: data/retrocast/1-benchmarks/definitions/{benchmark_name}.json.gz
Results are saved to: data/retrocast/2-raw/synplanner-{version}-nmcs-time{max_time}/{benchmark_name}/
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
)

configure_script_logging()
# Synplanner version - update when upgrading the library
PLANNER_VERSION = "1.3.2"
if __name__ == "__main__":
    parser = create_benchmark_parser("Run Synplanner NMCS (Nested Monte Carlo Search)")
    parser.add_argument(
        "--max-time",
        type=int,
        default=60,
        choices=[60, 120],
        help="Maximum search time in seconds.",
    )
    args = parser.parse_args()

    benchmark, building_blocks, bench_path, stock_path = load_benchmark_and_stock(args.benchmark)

    folder_name = f"synplanner-{PLANNER_VERSION}-nmcs-time{args.max_time}"
    save_dir = RAW_DIR / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"max time: {args.max_time}")

    config_path = SYNPLANNER_DIR / "nmcs-config.yaml"
    config = load_synplanner_config(config_path)
    resources = config["resources"]

    config["tree"]["max_time"] = args.max_time
    tree_config = TreeConfig.from_dict(config["tree"])

    policy_function = load_policy_from_config(
        policy_params=config.get("node_expansion", {}),
        resources=resources,
    )

    reaction_rules = load_reaction_rules(resources["reaction_rules"])

    # Create evaluation function for NMCS
    eval_config = RolloutEvaluationConfig(
        policy_network=policy_function,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=tree_config.min_mol_size,
        max_depth=tree_config.max_depth,
        normalize=False,
    )
    evaluation_function = load_evaluation_function(eval_config)

    # Run predictions
    logger.info("Retrosynthesis starting with NMCS algorithm")
    results, solved_count, runtime = run_synplanner_predictions(
        benchmark=benchmark,
        tree_config=tree_config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_function,
        evaluation_function=evaluation_function,
    )

    # Save results
    save_synplanner_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        stock_path=stock_path,
        config_path=config_path,
        script_name="runtime/synplanner/4-run-synp-nmcs.py",
        benchmark=benchmark,
        planner_version=PLANNER_VERSION,
        parameters={"max_time": args.max_time},
    )
