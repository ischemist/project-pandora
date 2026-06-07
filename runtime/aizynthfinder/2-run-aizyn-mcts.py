"""
Run AiZynthFinder MCTS retrosynthesis predictions on a batch of targets.

Example usage:
    uv run --directory runtime/aizynthfinder 2-run-aizyn-mcts.py --benchmark random-n5-50 --limit 2
    uv run --directory runtime/aizynthfinder 2-run-aizyn-mcts.py \
        --benchmark random-n5-50 --model v2026-06-05-route --iteration-limit 500 --max-transforms 10 --limit 2

Ubuntu runtime deps:
    sudo apt-get install -y libxrender1 libxext6 libsm6
"""

from retrocast.utils.logging import configure_script_logging, logger
from utils import (
    AIZYNTHFINDER_DIR,
    RAW_DIR,
    benchmark_stock_name,
    create_benchmark_parser,
    load_aizynthfinder_benchmark,
    load_config,
    quiet_aizynthfinder_debug_logs,
    run_aizynthfinder_predictions,
    save_aizynthfinder_results,
    write_effective_config,
)

configure_script_logging()
quiet_aizynthfinder_debug_logs()

PLANNER_VERSION = "4.4.1"
MODEL_CONFIGS = {
    "aizyn": "uspto",
    "v2026-06-05-reaction": "retrocast_v2026-06-05_reaction",
    "v2026-06-05-route": "retrocast_v2026-06-05_route",
}

if __name__ == "__main__":
    parser = create_benchmark_parser("Run AiZynthFinder MCTS")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="aizyn",
        help="Expansion policy model/config to use.",
    )
    args = parser.parse_args()

    benchmark, bench_path = load_aizynthfinder_benchmark(args.benchmark)

    folder_name = (
        f"aizynthfinder-{PLANNER_VERSION}-mcts-{args.model}"
        f"-iter{args.iteration_limit}-depth{args.max_transforms}"
    )
    save_dir = RAW_DIR / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = AIZYNTHFINDER_DIR / "config-mcts.yaml"
    expansion_policy_name = MODEL_CONFIGS[args.model]
    stock_name = benchmark_stock_name(benchmark)
    assert stock_name is not None

    logger.info(f"stock: {stock_name}")
    logger.info(f"model: {args.model} (expansion policy: {expansion_policy_name})")
    logger.info(f"iteration limit: {args.iteration_limit} (config: {config_path.name})")
    logger.info(f"max transforms: {args.max_transforms}")

    config = load_config(config_path, stock_name, args.iteration_limit, args.max_transforms)
    effective_config_path = write_effective_config(config, save_dir)

    results, solved_count, runtime = run_aizynthfinder_predictions(
        benchmark=benchmark,
        config=config,
        expansion_policy_name=expansion_policy_name,
        limit=args.limit,
    )
    parameters = {
        "planner_version": PLANNER_VERSION,
        "search_strategy": "mcts",
        "model": args.model,
        "iteration_limit": args.iteration_limit,
        "max_transforms": args.max_transforms,
    }
    if args.limit is not None:
        parameters["limit"] = args.limit

    save_aizynthfinder_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        effective_config_path=effective_config_path,
        config_template_path=config_path,
        script_name="runtime/aizynthfinder/2-run-aizyn-mcts.py",
        benchmark=benchmark,
        parameters=parameters,
        solved_count=solved_count,
    )
