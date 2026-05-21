"""
Run AiZynthFinder MCTS retrosynthesis predictions on a batch of targets.

Example usage:
    uv run --directory runtime/aizynthfinder 3-run-aizyn-mcts.py --benchmark random-n5-50 --limit 2
"""

from utils import (
    create_benchmark_parser,
    get_aizynthfinder_paths,
    load_aizynthfinder_benchmark,
    quiet_aizynthfinder_debug_logs,
    run_aizynthfinder_predictions,
    save_aizynthfinder_results,
)

from retrocast.utils.logging import configure_script_logging, logger

configure_script_logging()
quiet_aizynthfinder_debug_logs()

PLANNER_VERSION = "4.4.1"

if __name__ == "__main__":
    parser = create_benchmark_parser("Run AiZynthFinder MCTS")
    args = parser.parse_args()

    paths = get_aizynthfinder_paths()
    benchmark, bench_path = load_aizynthfinder_benchmark(args.benchmark, paths)

    folder_name = (
        f"aizynthfinder-{PLANNER_VERSION}-mcts"
        if args.effort == "normal"
        else f"aizynthfinder-{PLANNER_VERSION}-mcts-{args.effort}"
    )
    save_dir = paths.raw_dir / folder_name / benchmark.name
    save_dir.mkdir(parents=True, exist_ok=True)

    config_suffix = "" if args.effort == "normal" else f"-{args.effort}"
    config_path = paths.aizynthfinder_dir / f"config-mcts{config_suffix}.yaml"

    logger.info(f"stock: {benchmark.stock_name}")
    logger.info(f"effort: {args.effort} (config: {config_path.name})")

    results, solved_count, runtime = run_aizynthfinder_predictions(
        benchmark=benchmark,
        config_path=config_path,
        project_root=paths.project_root,
        limit=args.limit,
    )

    save_aizynthfinder_results(
        results=results,
        runtime=runtime,
        save_dir=save_dir,
        bench_path=bench_path,
        config_path=config_path,
        script_name="runtime/aizynthfinder/3-run-aizyn-mcts.py",
        benchmark=benchmark,
        planner_version=PLANNER_VERSION,
        solved_count=solved_count,
    )
