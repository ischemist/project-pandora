"""Shared utilities for Synplanner scripts."""

from __future__ import annotations

import argparse
import gzip
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from retrocast.io import create_manifest, load_benchmark, save_execution_stats, save_json_gz
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.utils import ExecutionTimer
from retrocast.utils.logging import logger
from synplan.chem.reaction_routes.io import make_json
from synplan.chem.reaction_routes.route_cgr import extract_reactions
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.tree import Tree, TreeConfig
from synplan.utils.config import CombinedPolicyConfig, PolicyNetworkConfig
from synplan.utils.loading import load_building_blocks, load_combined_policy_function, load_policy_function
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RETROCAST_DATA_DIR", PROJECT_ROOT / "data" / "retrocast"))
SYNPLANNER_DIR = DATA_DIR / "0-assets" / "model-configs" / "synplanner"
STOCKS_DIR = DATA_DIR / "1-benchmarks" / "stocks"
BENCHMARKS_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RAW_DIR = DATA_DIR / "2-raw"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def create_benchmark_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark set (e.g. stratified-linear-600)",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum number of targets to process. Useful for smoke tests.",
    )
    return parser


def load_benchmark_and_stock(
    benchmark_name: str,
) -> tuple[BenchmarkSet, set[str], Path, Path]:
    bench_path = BENCHMARKS_DIR / f"{benchmark_name}.json.gz"
    benchmark = load_benchmark(bench_path)
    assert benchmark.stock_name is not None, f"Stock name not found in benchmark {benchmark_name}"

    stock_path = STOCKS_DIR / f"{benchmark.stock_name}.csv.gz"
    building_blocks = load_building_blocks_cached(stock_path)

    return benchmark, building_blocks, bench_path, stock_path


def load_policy_from_config(
    policy_params: dict,
    resources: dict[str, str],
) -> Callable:
    mode = policy_params.get("mode", "ranking")
    if mode == "combined":
        combined_policy_config = CombinedPolicyConfig(
            filtering_weights_path=resources["filtering_weights"],
            ranking_weights_path=resources["ranking_weights"],
            top_rules=policy_params.get("top_rules", 50),
            rule_prob_threshold=policy_params.get("rule_prob_threshold", 0.0),
        )
        return load_combined_policy_function(combined_config=combined_policy_config)
    return load_policy_function(policy_config=PolicyNetworkConfig(weights_path=resources["ranking_weights"]))


def load_synplanner_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    for key, value in config["resources"].items():
        path = Path(value)
        resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Required Synplanner resource {key!r} not found at {resolved_path}")
        config["resources"][key] = str(resolved_path)

    return config


def write_effective_config(config: dict[str, Any], save_dir: Path) -> Path:
    effective_config_path = save_dir / "config.effective.yaml"
    with open(effective_config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=True)
    return effective_config_path


def run_synplanner_predictions(
    benchmark: BenchmarkSet,
    tree_config: TreeConfig,
    reaction_rules: Any,
    building_blocks: set[str],
    expansion_function: Callable,
    evaluation_function: Callable,
    limit: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int, ExecutionStats]:
    """Run Synplanner search over all benchmark targets.

    Args:
        benchmark: Benchmark containing targets to process.
        tree_config: Configuration for the search tree.
        reaction_rules: Loaded reaction rules.
        building_blocks: Set of building block SMILES.
        expansion_function: Policy function for node expansion.
        evaluation_function: Evaluation function for node scoring.

    Returns:
        Tuple of (results_dict, solved_count, execution_runtime).
    """
    results: dict[str, list[dict[str, Any]]] = {}
    solved_count = 0
    timer = ExecutionTimer()

    targets = list(benchmark.targets.values())
    if limit is not None:
        targets = targets[:limit]

    for target in tqdm(targets, desc="Finding retrosynthetic paths"):
        with timer.measure(target.id):
            try:
                target_mol = mol_from_smiles(target.smiles, standardize=True)
                if not target_mol:
                    logger.warning(f"Could not create molecule for target {target.id} ({target.smiles}). Skipping.")
                    results[target.id] = []
                else:
                    search_tree = Tree(
                        target=target_mol,
                        config=tree_config,
                        reaction_rules=reaction_rules,
                        building_blocks=building_blocks,
                        expansion_function=expansion_function,
                        evaluation_function=evaluation_function,
                    )

                    # run the search
                    _ = list(search_tree)

                    if bool(search_tree.winning_nodes):
                        raw_routes = make_json(extract_reactions(search_tree))
                        results[target.id] = list(raw_routes.values())
                        solved_count += 1
                    else:
                        results[target.id] = []

            except Exception as e:
                logger.error(f"Failed to process target {target.id} ({target.smiles}): {e}", exc_info=True)
                results[target.id] = []

    return results, solved_count, timer.to_model()


def save_synplanner_results(
    results: dict[str, list[dict[str, Any]]],
    runtime: ExecutionStats,
    save_dir: Path,
    bench_path: Path,
    stock_path: Path,
    effective_config_path: Path,
    config_template_path: Path,
    script_name: str,
    benchmark: BenchmarkSet,
    planner_version: str,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Save Synplanner results, execution stats, and manifest.

    Args:
        results: Dictionary mapping target IDs to route lists.
        runtime: Execution timing information.
        save_dir: Directory to save outputs.
        bench_path: Path to benchmark definition file.
        stock_path: Path to stock file.
        config_path: Path to config file used.
        script_name: Name of the calling script (for manifest).
        benchmark: Benchmark object (for statistics).
        planner_version: Version of the Synplanner library used.
        parameters: Extra manifest parameters to record.
    """
    solved_count = sum(1 for routes in results.values() if routes)

    summary = {
        "solved_count": solved_count,
        "total_targets": len(results),
        "benchmark_total_targets": len(benchmark.targets),
    }

    save_json_gz(results, save_dir / "results.json.gz")
    save_execution_stats(runtime, save_dir / "execution_stats.json.gz")

    manifest_parameters = {
        "planner_version": planner_version,
        "config_template_path": str(config_template_path.relative_to(DATA_DIR)),
        "effective_config_path": str(effective_config_path.relative_to(DATA_DIR)),
    }
    if parameters:
        manifest_parameters.update(parameters)
    manifest_directives = {
        "adapter": "synplanner",
        "raw_results_filename": "results.json.gz",
    }

    manifest = create_manifest(
        action=script_name,
        sources=[bench_path, stock_path, effective_config_path],
        root_dir=save_dir.parents[2],  # data/retrocast directory
        outputs=[(save_dir / "results.json.gz", results, "unknown")],
        statistics=summary,
        parameters=manifest_parameters,
        directives=manifest_directives,
    )

    with open(save_dir / "manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Completed processing {len(benchmark.targets)} targets")
    logger.info(f"Solved: {solved_count}")


def load_building_blocks_cached(
    stock_path: Path,
    *,
    silent: bool = False,
) -> set[str]:
    """Load building blocks with caching for SynPlanner's standardization.

    SynPlanner uses special canonicalization that takes ~5 minutes for large stocks.
    This function checks for a pre-standardized cache file and uses it if available,
    otherwise standardizes and saves the result for future runs.

    Args:
        stock_path: Path to the original stock CSV file (e.g., buyables-stock.csv.gz).
        silent: Suppress progress output from load_building_blocks.

    Returns:
        Set of SMILES strings representing building blocks.
    """
    # Check for cached standardized version (e.g., buyables-stock-synplanner.csv.gz)
    cached_path = stock_path.with_name(stock_path.name.replace(".csv.gz", "-synplanner.csv.gz"))

    if cached_path.exists():
        return load_building_blocks(cached_path, standardize=False, silent=silent)

    # Load with standardization (slow ~5 min)
    building_blocks = load_building_blocks(stock_path, standardize=True, silent=silent)

    # Save cached version for next time
    with gzip.open(cached_path, "wt", encoding="utf-8") as f:
        f.write("SMILES\n")  # header
        for smiles in building_blocks:
            f.write(f"{smiles}\n")

    return building_blocks
