#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

benchmarks=(
  "ursa-expert-100"
  "ursa-bridge-100"
)

mcts_models="${AIZYNTH_MCTS_MODELS:-aizyn}"
iteration_limits=(100 500)
max_transforms=10

"$script_dir/1-download-assets.sh"

for benchmark in "${benchmarks[@]}"; do
  for iteration_limit in "${iteration_limits[@]}"; do
    for model in $mcts_models; do
      uv run --directory "$script_dir" 2-run-aizyn-mcts.py \
        --benchmark "$benchmark" \
        --model "$model" \
        --iteration-limit "$iteration_limit" \
        --max-transforms "$max_transforms"
    done

    uv run --directory "$script_dir" 3-run-aizyn-retro-star.py \
      --benchmark "$benchmark" \
      --iteration-limit "$iteration_limit" \
      --max-transforms "$max_transforms"
  done
done
