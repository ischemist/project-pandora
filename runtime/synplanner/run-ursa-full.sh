#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

benchmarks=(
  "ursa-expert-100"
  "ursa-bridge-100"
)

iteration_limits=(100 500)

uv run --directory "$script_dir" 1-download-assets.py

for benchmark in "${benchmarks[@]}"; do
  for iteration_limit in "${iteration_limits[@]}"; do
    uv run --directory "$script_dir" 2-run-synp-val.py \
      --benchmark "$benchmark" \
      --iteration-limit "$iteration_limit"

    uv run --directory "$script_dir" 3-run-synp-rollout.py \
      --benchmark "$benchmark" \
      --iteration-limit "$iteration_limit"
  done
done
