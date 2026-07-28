#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmark="ursa-dca-2026"
iteration_limits=(100 500)

uv run --directory "$script_dir" 1-download-assets.py

for iteration_limit in "${iteration_limits[@]}"; do
  uv run --directory "$script_dir" 2-run-synp-val.py \
    --benchmark "$benchmark" \
    --iteration-limit "$iteration_limit"

  uv run --directory "$script_dir" 3-run-synp-rollout.py \
    --benchmark "$benchmark" \
    --iteration-limit "$iteration_limit"
done
