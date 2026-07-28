#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmark="ursa-dca-2026"
iteration_limits=(100 500)
max_transforms=10

"$script_dir/1-download-assets.sh"

for iteration_limit in "${iteration_limits[@]}"; do
  uv run --directory "$script_dir" 3-run-aizyn-retro-star.py \
    --benchmark "$benchmark" \
    --iteration-limit "$iteration_limit" \
    --max-transforms "$max_transforms"
done
