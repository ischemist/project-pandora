#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmark="ursa-dca-2026"
iteration_limit=500

uv run --directory "$script_dir" 1-download-assets.py

uv run --directory "$script_dir" 2-run-synp-val.py \
  --benchmark "$benchmark" \
  --iteration-limit "$iteration_limit"

uv run --directory "$script_dir" 3-run-synp-rollout.py \
  --benchmark "$benchmark" \
  --iteration-limit "$iteration_limit"
