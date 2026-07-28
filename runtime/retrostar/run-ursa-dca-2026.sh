#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
data_dir="${RETROCAST_DATA_DIR:-$project_root/data/retrocast}"
base_url="https://files.ischemist.com/retrocast/data/1-benchmarks"
benchmark="ursa-dca-2026"

download_file() {
  local source_path="$1"
  local destination="$data_dir/1-benchmarks/$source_path"

  if [[ -s "$destination" ]]; then
    printf 'skipping existing %s\n' "$destination"
    return
  fi

  mkdir -p "$(dirname "$destination")"
  printf 'downloading %s -> %s\n' "$base_url/$source_path" "$destination"
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 \
    -o "$destination" "$base_url/$source_path"
}

uv run --directory "$script_dir" 1-download-assets.py

download_file "definitions/$benchmark.json.gz"
download_file "definitions/$benchmark.manifest.json"
download_file "stocks/ursa-stock.csv.gz"
download_file "stocks/ursa-stock.manifest.json"

uv run --directory "$script_dir" 2-run-retrostar.py \
  --benchmark "$benchmark" \
  --effort high
