#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
data_dir="${RETROCAST_DATA_DIR:-$project_root/data/retrocast}"
aizynthfinder_dir="$data_dir/0-assets/model-configs/aizynthfinder"
benchmark_base_url="https://files.ischemist.com/retrocast/data/1-benchmarks"

download_file() {
  local url="$1"
  local destination="$2"

  if [[ -s "$destination" ]]; then
    printf 'skipping existing %s\n' "$destination"
    return
  fi

  mkdir -p "$(dirname "$destination")"
  printf 'downloading %s -> %s\n' "$url" "$destination"
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 -o "$destination" "$url"
}

model_files=(
  "https://zenodo.org/record/7797465/files/uspto_model.onnx|uspto_model.onnx"
  "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz|uspto_templates.csv.gz"
  "https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx|uspto_ringbreaker_model.onnx"
  "https://zenodo.org/record/7341155/files/uspto_ringbreaker_unique_templates.csv.gz|uspto_ringbreaker_templates.csv.gz"
  "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx|uspto_filter_model.onnx"
  "https://files.ischemist.com/assets/aizynthfinder/retrocast_v2026-06-05_ss_reaction-holdout-n1-n5_model.onnx|retrocast_v2026-06-05_ss_reaction-holdout-n1-n5_model.onnx"
  "https://files.ischemist.com/assets/aizynthfinder/retrocast_v2026-06-05_ss_reaction-holdout-n1-n5_templates.csv.gz|retrocast_v2026-06-05_ss_reaction-holdout-n1-n5_templates.csv.gz"
  "https://files.ischemist.com/assets/aizynthfinder/retrocast_v2026-06-05_ss_reaction-holdout-plus-n5_model.onnx|retrocast_v2026-06-05_ss_reaction-holdout-plus-n5_model.onnx"
  "https://files.ischemist.com/assets/aizynthfinder/retrocast_v2026-06-05_ss_reaction-holdout-plus-n5_templates.csv.gz|retrocast_v2026-06-05_ss_reaction-holdout-plus-n5_templates.csv.gz"
  "https://files.ischemist.com/assets/aizynthfinder/retrocast_v2026-06-05_ss_route-holdout-n1-n5_model.onnx|retrocast_v2026-06-05_ss_route-holdout-n1-n5_model.onnx"
  "https://files.ischemist.com/assets/aizynthfinder/retrocast_v2026-06-05_ss_route-holdout-n1-n5_templates.csv.gz|retrocast_v2026-06-05_ss_route-holdout-n1-n5_templates.csv.gz"
  "https://github.com/MolecularAI/PaRoutes/blob/main/publication/retrostar_value_model.pickle?raw=true|retrostar_value_model.pickle"
)

benchmark_files=(
  "definitions/ursa-dca-2026.json.gz|1-benchmarks/definitions/ursa-dca-2026.json.gz"
  "definitions/ursa-dca-2026.manifest.json|1-benchmarks/definitions/ursa-dca-2026.manifest.json"
  "definitions/ursa-expert-100.json.gz|1-benchmarks/definitions/ursa-expert-100.json.gz"
  "definitions/ursa-expert-100.manifest.json|1-benchmarks/definitions/ursa-expert-100.manifest.json"
  "definitions/ursa-bridge-100.json.gz|1-benchmarks/definitions/ursa-bridge-100.json.gz"
  "definitions/ursa-bridge-100.manifest.json|1-benchmarks/definitions/ursa-bridge-100.manifest.json"
  "stocks/ursa-stock.csv.gz|1-benchmarks/stocks/ursa-stock.csv.gz"
  "stocks/ursa-stock.txt.gz|1-benchmarks/stocks/ursa-stock.txt.gz"
  "stocks/ursa-stock.hdf5|1-benchmarks/stocks/ursa-stock.hdf5"
  "stocks/ursa-stock-meta.json.gz|1-benchmarks/stocks/ursa-stock-meta.json.gz"
  "stocks/ursa-stock.manifest.json|1-benchmarks/stocks/ursa-stock.manifest.json"
)

for entry in "${model_files[@]}"; do
  IFS='|' read -r url filename <<< "$entry"
  download_file "$url" "$aizynthfinder_dir/$filename"
done

for entry in "${benchmark_files[@]}"; do
  IFS='|' read -r source_path destination_path <<< "$entry"
  download_file "$benchmark_base_url/$source_path" "$data_dir/$destination_path"
done
