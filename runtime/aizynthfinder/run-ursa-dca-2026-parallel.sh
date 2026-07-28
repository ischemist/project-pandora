#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
benchmark="ursa-dca-2026"
shard_count=4
iteration_limit=500
max_transforms=10
logs_dir="$project_root/data/retrocast/2-raw/.logs/aizynthfinder-ursa-dca-2026"

"$script_dir/1-download-assets.sh"
uv sync --directory "$script_dir" --frozen
mkdir -p "$logs_dir"

python_bin="$script_dir/.venv/bin/python"
pids=()
labels=()

start_worker() {
  local label="$1"
  shift
  printf 'starting %s\n' "$label"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$python_bin" "$@" >"$logs_dir/$label.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
}

for shard_index in $(seq 0 $((shard_count - 1))); do
  start_worker "mcts-$shard_index" \
    "$script_dir/2-run-aizyn-mcts.py" \
    --benchmark "$benchmark" \
    --model aizyn \
    --iteration-limit "$iteration_limit" \
    --max-transforms "$max_transforms" \
    --shard-count "$shard_count" \
    --shard-index "$shard_index"

  start_worker "retrostar-$shard_index" \
    "$script_dir/3-run-aizyn-retro-star.py" \
    --benchmark "$benchmark" \
    --iteration-limit "$iteration_limit" \
    --max-transforms "$max_transforms" \
    --shard-count "$shard_count" \
    --shard-index "$shard_index"
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'completed %s\n' "${labels[$index]}"
  else
    printf 'failed %s; see %s/%s.log\n' "${labels[$index]}" "$logs_dir" "${labels[$index]}" >&2
    status=1
  fi
done

exit "$status"
