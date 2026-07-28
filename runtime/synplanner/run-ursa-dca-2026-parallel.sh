#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
benchmark="ursa-dca-2026"
shard_count="${SYNPLANNER_SHARD_COUNT:-10}"
run_val="${SYNPLANNER_RUN_VAL:-1}"
run_rollout="${SYNPLANNER_RUN_ROLLOUT:-1}"
iteration_limit=500
logs_dir="${SYNPLANNER_LOGS_DIR:-$project_root/data/retrocast/2-raw/.logs/synplanner-ursa-dca-2026}"

if ((shard_count < 1)); then
  printf 'SYNPLANNER_SHARD_COUNT must be at least 1\n' >&2
  exit 2
fi
if [[ "$run_val" != "1" && "$run_rollout" != "1" ]]; then
  printf 'enable at least one of SYNPLANNER_RUN_VAL or SYNPLANNER_RUN_ROLLOUT\n' >&2
  exit 2
fi

uv run --directory "$script_dir" 1-download-assets.py
uv run --directory "$script_dir" python -c \
  "from utils import load_benchmark_and_stock; load_benchmark_and_stock('$benchmark')"
mkdir -p "$logs_dir"

pids=()
labels=()

start_worker() {
  local label="$1"
  shift
  printf 'starting %s\n' "$label"
  "$@" >"$logs_dir/$label.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
}

for shard_index in $(seq 0 $((shard_count - 1))); do
  if [[ "$run_val" == "1" ]]; then
    start_worker "val-$shard_index" \
      uv run --directory "$script_dir" 2-run-synp-val.py \
      --benchmark "$benchmark" \
      --iteration-limit "$iteration_limit" \
      --shard-count "$shard_count" \
      --shard-index "$shard_index"
  fi

  if [[ "$run_rollout" == "1" ]]; then
    start_worker "rollout-$shard_index" \
      uv run --directory "$script_dir" 3-run-synp-rollout.py \
      --benchmark "$benchmark" \
      --iteration-limit "$iteration_limit" \
      --shard-count "$shard_count" \
      --shard-index "$shard_index"
  fi
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

if [[ "$status" -eq 0 && "$run_val" == "1" ]]; then
  uv run --directory "$script_dir" 5-combine-shards.py \
    --benchmark "$benchmark" \
    --run-name "synplanner-1.3.2-mcts-val-iter${iteration_limit}" \
    --shard-count "$shard_count"
fi

if [[ "$status" -eq 0 && "$run_rollout" == "1" ]]; then
  uv run --directory "$script_dir" 5-combine-shards.py \
    --benchmark "$benchmark" \
    --run-name "synplanner-1.3.2-mcts-rollout-iter${iteration_limit}" \
    --shard-count "$shard_count"
fi

exit "$status"
