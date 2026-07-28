#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
benchmark="ursa-dca-2026"
shard_count="${AIZYN_SHARD_COUNT:-4}"
run_mcts="${AIZYN_RUN_MCTS:-1}"
run_retrostar="${AIZYN_RUN_RETROSTAR:-1}"
iteration_limit=500
max_transforms=10
logs_dir="${AIZYN_LOGS_DIR:-$project_root/data/retrocast/2-raw/.logs/aizynthfinder-ursa-dca-2026}"

if ((shard_count < 1)); then
  printf 'AIZYN_SHARD_COUNT must be at least 1\n' >&2
  exit 2
fi
if [[ "$run_mcts" != "1" && "$run_retrostar" != "1" ]]; then
  printf 'enable at least one of AIZYN_RUN_MCTS or AIZYN_RUN_RETROSTAR\n' >&2
  exit 2
fi

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
  PANDORA_FORCE_TERMINAL=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$python_bin" "$@" >"$logs_dir/$label.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
}

for shard_index in $(seq 0 $((shard_count - 1))); do
  if [[ "$run_mcts" == "1" ]]; then
    start_worker "mcts-$shard_index" \
      "$script_dir/2-run-aizyn-mcts.py" \
      --benchmark "$benchmark" \
      --model aizyn \
      --iteration-limit "$iteration_limit" \
      --max-transforms "$max_transforms" \
      --shard-count "$shard_count" \
      --shard-index "$shard_index"
  fi

  if [[ "$run_retrostar" == "1" ]]; then
    start_worker "retrostar-$shard_index" \
      "$script_dir/3-run-aizyn-retro-star.py" \
      --benchmark "$benchmark" \
      --iteration-limit "$iteration_limit" \
      --max-transforms "$max_transforms" \
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

if [[ "$status" -eq 0 && "$run_mcts" == "1" ]]; then
  "$python_bin" "$script_dir/4-combine-shards.py" \
    --benchmark "$benchmark" \
    --run-name "aizynthfinder-4.4.1-mcts-aizyn-iter${iteration_limit}-depth${max_transforms}" \
    --shard-count "$shard_count"
fi

if [[ "$status" -eq 0 && "$run_retrostar" == "1" ]]; then
  "$python_bin" "$script_dir/4-combine-shards.py" \
    --benchmark "$benchmark" \
    --run-name "aizynthfinder-4.4.1-retro-star-iter${iteration_limit}-depth${max_transforms}" \
    --shard-count "$shard_count"
fi

exit "$status"
