# DirectMultiStep runtime

This directory is the isolated Pandora runtime for DirectMultiStep 1.1.3.

```bash
uv sync --directory runtime/directmultistep
runtime/directmultistep/1-download-assets.sh
uv run --directory runtime/directmultistep 2-run-dms.py \
  --benchmark uspto-190 \
  --model-name "explorer XL" \
  --device cuda \
  --use-fp16
```

The runner writes stock-agnostic DMS route trees to `results.json.gz`. Stock
termination is evaluated later by RetroCast, so the producer does not emit
separate buyables or N1/N5-filtered copies of the same search.

Partitioned tasks can be gathered after every part has a valid DMS planner
manifest:

```bash
uv run --directory runtime/directmultistep 3-combine-results.py \
  --run-name dms-wide-fp16 \
  --benchmark uspto-190 \
  --parts pt1 pt2 pt3 pt4
```

The combiner rejects missing, duplicate, or unexpected target IDs. It also
combines per-target execution statistics and records the part manifests as
provenance sources.

The RetroCast dependency is temporarily locked to the exact commit containing
the 0.8.2 producer API. Replace the Git dependency with `retrocast==0.8.2`
after that release is published and regenerate `uv.lock`.
