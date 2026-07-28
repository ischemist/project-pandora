# MultiStepTTL runtime

This runtime converts MultiStepTTL's per-target pandas pickle outputs into the
raw JSON shape consumed by RetroCast's `multistepttl` adapter. Pickle discovery,
dataframe joins, route scoring, partial-failure behavior, and JSON-compatible
runtime-object serialization are owned by Pandora.

Place one directory per task target under the input directory. Each target
directory must contain exactly one `*__tree.pkl` and one
`*__prediction.pkl`. Then run:

```bash
uv run --directory runtime/multistepttl 1-serialize-pickles.py \
  --task uspto-190 \
  --input-dir staging/multistepttl/uspto-190
```

Paths are relative to `RETROCAST_DATA_DIR` unless absolute. The default output is
`2-raw/multistepttl/<task>/results.json.gz`. The serializer writes and strictly
verifies the planner manifest after hashing the task and every consumed pickle.

The output remains a target-ID mapping to route lists. Each route contains
`reactions` with `product` and dot-split `reactants`, plus `metadata` containing
`fwd_conf_score`, `score`, and `steps`.

Pandas pickles can execute arbitrary code during loading. Only process trusted
MultiStepTTL outputs.
