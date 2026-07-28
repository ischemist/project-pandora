# SynLlama conversion runtime

Pandora currently has no tracked SynLlama execution source or installable
upstream runtime. This directory therefore does not fabricate a model runner.
It converts a SynLlama `results.csv` produced elsewhere into the documented raw
format consumed by RetroCast's `synllama` adapter.

```bash
uv run --directory runtime/synllama 1-convert-to-json.py \
  --task uspto-190 \
  --input staging/synllama/uspto-190/results.csv
```

Input and output paths are relative to `RETROCAST_DATA_DIR` unless absolute.
The default output is `2-raw/synllama/<task>`.

The converter preserves each CSV `synthesis` value exactly and groups rows by
`Structure ID` as:

```json
{
  "target-id": [
    {"synthesis_string": "reactant;R1;product"}
  ]
}
```

It also writes the historical summary, converts unambiguous non-negative
`time, s` values to per-target execution statistics, and strictly verifies a
planner manifest that hashes the task and source CSV. Conflicting per-route
times for one target are omitted from execution statistics rather than guessed.
