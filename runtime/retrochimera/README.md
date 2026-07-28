# RetroChimera gather runtime

RetroChimera execution happens outside this repository and emits one JSON file
per target. This runtime owns the planner-specific gather step that maps those
files back to schema-v2 task target IDs and writes the documented raw
`results.json.gz` payload.

```bash
uv run --directory runtime/retrochimera 1-gather-results.py \
  --benchmark uspto-190 \
  --eval-dir data/retrocast/2-raw/retrochimera/uspto-190/parts
```

The evaluation directory and output directory must be inside the configured
RetroCast data root so every contributing file can be hashed in the planner
manifest. By default, the combined output is written beside `parts/`, under
`data/retrocast/2-raw/retrochimera/<benchmark>/results.json.gz`.

Set `RETROCAST_DATA_DIR` to use a different data root. Missing or malformed
per-target files are omitted and reported, preserving the historical gather
behavior.
