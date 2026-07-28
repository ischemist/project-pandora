# ASKCOS runtime

This directory is an isolated HTTP client for an external ASKCOS deployment. It does not install or configure the ASKCOS server.

Run a task directly against the server:

```bash
uv run --directory runtime/askcos 1-run-askcos.py \
  --task mkt-lin-500 \
  --askcos-url http://localhost:9321/get_buyable_paths \
  --server-release <image-tag-or-git-revision>
```

The runner preserves each ASKCOS response object unchanged under its target ID in
`2-raw/askcos/<task>/results.json.gz`. It also writes per-target execution timing,
the runner-controlled effective configuration, and a strict planner manifest.

If ASKCOS was run by another process that emitted one JSON file per target, gather
those files with:

```bash
uv run --directory runtime/askcos 2-gather-askcos-results.py \
  --task mkt-lin-500 \
  --eval-dir staging/askcos/mkt-lin-500
```

The input directory is relative to `RETROCAST_DATA_DIR` and files are matched first
by their one-based `0001_*.json` position, then by target name. The gather step is
Pandora-owned and writes the same documented `results.json.gz` raw format.

ASKCOS search policy, template models, and buyability data live in the server. The
client records the task's effective stock name, but cannot inspect or enforce the
server's configured stock. Pass `--server-release` and ensure `--askcos-url`
identifies an immutable deployment if the run must be reproducible.
