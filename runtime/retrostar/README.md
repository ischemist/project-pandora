# Original Retro* runtime

This runtime isolates the original Retro* implementation and its transitive
dependencies from Pandora's other planners.

Download the model assets:

```bash
uv run --directory runtime/retrostar 1-download-assets.py
```

Run a benchmark:

```bash
uv run --directory runtime/retrostar 2-run-retrostar.py --benchmark random-n5-50
```

Use `--effort high` for 500 search iterations or `--limit N` for a smoke run.
The runner accepts schema-v2 task definitions and CSV stocks from
`data/retrocast/1-benchmarks`.
