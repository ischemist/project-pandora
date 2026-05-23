# smoke tests

smoke tests run real runtime commands through their own uv environments:

```bash
uv run pytest -m smoke
uv run pytest tests/smoke/test_aizynthfinder_smoke.py
uv run pytest tests/smoke/test_synplanner_smoke.py
```

protocol:

- keep committed fixtures small and deterministic under `tests/fixtures`
- materialize ignored runtime inputs inside the test under `data/retrocast-tests`
- pass `RETROCAST_DATA_DIR=data/retrocast-tests` so smoke runs do not write into production retrocast data
- declare required ignored assets per smoke case and skip with a clear message if they are absent
- shell out with `uv run --directory runtime/<name> ...`; do not import runtime internals from the root test env
- assert output shape and manifest stats, not exact chemistry results
- keep default smoke cases short enough for local use; add slower cases only when they test a distinct runtime path
