# paroutes curation scripts

## benchmark-prep

Scripts that build benchmark definitions from raw PaRoutes assets.

```bash
uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py --prune-intermediates
uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py --check-buyables --prune-intermediates
uv run scripts/paroutes/benchmark-prep/02-create-subsets.py
```

Public training-set release prep stays in project-procrustes with the release
artifacts and publish script.
