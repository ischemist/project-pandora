# DreamRetroer result runtime

Pandora currently receives DreamRetroer output from an external upstream v1.0
checkout. This isolated runtime owns the existing post-run responsibility: it
validates and serializes native `results.json`, writes execution statistics,
and creates a strict RetroCast planner manifest.

The upstream checkout must place these files under
`2-raw/<run-name>/<benchmark>/`:

- `results.json`: target-keyed native `RSPlanner` output
- `config.effective.yaml`: the exact planner settings and asset identities used
  for the run

Then gather one benchmark:

```bash
uv run --directory runtime/dreamretroer 1-gather-results.py \
  --benchmark uspto-190 \
  --source 0-assets/model-configs/dreamretroer/origin_dict.csv \
  --source 0-assets/model-configs/dreamretroer/template_rules.dat \
  --source 0-assets/model-configs/dreamretroer/retro_star_value.ckpt
```

Omit `--benchmark` to gather every directory containing `results.json`.
Additional sources are hashed into provenance and must be inside the active
`RETROCAST_DATA_DIR`.

DreamRetroer's upstream v1.0 distribution targets Python 3.7 and Conda, vendors
modified research packages, and does not expose a reproducible modern package
installation. This runtime therefore does not claim to execute the planner.
Adding a maintained execution wrapper requires a separately qualified upstream
environment and is not a missing RetroCast producer API.

The RetroCast dependency is temporarily locked to the exact commit containing
the 0.8.2 producer API. Replace the Git dependency with `retrocast==0.8.2`
after that release is published and regenerate `uv.lock`.
