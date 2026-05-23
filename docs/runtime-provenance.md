# Runtime provenance

A central theme of RetroCast is clear and verifiable provenance for every workflow. This reproducibility begins with the initial execution of planners. This document defines what pandora runners must preserve when they write raw RetroCast results.

Everything in this doc is motivated by a single principle: when inspecting results of a particular model in the `2-raw` folder, you MUST be able to see the values of every parameter or toggle that could have affected planner output.

Many planners specify those parameters in a config file. Default values should be placed under `data/retrocast/0-assets/model-configs`.

examples:

- `0-assets/model-configs/<adapter>/<strategy>.yaml`
- `0-assets/model-configs/<adapter>/<planner-family>.yaml`

However, a user might run the planner with a larger maximum depth, longer time limit, different stock, or other cli overrides. The effective config is the exact planner configuration after template defaults, cli overrides, benchmark choices, and runner-imposed choices are applied.

Write the effective config to:

```bash
2-raw/<runner>/<benchmark>/config.effective.yaml
```

The manifest records hashed inputs, hashed outputs, and compact metadata for discovery.

## What goes where

### decision rule

If a value is needed to rerun the planner, put it in `config.effective.yaml`.

If a value is needed to find, group, compare, or label runs, duplicate it in manifest `parameters`.

Do not put a result-affecting value only in `parameters`. `parameters` are easy to query for, but they do not replace hashed planner state.

### Manifest

Put queryable run identity in `parameters`. Keep it compact and scalar-ish.

examples:

- `adapter`
- `planner_version`
- `algorithm`
- `raw_results_filename`
- `config_template_path`
- `effective_config_path`
- `iteration_limit`
- `max_transforms`
- `max_time`
- `search_strategy`
- `evaluation_kind`
- `limit`, when a runner processes only part of a benchmark

Put hashed provenance files in `source_files`. These are the files used as evidence for verification.

examples:

- benchmark definition
- stock file, when used directly
- `config.effective.yaml`

Put generated artifacts in `output_files`.

examples:

- `results.json.gz`
- other raw planner outputs consumed downstream

Put summary counts in `statistics`. Keep execution timing in `execution_stats.json.gz`.

Path fields in `parameters` identify important artifacts:

- `config_template_path` records where the mutable shared template came from.
- `effective_config_path` records where the immutable run-local effective config was written.

### effective config

Put complete result-affecting planner state in `config.effective.yaml`.

examples:

- tree limits
- depth limits
- search strategy
- policy settings
- filter settings
- evaluation settings
- expansion settings
- model/resource paths or ids
- selected stock used by the planner
- inherited template defaults that affect output

Rule: if changing a value can change planner output, the value belongs in the effective config or in another hashed source artifact.

The runner should hash the written effective config file by including it in manifest `source_files`.

## runner contract

every raw runner must satisfy these rules.

1. create the raw result directory before writing provenance files.

2. read the shared template from `0-assets`, but do not use that shared path as the run's only config provenance.

3. build the effective config before running the planner.

4. apply cli overrides to the effective config.

5. record runner-imposed behavior in the effective config, not only in python code.

6. write `config.effective.yaml` into the raw result directory before manifest creation.

7. pass the same effective config to the planner that was written to disk.

8. create the manifest with the raw-directory effective config in `source_files`, so the effective config is hashed.

9. store the original template path in manifest `parameters.config_template_path`.

10. store high-level cli knobs in manifest `parameters`, even when they also appear in the effective config.

11. manifest paths must be relative to the active RetroCast data root, which defaults to `data/retrocast` and may be overridden with `RETROCAST_DATA_DIR`.

12. `retrocast verify --target <raw-run-dir>/manifest.json` must still pass after editing `0-assets/model-configs`.
