# Runtime provenance

A central theme of RetroCast is clear and verifiable provenance for every workflow. This reproducibility begins with the initial execution of planners. This document defines what pandora runners must preserve when they write raw RetroCast results.

Everything in this doc motivated by a single principle: when inspecting results of a particular model in the 2-raw folder, you MUST be able to see the values of every single parameter/toggle that could've affected the performance of the planner. 

Many planners specify those parameters in a config file. Default values should be placed under `data/retrocast/0-assets/model-configs`.

examples:

- `0-assets/model-configs/<adapter>/<strategy>.yaml`
- `0-assets/model-configs/<adapter>/<planner-family>.yaml`

However, a user might wish to run the planner with larger maximum depth or longer time limit than the defaults, modifying those config values with cli overrides. This effective config should be written to

```bash
2-raw/<runner>/<benchmark>/config.effective.yaml
```

In addition to an effective config, 2-raw folder should write a manifest, which records hashed inputs, hashed outputs, and compact metadata for discovery.

## what goes where

### manifest

put queryable run identity in `parameters`.

examples:

- `adapter`
- `planner_version`
- `raw_results_filename`
- `config_template_path`
- `effective_config_path`
- `iteration_limit`
- `max_transforms`
- `max_time`
- `search_strategy`
- `evaluation_kind`
- `limit`, when a runner processes only part of a benchmark

put provenance files in `source_files`.

examples:

- benchmark definition
- stock file, when used directly
- `config.effective.yaml`
- `config.template.yaml`, if the runner keeps a template copy

put generated artifacts in `output_files`.

examples:

- `results.json.gz`
- other raw planner outputs consumed downstream

put summary counts in `statistics`. keep execution timing in `execution_stats.json.gz`.

### effective config

put complete result-affecting planner state in `config.effective.yaml`.

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

rule: if changing a value can change planner output, the value belongs in the effective config or in another hashed source artifact.

## runner contract

every raw runner must satisfy these rules.

1. create the raw result directory before writing provenance files.

2. read the shared template from `0-assets`, but do not use that shared path as the run's only config provenance.

3. build the effective config before running the planner.

4. apply cli overrides to the effective config.

5. record runner-imposed behavior in the effective config, not only in python code.

6. write `config.effective.yaml` into the raw result directory before manifest creation.

7. pass the same effective config to the planner that was written to disk.

8. create the manifest with the raw-directory effective config in `source_files`.

9. store the original template path in manifest `parameters.config_template_path`.

10. store high-level cli knobs in manifest `parameters`, even when they also appear in the effective config.

11. manifest paths must be relative to `data/retrocast`.

12. `retrocast verify --target <raw-run-dir>/manifest.json` must still pass after editing `0-assets/model-configs`.

optional but recommended: copy the unmodified template to `config.template.yaml` and include it in `source_files`. this makes template-vs-effective diffs easy without relying on mutable shared assets.

## review checklist

before adding or changing a runner:

- can the run verify after editing the shared template?
- are result-affecting cli args in manifest `parameters`?
- are result-affecting planner values in `config.effective.yaml`?
- does the manifest source the copied effective config?
- are manifest paths rooted at `data/retrocast`?
- does the runner avoid machine-local paths unless required by the planner?
- does `retrocast verify --target <raw-run-dir>/manifest.json` pass?
