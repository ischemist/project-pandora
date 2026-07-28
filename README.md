# project-pandora

isolated runner scripts and model-specific glue for synthesis planning.

## why this repo exists

these scripts used to live inside [`project-procrustes`](https://github.com/ischemist/project-procrustes), which is the home of RetroCast itself: canonical schemas, adapters, scoring, analysis, and benchmark tooling.

that arrangement worked, but it blurred two different concerns:

- `project-procrustes` is evaluation and normalization infrastructure
- `project-pandora` is the execution-plane junk drawer for planner-specific runtimes

the split exists for a few reasons:

- many planner runners are independent of RetroCast as a *product*, even when they still reuse some RetroCast helper code
- runner environments pull in old, research-grade, or model-specific dependency stacks that we do not want contaminating the main retrocast repo
- keeping these dependencies out of procrustes reduces noisy security alerts and keeps the core repo focused on the canonicalization/evaluation layer
- Daedalus-style execution via isolated directories makes it more natural to treat runners as their own external runtime surface

## relationship to retrocast

RetroCast docs live at [retrocast.ischemist.com](https://retrocast.ischemist.com/). RetroCast still defines the data conventions most scripts here follow:

- benchmark definitions
- stock files
- raw result layout
- manifest and execution-stat schemas
- adapter expectations for downstream ingestion/scoring

Pandora owns planner execution, logging, progress, timing measurement, planner-specific gather behavior, and serialization of planner runtime objects into documented raw output. RetroCast supplies the shared task, stock, JSON, execution-stat, manifest, and provenance contracts used to publish those raw artifacts.

## what lives here

`project-pandora` holds model-facing runtime code, including:

- planner wrappers and execution scripts
- asset/bootstrap helpers
- one-off preprocessing steps tied to specific models
- per-runner locked environments where needed
- shared RetroCast-style runtime assets under `data/retrocast`

Planner environments live under `runtime/<planner>` with their own locks. Legacy entries in the list below are being migrated from `scripts/` into that layout:

- `runtime/aizynthfinder`
- `runtime/askcos`
- `runtime/directmultistep`
- `runtime/dreamretroer`
- `runtime/multistepttl`
- `runtime/retrochimera`
- `runtime/retrostar`
- `runtime/synllama`
- `runtime/syntheseus`
- `runtime/synplanner`

## what does not live here

things that remain in `project-procrustes`:

- retrocast library code
- adapters and canonical schemas
- scoring/statistical analysis
- benchmark curation and preparation
- general comparison/reporting workflows

## status

Legacy scripts are being moved into isolated runtimes. New runner code must use RetroCast's curated public producer API rather than importing its internal Python package architecture.

`data/retrocast/0-assets/model-configs` is the shared transitional home for model configs. lightweight yaml/config files are tracked; large downloaded model payloads such as checkpoints, onnx files, hdf5 stocks, pickles, and generated outputs are ignored locally.
