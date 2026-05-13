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
- manifests and result serialization
- adapter expectations for downstream ingestion/scoring

so this repo is split out physically, but not fully severed conceptually. some scripts here still import `RetroCast` helpers for benchmark io, manifests, and serialization.

## what lives here

`project-pandora` holds model-facing runtime code, including:

- planner wrappers and execution scripts
- asset/bootstrap helpers
- one-off preprocessing steps tied to specific models
- per-runner locked environments where needed

currently that includes:

- `aizynthfinder`
- `askcos`
- `directmultistep`
- `dreamretroer`
- `multistepttl`
- `retrochimera`
- `retrostar`
- `synllama`
- `syntheseus`
- `runtime/synplanner`

## what does not live here

things that remain in `project-procrustes`:

- retrocast library code
- adapters and canonical schemas
- scoring/statistical analysis
- benchmark curation and preparation
- general comparison/reporting workflows

## status

this repo is currently an extraction, not a total dependency divorce. expect some awkward transitional coupling while runner code is peeled away from shared retrocast utilities.
