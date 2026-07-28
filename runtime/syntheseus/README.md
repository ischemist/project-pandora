# Syntheseus runtime

This environment runs Syntheseus 0.7.2 with LocalRetro on Linux x86-64. DGL and
torch-scatter are pinned to their upstream PyTorch 2.6 wheels because their
compiled extensions must match the runtime's PyTorch release. DGL 2.5 is the
newest upstream build published for that PyTorch line, and PyTorch 2.6 is the
newest DGL wheel index published for this platform. Advancing PyTorch further
requires a matching upstream GraphBolt build.

DistDGL is unsupported in this runtime; model initialization replaces its
pickle-based RPC deserializer with a fail-closed error.

Pandora owns the conversion from Syntheseus runtime objects to the raw
`syntheseus` adapter tree. `TemplateAwareLocalRetroModel` mirrors the upstream
LocalRetro decoding path while retaining the selected source reaction SMARTS in
reaction metadata:

- `metadata.template` is the source reaction SMARTS.
- `metadata.local_retro_template` is the complete LocalRetro template identifier.
- `metadata.probability` is the decoded prediction score.

LocalRetro removes atom mapping while decoding its concrete reactants, so this
runtime deliberately does not emit `metadata.mapped_reaction_smiles`.

Run either search strategy with:

```bash
uv run --directory runtime/syntheseus 1-run-synth-bfs-local-retro.py --benchmark <task>
uv run --directory runtime/syntheseus 2-run-synth-retro0-local-retro.py --benchmark <task>
```
