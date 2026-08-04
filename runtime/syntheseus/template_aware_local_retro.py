"""LocalRetro inference wrapper that retains the selected source template."""

from __future__ import annotations

from typing import Any, Never

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference import LocalRetroModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_backwards

_DISTDGL_DISABLED_MESSAGE = (
    "DistDGL RPC deserialization is disabled in Pandora's LocalRetro runtime because it accepts pickle payloads"
)


def _reject_distdgl_rpc_payload(*_: Any, **__: Any) -> Never:
    raise RuntimeError(_DISTDGL_DISABLED_MESSAGE)


def _disable_distdgl_rpc_deserialization() -> None:
    """Make the unused DistDGL RPC path fail closed before LocalRetro initializes."""
    from dgl.distributed import rpc

    if rpc.deserialize_from_payload is not _reject_distdgl_rpc_payload:
        rpc.deserialize_from_payload = _reject_distdgl_rpc_payload


def _decode_predictions(
    input_molecule: Molecule,
    raw_predictions: list[str],
    args: dict[str, Any],
    num_results: int,
) -> list[SingleProductReaction]:
    from local_retro.LocalTemplate.template_decoder import decode_localtemplate, read_prediction

    outputs: list[str] = []
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()

    for prediction in raw_predictions:
        molecule, prediction_site, source_template, template_info, score = read_prediction(
            input_molecule.smiles,
            prediction,
            args["atom_templates"],
            args["bond_templates"],
            args["template_infos"],
        )
        template_smarts = source_template.split("_", 1)[0]
        decoder_template = ">>".join(f"({side})" for side in template_smarts.split(">>"))
        try:
            decoded_smiles = decode_localtemplate(molecule, prediction_site, decoder_template, template_info)
        except Exception:
            continue
        result_key = str((decoded_smiles, score))
        if decoded_smiles is None or result_key in seen:
            continue

        seen.add(result_key)
        outputs.append(decoded_smiles)
        metadata.append(
            {
                "probability": score,
                "template": template_smarts,
                "local_retro_template": source_template,
            }
        )
        if len(outputs) >= num_results:
            break

    return list(
        process_raw_smiles_outputs_backwards(
            input=input_molecule,
            output_list=outputs,
            metadata_list=metadata,
        )
    )


class TemplateAwareLocalRetroModel(LocalRetroModel):
    """LocalRetro model preserving its chosen reaction SMARTS on each prediction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disable_distdgl_rpc_deserialization()
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return "LocalRetro"

    def _build_batch_predictions(
        self,
        batch: Any,
        num_results: int,
        inputs: list[Molecule],
        batch_atom_logits: Any,
        batch_bond_logits: Any,
    ) -> list[list[SingleProductReaction]]:
        from local_retro.scripts.get_edit import combined_edit, get_bg_partition

        graphs, node_boundaries, edge_boundaries = get_bg_partition(batch)
        start_node = 0
        start_edge = 0
        raw_predictions_by_input: list[list[str]] = []

        for graph, end_node, end_edge in zip(graphs, node_boundaries, edge_boundaries, strict=True):
            prediction_types, prediction_sites, prediction_scores = combined_edit(
                graph,
                batch_atom_logits[start_node:end_node],
                batch_bond_logits[start_edge:end_edge],
                num_results,
            )
            start_node, start_edge = end_node, end_edge
            raw_predictions_by_input.append(
                [
                    (
                        f"({prediction_types[index]}, {prediction_sites[index][0]}, "
                        f"{prediction_sites[index][1]}, {prediction_scores[index]:.3f})"
                    )
                    for index in range(num_results)
                ]
            )

        return [
            _decode_predictions(input_molecule, raw_predictions, self.args, num_results)
            for input_molecule, raw_predictions in zip(inputs, raw_predictions_by_input, strict=True)
        ]
