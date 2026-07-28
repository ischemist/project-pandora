from __future__ import annotations

import local_retro.LocalTemplate.template_decoder as decoder
from syntheseus import Molecule
from template_aware_local_retro import _decode_predictions


def test_decode_predictions_preserves_source_template(monkeypatch) -> None:
    source_template = "[C:1]-[O:2]>>[C:1].[O:2]_10_00_00"
    monkeypatch.setattr(
        decoder,
        "read_prediction",
        lambda *args, **kwargs: (object(), 1, source_template, {}, 0.8),
    )
    monkeypatch.setattr(decoder, "decode_localtemplate", lambda *args, **kwargs: "C.C")

    reaction = _decode_predictions(
        Molecule("CCO"),
        ["ignored"],
        {"atom_templates": {}, "bond_templates": {}, "template_infos": {}},
        1,
    )[0]

    assert reaction.reaction_smiles == "C.C>>CCO"
    assert reaction.metadata["probability"] == 0.8
    assert reaction.metadata["template"] == "[C:1]-[O:2]>>[C:1].[O:2]"
    assert reaction.metadata["local_retro_template"] == source_template
    assert "mapped_reaction_smiles" not in reaction.metadata
