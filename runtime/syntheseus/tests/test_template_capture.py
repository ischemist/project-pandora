from __future__ import annotations

import dgl
import local_retro.LocalTemplate.template_decoder as decoder
import pytest
import torch
from dgl.distributed import rpc
from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel
from template_aware_local_retro import (
    TemplateAwareLocalRetroModel,
    _decode_predictions,
    _disable_distdgl_rpc_deserialization,
)


def test_distdgl_rpc_deserialization_is_blocked(monkeypatch) -> None:
    original_deserializer = rpc.deserialize_from_payload
    monkeypatch.setattr(rpc, "deserialize_from_payload", original_deserializer)

    _disable_distdgl_rpc_deserialization()
    blocked_deserializer = rpc.deserialize_from_payload

    with pytest.raises(RuntimeError, match="DistDGL RPC deserialization is disabled"):
        blocked_deserializer(object, b"untrusted payload", [])

    _disable_distdgl_rpc_deserialization()
    assert rpc.deserialize_from_payload is blocked_deserializer


def test_model_blocks_distdgl_before_localretro_initializes(monkeypatch) -> None:
    original_deserializer = rpc.deserialize_from_payload
    monkeypatch.setattr(rpc, "deserialize_from_payload", original_deserializer)

    def assert_rpc_is_blocked(*args, **kwargs) -> None:
        with pytest.raises(RuntimeError, match="DistDGL RPC deserialization is disabled"):
            rpc.deserialize_from_payload(object, b"untrusted payload", [])

    monkeypatch.setattr(LocalRetroModel, "__init__", assert_rpc_is_blocked)
    TemplateAwareLocalRetroModel()


def test_local_dgl_graph_operations_remain_available() -> None:
    first_graph = dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=2)
    second_graph = dgl.graph((torch.tensor([1]), torch.tensor([0])), num_nodes=2)
    batch = dgl.batch([first_graph, second_graph])

    assert batch.batch_size == 2
    assert batch.num_nodes() == 4
    assert decoder.__package__ == "local_retro.LocalTemplate"


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
