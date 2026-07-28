from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

SERIALIZATION_PATH = Path(__file__).parents[1] / "runtime" / "syntheseus" / "serialization.py"
SPEC = spec_from_file_location("pandora_syntheseus_serialization", SERIALIZATION_PATH)
assert SPEC is not None and SPEC.loader is not None
SERIALIZATION = module_from_spec(SPEC)
SPEC.loader.exec_module(SERIALIZATION)
SyntheseusSerializationError = SERIALIZATION.SyntheseusSerializationError
serialize_route = SERIALIZATION.serialize_route


@dataclass(frozen=True)
class Molecule:
    smiles: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reaction:
    product: Molecule
    reactants: tuple[Molecule, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reaction_smiles(self) -> str:
        return f"{'.'.join(molecule.smiles for molecule in self.reactants)}>>{self.product.smiles}"


@dataclass(frozen=True, eq=False)
class MoleculeNode:
    mol: Molecule


@dataclass(frozen=True, eq=False)
class ReactionNode:
    reaction: Reaction


@dataclass
class Graph:
    root_node: MoleculeNode
    edges: dict[int, list[Any]]

    def successors(self, node: Any) -> list[Any]:
        return self.edges.get(id(node), [])


def test_serialize_route_preserves_reaction_provenance() -> None:
    target = Molecule("CCO")
    methane = Molecule("C", {"is_purchasable": True})
    ethane = Molecule("CC", {"is_purchasable": True})
    reaction = Reaction(
        product=target,
        reactants=(methane, ethane),
        metadata={
            "probability": 0.8,
            "template": "[C:1]-[O:2]>>[C:1].[O:2]",
            "local_retro_template": "[C:1]-[O:2]>>[C:1].[O:2]_10_00_00",
        },
    )
    target_node = MoleculeNode(target)
    methane_node = MoleculeNode(methane)
    ethane_node = MoleculeNode(ethane)
    reaction_node = ReactionNode(reaction)
    nodes = [target_node, methane_node, ethane_node, reaction_node]
    graph = Graph(
        root_node=target_node,
        edges={
            id(target_node): [reaction_node],
            id(reaction_node): [methane_node, ethane_node],
        },
    )

    serialized = serialize_route(graph, nodes, target.smiles)

    serialized_reaction = serialized["children"][0]
    assert serialized_reaction["smiles"] == "C.CC>>CCO"
    assert serialized_reaction["metadata"] == reaction.metadata
    assert [child["smiles"] for child in serialized_reaction["children"]] == ["C", "CC"]
    assert all(child["in_stock"] for child in serialized_reaction["children"])


def test_serialize_route_rejects_missing_target() -> None:
    target_node = MoleculeNode(Molecule("CCO"))
    with pytest.raises(SyntheseusSerializationError, match="Target molecule"):
        serialize_route(Graph(target_node, {}), [], "CCO")


def test_serialize_route_rejects_cycles() -> None:
    target = Molecule("CCO")
    reaction = Reaction(product=target, reactants=(target,))
    target_node = MoleculeNode(target)
    duplicate_target_node = MoleculeNode(target)
    reaction_node = ReactionNode(reaction)
    nodes = [target_node, duplicate_target_node, reaction_node]
    graph = Graph(
        target_node,
        {
            id(target_node): [reaction_node],
            id(reaction_node): [duplicate_target_node],
            id(duplicate_target_node): [reaction_node],
        },
    )

    with pytest.raises(SyntheseusSerializationError, match="Cycle"):
        serialize_route(graph, nodes, target.smiles)


def test_serialize_route_preserves_duplicate_molecule_occurrences() -> None:
    target = Molecule("CCO")
    left_branch = Molecule("CC")
    right_branch = Molecule("O")
    duplicate = Molecule("C", {"is_purchasable": True})
    root_reaction = Reaction(product=target, reactants=(left_branch, right_branch))
    left_reaction = Reaction(product=left_branch, reactants=(duplicate,))
    right_reaction = Reaction(product=right_branch, reactants=(duplicate,))

    target_node = MoleculeNode(target)
    left_branch_node = MoleculeNode(left_branch)
    right_branch_node = MoleculeNode(right_branch)
    first_duplicate_node = MoleculeNode(duplicate)
    second_duplicate_node = MoleculeNode(duplicate)
    root_reaction_node = ReactionNode(root_reaction)
    left_reaction_node = ReactionNode(left_reaction)
    right_reaction_node = ReactionNode(right_reaction)
    nodes = [
        target_node,
        root_reaction_node,
        left_branch_node,
        right_branch_node,
        left_reaction_node,
        right_reaction_node,
        first_duplicate_node,
        second_duplicate_node,
    ]
    graph = Graph(
        target_node,
        {
            id(target_node): [root_reaction_node],
            id(root_reaction_node): [left_branch_node, right_branch_node],
            id(left_branch_node): [left_reaction_node],
            id(right_branch_node): [right_reaction_node],
            id(left_reaction_node): [first_duplicate_node],
            id(right_reaction_node): [second_duplicate_node],
        },
    )

    serialized = serialize_route(graph, nodes, target.smiles)

    branch_nodes = serialized["children"][0]["children"]
    duplicate_nodes = [branch["children"][0]["children"][0] for branch in branch_nodes]
    assert len(duplicate_nodes) == 2
    assert [node["smiles"] for node in duplicate_nodes] == ["C", "C"]
    assert all(node["children"] == [] for node in duplicate_nodes)
