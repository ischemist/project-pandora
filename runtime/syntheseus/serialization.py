"""Serialize Syntheseus route nodes into Pandora's documented raw tree format."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any


class SyntheseusSerializationError(ValueError):
    """Raised when a Syntheseus route cannot be represented as a raw route tree."""


def _build_tree(
    graph: Any,
    molecule_node: Any,
    route_nodes: Collection[Any],
    ancestors: frozenset[int],
) -> dict[str, Any]:
    smiles = molecule_node.mol.smiles
    node_identity = id(molecule_node)
    if node_identity in ancestors:
        raise SyntheseusSerializationError(f"Cycle found while expanding molecule {smiles!r}")

    is_purchasable = bool(molecule_node.mol.metadata.get("is_purchasable", False))
    molecule = {
        "smiles": smiles,
        "type": "mol",
        "in_stock": is_purchasable,
        "children": [],
    }
    reaction_nodes = [node for node in graph.successors(molecule_node) if node in route_nodes]
    if is_purchasable or not reaction_nodes:
        return molecule
    if len(reaction_nodes) != 1:
        raise SyntheseusSerializationError(
            f"Route molecule {smiles!r} has {len(reaction_nodes)} selected reactions; expected exactly one"
        )

    reaction_node = reaction_nodes[0]
    reaction = reaction_node.reaction
    reaction_tree: dict[str, Any] = {
        "smiles": reaction.reaction_smiles,
        "type": "reaction",
        "children": [],
    }
    metadata = dict(reaction.metadata)
    if metadata:
        reaction_tree["metadata"] = metadata

    next_ancestors = ancestors | {node_identity}
    reactant_nodes = [node for node in graph.successors(reaction_node) if node in route_nodes]
    if not reactant_nodes:
        raise SyntheseusSerializationError(f"Reaction {reaction.reaction_smiles!r} has no reactant nodes in the route")
    reaction_tree["children"] = [
        _build_tree(graph, reactant_node, route_nodes, next_ancestors) for reactant_node in reactant_nodes
    ]
    molecule["children"].append(reaction_tree)
    return molecule


def serialize_route(graph: Any, route_nodes: Iterable[Any], target_smiles: str) -> dict[str, Any]:
    """Convert one extracted Syntheseus route to the raw bipartite tree consumed by RetroCast."""
    nodes = set(route_nodes)
    root_node = graph.root_node
    if root_node not in nodes or not hasattr(root_node, "mol"):
        raise SyntheseusSerializationError(f"Target molecule {target_smiles!r} is missing from the route")
    return _build_tree(graph, root_node, nodes, frozenset())
