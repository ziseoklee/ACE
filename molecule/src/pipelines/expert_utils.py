from __future__ import annotations

from typing import Protocol

import torch
from rdkit.Chem import Mol

from configs.config_moe_component import (
    CONDITION_FRAGMENT_MOL,
    FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL,
    MoEComponentConfig,
)
from sampling.moe_layout import COORDS_DIM, DynamicMoELayout


class FragmentCondition(Protocol):
    fragment: Mol


def mol_atom_type_indices(mol: Mol, layout: DynamicMoELayout, device: str) -> torch.Tensor:
    atom_type_indices = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in layout.atom_type_index:
            supported_atoms = ", ".join(layout.atom_type_index)
            raise ValueError(f"Unsupported atom type {symbol!r}. Supported atom types: {supported_atoms}.")
        atom_type_indices.append(layout.atom_type_index[symbol])

    return torch.tensor(atom_type_indices, device=device, dtype=torch.long)


def mol_atom_feature_point(
    mol: Mol,
    layout: DynamicMoELayout,
    auxiliary_mask: torch.Tensor,
    device: str,
    atom_type_value: float,
) -> torch.Tensor:
    point = torch.zeros(mol.GetNumAtoms(), layout.node_feature_dim, device=device)
    atom_type_indices = mol_atom_type_indices(mol, layout=layout, device=device)
    point[
        torch.arange(mol.GetNumAtoms(), device=device),
        COORDS_DIM + atom_type_indices,
    ] = atom_type_value

    # EDM's extra scalar is nuclear charge, not formal charge. Molecule construction
    # uses atom symbols, so this auxiliary feature remains zero-padded.
    return point.flatten()[auxiliary_mask].reshape(mol.GetNumAtoms(), -1)


def condition_mol_for_source(component: MoEComponentConfig, condition: FragmentCondition, source: str) -> Mol:
    if source == FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL:
        require_condition_key(component, CONDITION_FRAGMENT_MOL)
        return condition.fragment

    raise ValueError(f"Unsupported fixed atom-type source {source!r} for component {component.name}.")


def validate_condition_mol_size(component: MoEComponentConfig, mol: Mol, expected_num_nodes: int) -> None:
    actual_num_nodes = mol.GetNumAtoms()
    if actual_num_nodes != expected_num_nodes:
        raise ValueError(
            f"Component {component.name} uses node_scope={component.node_scope!r} with {expected_num_nodes} nodes, "
            f"but fixed atom/topology source {component.fixed_atom_type_source!r} has {actual_num_nodes} atoms."
        )


def require_condition_key(component: MoEComponentConfig, condition_key: str) -> None:
    if condition_key not in component.condition_keys:
        raise ValueError(
            f"Component {component.name} requires condition key {condition_key!r}, "
            f"but its condition_keys are {component.condition_keys}."
        )
