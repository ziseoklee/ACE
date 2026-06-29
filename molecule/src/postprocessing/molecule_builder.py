from typing import Any

import torch
from diffsbdd.analysis.molecule_builder import build_molecule, process_molecule
from jaxtyping import Float
from rdkit import Chem


class MoleculeBuilder:
    """Build RDKit molecules from generated coordinates and atom types."""

    @classmethod
    def build_batch(
        cls,
        xh: Float[torch.Tensor, "B L coords+atom_types"],
        dataset_info: dict[str, Any],
        fragment_atom_types: torch.Tensor | None = None,
        x_dims: int = 3,
        add_coords: bool = True,
        add_hydrogens: bool = False,
        sanitize: bool = False,
        relax_iter: int = 0,
        largest_frag: bool = False,
    ) -> list[Chem.Mol | None]:
        molecules: list[Chem.Mol | None] = []
        coordinates = xh[..., :x_dims].detach().cpu()
        atom_types = xh[..., x_dims:].argmax(-1).detach().cpu()

        for coordinates_sample, atom_types_sample in zip(coordinates, atom_types):
            atom_types_sample = atom_types_sample.clone()
            if fragment_atom_types is not None:
                atom_types_sample[: len(fragment_atom_types)] = fragment_atom_types.detach().cpu()

            mol = build_molecule(coordinates_sample, atom_types_sample, dataset_info, add_coords=add_coords)
            if mol is None:
                molecules.append(None)
                continue

            mol = process_molecule(
                mol,
                add_hydrogens=add_hydrogens,
                sanitize=sanitize,
                relax_iter=relax_iter,
                largest_frag=largest_frag,
            )
            if mol is not None:
                molecules.append(mol)

        return molecules

    @staticmethod
    def build_one(
        xh: Float[torch.Tensor, "B L coords+atom_types"],
        dataset_info: dict[str, Any],
        x_dims: int = 3,
        add_coords: bool = True,
        add_hydrogens: bool = False,
        sanitize: bool = False,
        relax_iter: int = 0,
        largest_frag: bool = False,
    ) -> Chem.Mol | None:
        coordinates = xh[..., :x_dims].detach().cpu()
        atom_types = xh[..., x_dims:].argmax(-1).detach().cpu()
        mol = build_molecule(coordinates, atom_types, dataset_info, add_coords=add_coords)
        if mol is None:
            return None

        return process_molecule(
            mol,
            add_hydrogens=add_hydrogens,
            sanitize=sanitize,
            relax_iter=relax_iter,
            largest_frag=largest_frag,
        )
