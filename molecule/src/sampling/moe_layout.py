from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import torch
from jaxtyping import Bool

DataMask = Bool[torch.Tensor, "data"]  # noqa: F821

COORDS_DIM: Final = 3
CANONICAL_ATOM_TYPE_ORDER: Final = (
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "H",
    "Si",
    "Al",
    "As",
    "Hg",
    "Bi",
    "others",
)
_CANONICAL_ATOM_TYPE_RANK: Final = {atom: rank for rank, atom in enumerate(CANONICAL_ATOM_TYPE_ORDER)}


def _ordered_atom_types(atoms: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            atoms,
            key=lambda atom: (_CANONICAL_ATOM_TYPE_RANK.get(atom, len(CANONICAL_ATOM_TYPE_ORDER)), atom),
        )
    )


class MoEComponentLayoutSpec(Protocol):
    """Minimum component metadata needed to allocate columns in a shared MoE latent."""

    name: str
    node_scope: str
    supported_atoms: tuple[str, ...]
    scored_atoms: tuple[str, ...]
    score_coordinates: bool
    score_atom_types: bool
    uses_nuclear_charge_feature: bool
    score_nuclear_charge_feature: bool


@dataclass(frozen=True)
class DynamicMoEAtomLayout:
    """Atom-feature layout derived from the expert components used in one MoE run."""

    atom_type_index: dict[str, int]
    coords_dim: int = COORDS_DIM
    include_nuclear_charge_feature: bool = True

    @classmethod
    def from_components(
        cls,
        components: Sequence[MoEComponentLayoutSpec],
        *,
        include_nuclear_charge_feature: bool | None = None,
    ) -> "DynamicMoEAtomLayout":
        if not components:
            raise ValueError("At least one MoE component is required to build a dynamic layout.")

        atoms = _ordered_atom_types({atom for component in components for atom in component.supported_atoms})
        if not atoms:
            raise ValueError("MoE components must expose at least one supported atom.")

        if include_nuclear_charge_feature is None:
            include_nuclear_charge_feature = any(component.uses_nuclear_charge_feature for component in components)

        return cls(
            atom_type_index={atom: idx for idx, atom in enumerate(atoms)},
            include_nuclear_charge_feature=include_nuclear_charge_feature,
        )

    @property
    def atom_type_dim(self) -> int:
        return len(self.atom_type_index)

    @property
    def atom_type_decoder(self) -> dict[int, str]:
        return {index: atom for atom, index in self.atom_type_index.items()}

    @property
    def node_feature_dim(self) -> int:
        return self.coords_dim + self.atom_type_dim + int(self.include_nuclear_charge_feature)

    @property
    def nuclear_charge_feature_column(self) -> int:
        if not self.include_nuclear_charge_feature:
            raise ValueError("This layout does not include an EDM nuclear-charge feature column.")
        return self.node_feature_dim - 1

    def coord_columns(self) -> tuple[int, ...]:
        return tuple(range(self.coords_dim))

    def atom_type_column(self, atom: str) -> int:
        try:
            return self.coords_dim + self.atom_type_index[atom]
        except KeyError as exc:
            supported = ", ".join(self.atom_type_index)
            raise ValueError(f"Atom {atom!r} is not in this MoE layout. Supported atoms: {supported}.") from exc

    def atom_type_columns(self, atoms: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.atom_type_column(atom) for atom in atoms)

    def component_active_columns(self, component: MoEComponentLayoutSpec) -> tuple[int, ...]:
        columns: list[int] = []
        if component.score_coordinates:
            columns.extend(self.coord_columns())
        if component.score_atom_types:
            scored_atoms = component.scored_atoms or component.supported_atoms
            columns.extend(self.atom_type_columns(scored_atoms))
        if component.score_nuclear_charge_feature:
            columns.append(self.nuclear_charge_feature_column)
        return tuple(sorted(set(columns)))

    def component_padding_columns(self, component: MoEComponentLayoutSpec) -> tuple[int, ...]:
        active_columns = set(self.component_active_columns(component))
        return tuple(column for column in range(self.node_feature_dim) if column not in active_columns)

    def node_mask(
        self,
        num_nodes: int,
        columns: Sequence[int],
        *,
        device: torch.device | str = "cpu",
    ) -> DataMask:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")

        mask = torch.zeros(num_nodes, self.node_feature_dim, dtype=torch.bool, device=device)
        mask[:, list(columns)] = True
        return mask.flatten()


@dataclass(frozen=True)
class DynamicMoELayout:
    """Node and feature masks for a component-defined MoE latent."""

    components: tuple[MoEComponentLayoutSpec, ...]
    fragment_size: int
    ligand_size: int
    atom_layout: DynamicMoEAtomLayout
    device: torch.device | str = "cpu"

    @classmethod
    def from_components(
        cls,
        components: Sequence[MoEComponentLayoutSpec],
        *,
        fragment_size: int,
        ligand_size: int,
        device: torch.device | str = "cpu",
    ) -> "DynamicMoELayout":
        return cls(
            components=tuple(components),
            fragment_size=fragment_size,
            ligand_size=ligand_size,
            atom_layout=DynamicMoEAtomLayout.from_components(components),
            device=device,
        )

    def __post_init__(self) -> None:
        if self.fragment_size <= 0:
            raise ValueError("fragment_size must be positive.")
        if self.ligand_size <= 0:
            raise ValueError("ligand_size must be positive.")
        if self.fragment_size > self.ligand_size:
            raise ValueError("fragment_size cannot exceed ligand_size.")

    @property
    def sample_size(self) -> int:
        return self.ligand_size * self.atom_layout.node_feature_dim

    @property
    def node_feature_dim(self) -> int:
        return self.atom_layout.node_feature_dim

    @property
    def atom_type_index(self) -> dict[str, int]:
        return self.atom_layout.atom_type_index

    @property
    def atom_type_decoder(self) -> dict[int, str]:
        return self.atom_layout.atom_type_decoder

    @property
    def atom_type_dim(self) -> int:
        return self.atom_layout.atom_type_dim

    def num_nodes_for_scope(self, node_scope: str) -> int:
        if node_scope == "fragment":
            return self.fragment_size
        if node_scope == "ligand":
            return self.ligand_size
        raise ValueError(f"Unsupported node scope {node_scope!r}.")

    def state_mask_for_scope(self, node_scope: str) -> DataMask:
        if node_scope == "fragment":
            mask = torch.zeros(self.ligand_size, self.node_feature_dim, dtype=torch.bool, device=self.device)
            mask[: self.fragment_size, :] = True
            return mask.flatten()
        if node_scope == "ligand":
            return torch.ones(self.ligand_size, self.node_feature_dim, dtype=torch.bool, device=self.device).flatten()
        raise ValueError(f"Unsupported node scope {node_scope!r}.")

    def active_mask_for_component(self, component: MoEComponentLayoutSpec) -> DataMask:
        return self.atom_layout.node_mask(
            self.num_nodes_for_scope(component.node_scope),
            self.atom_layout.component_active_columns(component),
            device=self.device,
        )

    def auxiliary_mask_for_component(self, component: MoEComponentLayoutSpec) -> DataMask:
        active_mask = self.active_mask_for_component(component)
        return ~active_mask

    def h_atom_type_mask_for_scope(self, node_scope: str) -> DataMask | None:
        if "H" not in self.atom_type_index:
            return None
        return self.atom_layout.node_mask(
            self.num_nodes_for_scope(node_scope),
            (self.atom_layout.atom_type_column("H"),),
            device=self.device,
        )
