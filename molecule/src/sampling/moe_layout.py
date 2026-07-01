from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import torch
from jaxtyping import Bool

DataMask = Bool[torch.Tensor, "data"]  # noqa: F821

COORDS_DIM: Final = 3
ATOM_TYPE_INDEX: Final = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "P": 7,
    "I": 8,
    "F": 9,
    "H": 10,
}
ATOM_TYPE_DIM: Final = len(ATOM_TYPE_INDEX)
NUCLEAR_CHARGE_FEATURE_DIM: Final = 1
NODE_FEATURE_DIM: Final = COORDS_DIM + ATOM_TYPE_DIM + NUCLEAR_CHARGE_FEATURE_DIM

H_ATOM_TYPE_COLUMN: Final = COORDS_DIM + ATOM_TYPE_INDEX["H"]
NUCLEAR_CHARGE_FEATURE_COLUMN: Final = NODE_FEATURE_DIM - 1

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

_DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS: Final = ("C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F")
_DIFFSBDD_CROSSDOCKED_ACTIVE_COLUMNS: Final = tuple(range(COORDS_DIM)) + tuple(
    COORDS_DIM + ATOM_TYPE_INDEX[atom] for atom in _DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS
)
_DIFFSBDD_CROSSDOCKED_PADDING_COLUMNS: Final = (
    COORDS_DIM + ATOM_TYPE_INDEX["H"],
    NUCLEAR_CHARGE_FEATURE_COLUMN,
)

_EDM_QM9_ACTIVE_ATOMS: Final = ("H", "C", "N", "O", "F")
_EDM_QM9_ATOM_INDEX_TO_GLOBAL_ATOM_INDEX: Final = {
    0: ATOM_TYPE_INDEX["H"],
    1: ATOM_TYPE_INDEX["C"],
    2: ATOM_TYPE_INDEX["N"],
    3: ATOM_TYPE_INDEX["O"],
    4: ATOM_TYPE_INDEX["F"],
}
_EDM_QM9_ACTIVE_COLUMNS: Final = (
    tuple(range(COORDS_DIM))
    + tuple(COORDS_DIM + atom_index for atom_index in _EDM_QM9_ATOM_INDEX_TO_GLOBAL_ATOM_INDEX.values())
    + (NUCLEAR_CHARGE_FEATURE_COLUMN,)
)
_EDM_QM9_PADDING_COLUMNS: Final = tuple(i for i in range(NODE_FEATURE_DIM) if i not in _EDM_QM9_ACTIVE_COLUMNS)

_GEODIFF_QM9_COORD_COLUMNS: Final = tuple(range(COORDS_DIM))
_GEODIFF_QM9_AUXILIARY_FEATURE_COLUMNS: Final = tuple(range(COORDS_DIM, NODE_FEATURE_DIM))


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
    supported_atoms: tuple[str, ...]
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
            columns.extend(self.atom_type_columns(component.supported_atoms))
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
class CrossDockedMoEMasks:
    """Boolean masks describing how each expert views the shared MoE latent."""

    geodiff_fragment_coords: DataMask
    geodiff_fragment_atom_features: DataMask
    edm_fragment_xh: DataMask
    edm_fragment_padding: DataMask
    diffsbdd_ligand_xh: DataMask
    diffsbdd_ligand_padding: DataMask
    edm_ligand_xh: DataMask
    edm_ligand_h_atom_type: DataMask
    edm_ligand_padding: DataMask
    fragment_state_in_ligand: DataMask
    ligand_state: DataMask

    @property
    def sample_size(self) -> int:
        return self.ligand_state.numel()


@dataclass(frozen=True)
class CrossDockedMoELayout:
    """Shared latent layout for CrossDocked MoE inference."""

    fragment_size: int
    ligand_size: int
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        if self.fragment_size <= 0:
            raise ValueError("fragment_size must be positive.")
        if self.ligand_size <= 0:
            raise ValueError("ligand_size must be positive.")
        if self.fragment_size > self.ligand_size:
            raise ValueError("fragment_size cannot exceed ligand_size.")

    def masks(self) -> CrossDockedMoEMasks:
        geodiff_fragment_coords = self._node_mask(self.fragment_size, _GEODIFF_QM9_COORD_COLUMNS)
        geodiff_fragment_atom_features = self._node_mask(
            self.fragment_size,
            _GEODIFF_QM9_AUXILIARY_FEATURE_COLUMNS,
        )

        edm_fragment_xh = self._node_mask(self.fragment_size, _EDM_QM9_ACTIVE_COLUMNS)
        edm_fragment_padding = ~edm_fragment_xh

        diffsbdd_ligand_xh = self._node_mask(self.ligand_size, _DIFFSBDD_CROSSDOCKED_ACTIVE_COLUMNS)
        diffsbdd_ligand_padding = self._node_mask(self.ligand_size, _DIFFSBDD_CROSSDOCKED_PADDING_COLUMNS)

        edm_ligand_xh = self._node_mask(self.ligand_size, _EDM_QM9_ACTIVE_COLUMNS)
        edm_ligand_h_atom_type = self._node_mask(self.ligand_size, (H_ATOM_TYPE_COLUMN,))
        edm_ligand_padding = ~edm_ligand_xh

        fragment_state_in_ligand = torch.zeros(self.ligand_size, NODE_FEATURE_DIM, dtype=torch.bool, device=self.device)
        fragment_state_in_ligand[: self.fragment_size, :] = True
        ligand_state = torch.ones(self.ligand_size, NODE_FEATURE_DIM, dtype=torch.bool, device=self.device)

        return CrossDockedMoEMasks(
            geodiff_fragment_coords=geodiff_fragment_coords,
            geodiff_fragment_atom_features=geodiff_fragment_atom_features,
            edm_fragment_xh=edm_fragment_xh,
            edm_fragment_padding=edm_fragment_padding,
            diffsbdd_ligand_xh=diffsbdd_ligand_xh,
            diffsbdd_ligand_padding=diffsbdd_ligand_padding,
            edm_ligand_xh=edm_ligand_xh,
            edm_ligand_h_atom_type=edm_ligand_h_atom_type,
            edm_ligand_padding=edm_ligand_padding,
            fragment_state_in_ligand=fragment_state_in_ligand.flatten(),
            ligand_state=ligand_state.flatten(),
        )

    def _node_mask(self, num_nodes: int, columns: range | tuple[int, ...]) -> DataMask:
        mask = torch.zeros(num_nodes, NODE_FEATURE_DIM, dtype=torch.bool, device=self.device)
        mask[:, list(columns)] = True
        return mask.flatten()
