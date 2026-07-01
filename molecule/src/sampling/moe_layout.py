from dataclasses import dataclass
from typing import Final

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
