from typing import Final

from configs.config_moe_component import (
    CONDITION_FRAGMENT_MOL,
    FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL,
    NODE_SCOPE_FRAGMENT,
    MoEComponentConfig,
)

EXPERT_GEODIFF_QM9: Final = "GEODIFF_QM9"
SCHEDULER_GEODIFF: Final = "GEODIFF"

GEODIFF_QM9_ACTIVE_ATOMS: Final = ("H", "C", "N", "O", "F")

GEODIFF_QM9_FRAGMENT: Final = MoEComponentConfig(
    name="GEODIFF_QM9_FRAGMENT",
    expert_key=EXPERT_GEODIFF_QM9,
    scheduler_key=SCHEDULER_GEODIFF,
    node_scope=NODE_SCOPE_FRAGMENT,
    supported_atoms=GEODIFF_QM9_ACTIVE_ATOMS,
    condition_keys=(CONDITION_FRAGMENT_MOL,),
    fixed_atom_type_source=FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL,
    scored_atoms=(),
    score_coordinates=True,
    score_atom_types=False,
    uses_nuclear_charge_feature=False,
    score_nuclear_charge_feature=False,
)
