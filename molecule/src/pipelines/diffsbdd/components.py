from typing import Final

from configs.config_moe_component import CONDITION_POCKET, NODE_SCOPE_LIGAND, MoEComponentConfig

EXPERT_DIFFSBDD_CROSSDOCKED_FULLATOM_COND: Final = "DIFFSBDD_CROSSDOCKED_FULLATOM_COND"
SCHEDULER_DIFFSBDD: Final = "DIFFSBDD"

DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS: Final = ("C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F", "others")

DIFFSBDD_CROSSDOCKED_FULLATOM_COND: Final = MoEComponentConfig(
    name="DIFFSBDD_CROSSDOCKED_FULLATOM_COND",
    expert_key=EXPERT_DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
    scheduler_key=SCHEDULER_DIFFSBDD,
    node_scope=NODE_SCOPE_LIGAND,
    supported_atoms=DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS,
    condition_keys=(CONDITION_POCKET,),
    fixed_atom_type_source=None,
    scored_atoms=DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS[:-1],
    score_coordinates=True,
    score_atom_types=True,
    uses_nuclear_charge_feature=False,
    score_nuclear_charge_feature=False,
)
