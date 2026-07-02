from typing import Final

from configs.config_moe_component import NODE_SCOPE_FRAGMENT, NODE_SCOPE_LIGAND, MoEComponentConfig

EXPERT_EDM_QM9: Final = "EDM_QM9"
EXPERT_EDM_GEOM_DRUG: Final = "EDM_GEOM_DRUG"
SCHEDULER_EDM: Final = "EDM"

EDM_QM9_ACTIVE_ATOMS: Final = ("H", "C", "N", "O", "F")
EDM_GEOM_DRUG_ACTIVE_ATOMS: Final = ("B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi")

EDM_QM9_FRAGMENT: Final = MoEComponentConfig(
    name="EDM_QM9_FRAGMENT",
    expert_key=EXPERT_EDM_QM9,
    scheduler_key=SCHEDULER_EDM,
    node_scope=NODE_SCOPE_FRAGMENT,
    supported_atoms=EDM_QM9_ACTIVE_ATOMS,
    condition_keys=(),
    fixed_atom_type_source=None,
    scored_atoms=(),
    score_coordinates=True,
    score_atom_types=True,
    uses_nuclear_charge_feature=True,
    score_nuclear_charge_feature=True,
)

EDM_QM9_LIGAND: Final = MoEComponentConfig(
    name="EDM_QM9_LIGAND",
    expert_key=EXPERT_EDM_QM9,
    scheduler_key=SCHEDULER_EDM,
    node_scope=NODE_SCOPE_LIGAND,
    supported_atoms=EDM_QM9_ACTIVE_ATOMS,
    condition_keys=(),
    fixed_atom_type_source=None,
    scored_atoms=(),
    score_coordinates=True,
    score_atom_types=True,
    uses_nuclear_charge_feature=True,
    score_nuclear_charge_feature=True,
)

# TODO: update this config. Currenet EDM_GEOM_DRUG_LIGAND config is a placeholder and not yet fully implemented.
EDM_GEOM_DRUG_LIGAND: Final = MoEComponentConfig(
    name="EDM_GEOM_DRUG_LIGAND",
    expert_key=EXPERT_EDM_GEOM_DRUG,
    scheduler_key=SCHEDULER_EDM,
    node_scope=NODE_SCOPE_LIGAND,
    supported_atoms=EDM_GEOM_DRUG_ACTIVE_ATOMS,
)
