from dataclasses import dataclass
from typing import Final

NODE_SCOPE_FRAGMENT: Final = "fragment"
NODE_SCOPE_LIGAND: Final = "ligand"
CONDITION_POCKET: Final = "pocket"
CONDITION_FRAGMENT_MOL: Final = "fragment_mol"
FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL: Final = "fragment_mol"

EXPERT_EDM_QM9: Final = "EDM_QM9"
EXPERT_EDM_GEOM_DRUG: Final = "EDM_GEOM_DRUG"
EXPERT_GEODIFF_QM9: Final = "GEODIFF_QM9"
EXPERT_DIFFSBDD_CROSSDOCKED_FULLATOM_COND: Final = "DIFFSBDD_CROSSDOCKED_FULLATOM_COND"

SCHEDULER_EDM: Final = "EDM"
SCHEDULER_GEODIFF: Final = "GEODIFF"
SCHEDULER_DIFFSBDD: Final = "DIFFSBDD"


@dataclass
class MoEComponentConfig:
    """Structured config for one expert component in a shared MoE latent.

    Properties
    -----------
    name: str
        Unique name for this component, used for logging and config selection.
    expert_key: str
        Key used to select the expert wrapper and pretrained weights.
    scheduler_key: str
        Key used to select the scheduler paired with this expert path.
    node_scope: str
        The node subset that this component acts on, such as fragment nodes or full ligand nodes.
    supported_atoms: tuple[str, ...]
        The atom types that this component can generate or score.
    condition_keys: tuple[str, ...]
        External inputs required by the component; an empty tuple means ligand-only generation,
        while values such as "pocket" or "fragment_mol" request prepared runtime conditions.
    fixed_atom_type_source: str | None
        Used when atom-type channels are fixed from a condition instead of sampled by the component itself.
    scored_atoms: tuple[str, ...]
        Atom-type channels passed to the expert score model. If empty, all supported_atoms are scored.
    score_coordinates: bool
        Whether this component scores the coordinates of the nodes it acts on.
    score_atom_types: bool
        Whether this component scores the atom types of the nodes it acts on.
    uses_nuclear_charge_feature: bool
        Whether this component uses the nuclear charge feature in its input.
    score_nuclear_charge_feature: bool
        Whether this component scores the nuclear charge feature of the nodes it acts on.
    """

    name: str
    expert_key: str
    scheduler_key: str
    node_scope: str
    supported_atoms: tuple[str, ...]
    condition_keys: tuple[str, ...] = ()
    fixed_atom_type_source: str | None = None
    scored_atoms: tuple[str, ...] = ()
    score_coordinates: bool = True
    score_atom_types: bool = True
    uses_nuclear_charge_feature: bool = False
    score_nuclear_charge_feature: bool = False


EDM_QM9_ACTIVE_ATOMS: Final = ("H", "C", "N", "O", "F")
EDM_GEOM_DRUG_ACTIVE_ATOMS: Final = ("B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi")
DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS: Final = ("C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F", "others")
GEODIFF_QM9_ACTIVE_ATOMS: Final = ("H", "C", "N", "O", "F")


EDM_QM9_FRAGMENT: Final = MoEComponentConfig(
    name="EDM_QM9_FRAGMENT",
    expert_key=EXPERT_EDM_QM9,
    scheduler_key=SCHEDULER_EDM,
    node_scope=NODE_SCOPE_FRAGMENT,
    supported_atoms=EDM_QM9_ACTIVE_ATOMS,
    uses_nuclear_charge_feature=True,
    score_nuclear_charge_feature=True,
)

EDM_QM9_LIGAND: Final = MoEComponentConfig(
    name="EDM_QM9_LIGAND",
    expert_key=EXPERT_EDM_QM9,
    scheduler_key=SCHEDULER_EDM,
    node_scope=NODE_SCOPE_LIGAND,
    supported_atoms=EDM_QM9_ACTIVE_ATOMS,
    uses_nuclear_charge_feature=True,
    score_nuclear_charge_feature=True,
)

EDM_GEOM_DRUG_LIGAND: Final = MoEComponentConfig(
    name="EDM_GEOM_DRUG_LIGAND",
    expert_key=EXPERT_EDM_GEOM_DRUG,
    scheduler_key=SCHEDULER_EDM,
    node_scope=NODE_SCOPE_LIGAND,
    supported_atoms=EDM_GEOM_DRUG_ACTIVE_ATOMS,
)

GEODIFF_QM9_FRAGMENT: Final = MoEComponentConfig(
    name="GEODIFF_QM9_FRAGMENT",
    expert_key=EXPERT_GEODIFF_QM9,
    scheduler_key=SCHEDULER_GEODIFF,
    node_scope=NODE_SCOPE_FRAGMENT,
    supported_atoms=GEODIFF_QM9_ACTIVE_ATOMS,
    condition_keys=(CONDITION_FRAGMENT_MOL,),
    fixed_atom_type_source=FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL,
    score_atom_types=False,
)

DIFFSBDD_CROSSDOCKED_FULLATOM_COND: Final = MoEComponentConfig(
    name="DIFFSBDD_CROSSDOCKED_FULLATOM_COND",
    expert_key=EXPERT_DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
    scheduler_key=SCHEDULER_DIFFSBDD,
    node_scope=NODE_SCOPE_LIGAND,
    supported_atoms=DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS,
    condition_keys=(CONDITION_POCKET,),
    scored_atoms=DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS[:-1],
)

EDM_GEOM_DRUG: Final = EDM_GEOM_DRUG_LIGAND
