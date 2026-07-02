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

SUPPORTED_EXPERT_KEYS: Final = (
    EXPERT_EDM_QM9,
    EXPERT_EDM_GEOM_DRUG,
    EXPERT_GEODIFF_QM9,
    EXPERT_DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
)
SUPPORTED_SCHEDULER_KEYS: Final = (
    SCHEDULER_EDM,
    SCHEDULER_GEODIFF,
    SCHEDULER_DIFFSBDD,
)
SUPPORTED_NODE_SCOPES: Final = (
    NODE_SCOPE_FRAGMENT,
    NODE_SCOPE_LIGAND,
)
SUPPORTED_CONDITION_KEYS: Final = (
    CONDITION_POCKET,
    CONDITION_FRAGMENT_MOL,
)
SUPPORTED_FIXED_ATOM_TYPE_SOURCES: Final = (
    FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL,
)


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

    def __post_init__(self) -> None:
        self.supported_atoms = tuple(self.supported_atoms)
        self.condition_keys = tuple(self.condition_keys)
        self.scored_atoms = tuple(self.scored_atoms)

        if not self.name:
            raise ValueError("MoE component name must be non-empty.")
        if self.expert_key not in SUPPORTED_EXPERT_KEYS:
            raise ValueError(f"Unsupported expert_key {self.expert_key!r} for component {self.name}.")
        if self.scheduler_key not in SUPPORTED_SCHEDULER_KEYS:
            raise ValueError(f"Unsupported scheduler_key {self.scheduler_key!r} for component {self.name}.")
        if self.node_scope not in SUPPORTED_NODE_SCOPES:
            raise ValueError(f"Unsupported node_scope {self.node_scope!r} for component {self.name}.")
        if not self.supported_atoms:
            raise ValueError(f"Component {self.name} must define at least one supported atom.")

        unsupported_condition_keys = set(self.condition_keys) - set(SUPPORTED_CONDITION_KEYS)
        if unsupported_condition_keys:
            raise ValueError(
                f"Component {self.name} uses unsupported condition_keys: {sorted(unsupported_condition_keys)}."
            )

        if self.fixed_atom_type_source is not None:
            if self.fixed_atom_type_source not in SUPPORTED_FIXED_ATOM_TYPE_SOURCES:
                raise ValueError(
                    f"Component {self.name} uses unsupported fixed_atom_type_source "
                    f"{self.fixed_atom_type_source!r}."
                )
            if self.fixed_atom_type_source == FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL:
                if CONDITION_FRAGMENT_MOL not in self.condition_keys:
                    raise ValueError(
                        f"Component {self.name} fixes atom types from {FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL!r} "
                        f"but does not declare condition key {CONDITION_FRAGMENT_MOL!r}."
                    )

        unsupported_scored_atoms = set(self.scored_atoms) - set(self.supported_atoms)
        if unsupported_scored_atoms:
            raise ValueError(
                f"Component {self.name} scored_atoms must be a subset of supported_atoms. "
                f"Unsupported scored atoms: {sorted(unsupported_scored_atoms)}."
            )
        if self.scored_atoms and not self.score_atom_types:
            raise ValueError(f"Component {self.name} defines scored_atoms while score_atom_types=False.")
        if self.score_nuclear_charge_feature and not self.uses_nuclear_charge_feature:
            raise ValueError(
                f"Component {self.name} cannot score the nuclear-charge feature without using that feature."
            )


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
