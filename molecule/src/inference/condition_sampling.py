import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Mol

from configs.config_moe_component import (
    CONDITION_FRAGMENT_MOL,
    CONDITION_POCKET,
    FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL,
    NODE_SCOPE_LIGAND,
    MoEComponentConfig,
)
from configs.config_sampler import _BaseSamplerConfig
from experts import DiffSBDDExpert, EDMExpert, GeoDiffExpert
from experts.base_expert import SBDDExpert
from inference.sampling_runtime import ComponentRuntime, SamplingRuntime, seed_everything
from postprocessing import MoleculeBuilder
from sampling.moe_layout import (
    COORDS_DIM,
    DynamicMoELayout,
)
from sampling.path_factory import (
    build_padded_expert_path,
    make_zero_auxiliary_point,
)
from sampling.probability_path import MoEProbabilityPath
from sampling.sampler import InterleaveFn, MoEPDESampler, PostprocessFn
from utils.molecule_drawing import molecule_to_topology_png

logger = logging.getLogger(__name__)


def _mol_atom_type_indices(mol: Mol, layout: DynamicMoELayout, device: str) -> torch.Tensor:
    atom_type_indices = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in layout.atom_type_index:
            supported_atoms = ", ".join(layout.atom_type_index)
            raise ValueError(f"Unsupported atom type {symbol!r}. Supported atom types: {supported_atoms}.")
        atom_type_indices.append(layout.atom_type_index[symbol])

    return torch.tensor(atom_type_indices, device=device, dtype=torch.long)


def _mol_atom_feature_point(
    mol: Mol,
    layout: DynamicMoELayout,
    auxiliary_mask: torch.Tensor,
    device: str,
    atom_type_value: float,
) -> torch.Tensor:
    point = torch.zeros(mol.GetNumAtoms(), layout.node_feature_dim, device=device)
    atom_type_indices = _mol_atom_type_indices(mol, layout=layout, device=device)
    point[
        torch.arange(mol.GetNumAtoms(), device=device),
        COORDS_DIM + atom_type_indices,
    ] = atom_type_value

    # The last auxiliary feature is EDM's integer nuclear-charge feature.
    # It is left as zero padding because molecule construction uses atom symbols, not this feature.
    return point.flatten()[auxiliary_mask].reshape(mol.GetNumAtoms(), -1)


@dataclass(frozen=True)
class SamplingCondition:
    protein_pocket_pdb_path: Path
    fragment: Mol
    ref_ligand: Mol
    num_ligand_atoms: int | None = None
    condition_id: str | None = None


@dataclass
class ConditionProbabilityPath:
    path: MoEProbabilityPath
    layout: DynamicMoELayout
    fragment_atom_types: torch.Tensor
    sbdd_expert: SBDDExpert | None
    interleave_fns: list[InterleaveFn]
    postprocess_fns: list[PostprocessFn]


@dataclass
class SamplingResult:
    condition: SamplingCondition
    xyz_blocks: list[str]
    samples: list[Mol | None]
    logweight_trajectory: torch.Tensor
    choices: np.ndarray


def build_condition_probability_path(
    condition: SamplingCondition,
    runtime: SamplingRuntime,
    sampler_cfg: _BaseSamplerConfig,
) -> ConditionProbabilityPath:
    device = sampler_cfg.device
    batch_size = sampler_cfg.batch_size
    num_ligand_atoms = resolve_num_ligand_atoms(condition)
    num_fragment_atoms = condition.fragment.GetNumAtoms()

    layout = DynamicMoELayout.from_components(
        [component.config for component in runtime.components],
        fragment_size=num_fragment_atoms,
        ligand_size=num_ligand_atoms,
        device=device,
    )
    logger.info("Dynamic MoE atom layout: %s", layout.atom_type_index)

    q_list = []
    mask_list = []
    interleave_fns: list[InterleaveFn] = []
    postprocess_fns: list[PostprocessFn] = []
    sbdd_expert: SBDDExpert | None = None
    fragment_atom_types = _mol_atom_type_indices(condition.fragment, layout=layout, device=device)

    for component_runtime in runtime.components:
        component = component_runtime.config
        logger.info("  Building probability path for component %s...", component.name)
        _prepare_component_data(
            component_runtime=component_runtime,
            condition=condition,
            layout=layout,
            batch_size=batch_size,
            num_ligand_atoms=num_ligand_atoms,
        )

        active_mask = layout.active_mask_for_component(component)
        auxiliary_mask = layout.auxiliary_mask_for_component(component)
        auxiliary_point = _component_auxiliary_point(
            component_runtime=component_runtime,
            condition=condition,
            runtime=runtime,
            layout=layout,
            auxiliary_mask=auxiliary_mask,
            device=device,
        )
        q_list.append(
            build_padded_expert_path(
                expert=component_runtime.expert,
                scheduler=component_runtime.scheduler,
                active_mask=active_mask,
                auxiliary_mask=auxiliary_mask,
                auxiliary_point=auxiliary_point,
                device=device,
            )
        )
        mask_list.append(layout.state_mask_for_scope(component.node_scope))
        interleave_fns.append(_component_interleave_fn(component_runtime, active_mask))
        postprocess_fns.append(_component_postprocess_fn(component_runtime, layout, active_mask))

        # FIXME: This does not gracefully handle multiple DiffSBDD components. We assume that there is at most one DiffSBDD component in the runtime.
        if isinstance(component_runtime.expert, DiffSBDDExpert):
            sbdd_expert = component_runtime.expert

    global_scheduler = _select_global_scheduler(runtime)
    moe_probability_path = MoEProbabilityPath(
        scheduler=global_scheduler,
        q_list=q_list,
        mask_list=mask_list,
        exponent_list=runtime.exponent_list,
        sample_size=layout.sample_size,
        node_feature_dim=layout.node_feature_dim,
    )

    return ConditionProbabilityPath(
        path=moe_probability_path,
        layout=layout,
        fragment_atom_types=fragment_atom_types,
        sbdd_expert=sbdd_expert,
        interleave_fns=interleave_fns,
        postprocess_fns=postprocess_fns,
    )


def _prepare_component_data(
    *,
    component_runtime: ComponentRuntime,
    condition: SamplingCondition,
    layout: DynamicMoELayout,
    batch_size: int,
    num_ligand_atoms: int,
) -> None:
    component = component_runtime.config
    expert = component_runtime.expert
    num_nodes = layout.num_nodes_for_scope(component.node_scope)

    if isinstance(expert, EDMExpert):
        expert.prepare_data(batch_size, num_nodes)
        return

    if isinstance(expert, GeoDiffExpert):
        if component.fixed_atom_type_source is None:
            raise ValueError(
                f"GeoDiff component {component.name} must define fixed_atom_type_source because GeoDiff "
                "scores coordinates conditioned on a fixed molecular graph and atom types."
            )
        fixed_mol = _condition_mol_for_source(component, condition, component.fixed_atom_type_source)
        _validate_condition_mol_size(component, fixed_mol, expected_num_nodes=num_nodes)
        expert.prepare_data(batch_size, fixed_mol)
        return

    if isinstance(expert, DiffSBDDExpert):
        _require_condition_key(component, CONDITION_POCKET)
        if component.node_scope != NODE_SCOPE_LIGAND:
            raise ValueError(f"DiffSBDD component {component.name} must use node_scope={NODE_SCOPE_LIGAND!r}.")
        expert.prepare_data(
            batch_size,
            num_ligand_atoms,
            condition.protein_pocket_pdb_path,
            condition.ref_ligand,
        )
        return

    raise TypeError(f"Unsupported expert type for component {component.name}: {type(expert).__name__}.")


def _component_auxiliary_point(
    *,
    component_runtime: ComponentRuntime,
    condition: SamplingCondition,
    runtime: SamplingRuntime,
    layout: DynamicMoELayout,
    auxiliary_mask: torch.Tensor,
    device: str,
) -> torch.Tensor:
    component = component_runtime.config
    num_nodes = layout.num_nodes_for_scope(component.node_scope)
    if component.fixed_atom_type_source is not None:
        fixed_mol = _condition_mol_for_source(component, condition, component.fixed_atom_type_source)
        _validate_condition_mol_size(component, fixed_mol, expected_num_nodes=num_nodes)
        return _mol_atom_feature_point(
            fixed_mol,
            layout=layout,
            auxiliary_mask=auxiliary_mask,
            device=device,
            atom_type_value=_atom_type_value(runtime),
        )

    return make_zero_auxiliary_point(num_nodes, auxiliary_mask, device=device)


def _component_interleave_fn(
    component_runtime: ComponentRuntime,
    active_mask: torch.Tensor,
) -> InterleaveFn:
    expert = component_runtime.expert
    if isinstance(expert, DiffSBDDExpert):
        return functools.partial(expert.interleave, mask=active_mask)
    return expert.interleave


def _component_postprocess_fn(
    component_runtime: ComponentRuntime,
    layout: DynamicMoELayout,
    active_mask: torch.Tensor,
) -> PostprocessFn:
    component = component_runtime.config
    expert = component_runtime.expert

    if isinstance(expert, EDMExpert):
        h_mask = layout.h_atom_type_mask_for_scope(component.node_scope)
        if component.node_scope != NODE_SCOPE_LIGAND:
            h_mask = None
        return functools.partial(expert.postprocess, h_mask=h_mask)

    if isinstance(expert, DiffSBDDExpert):
        return functools.partial(expert.postprocess, mask=active_mask)

    return expert.postprocess


def _select_global_scheduler(runtime: SamplingRuntime):
    return runtime.global_scheduler


def _atom_type_value(runtime: SamplingRuntime) -> float:
    for component_runtime in runtime.components:
        expert = component_runtime.expert
        if isinstance(expert, EDMExpert):
            return 1.0 / float(expert.model.norm_values[1])
    return 1.0


def _condition_mol_for_source(component: MoEComponentConfig, condition: SamplingCondition, source: str) -> Mol:
    if source == FIXED_ATOM_TYPE_SOURCE_FRAGMENT_MOL:
        _require_condition_key(component, CONDITION_FRAGMENT_MOL)
        return condition.fragment

    raise ValueError(f"Unsupported fixed atom-type source {source!r} for component {component.name}.")


def _validate_condition_mol_size(component: MoEComponentConfig, mol: Mol, expected_num_nodes: int) -> None:
    actual_num_nodes = mol.GetNumAtoms()
    if actual_num_nodes != expected_num_nodes:
        raise ValueError(
            f"Component {component.name} uses node_scope={component.node_scope!r} with {expected_num_nodes} nodes, "
            f"but fixed atom/topology source {component.fixed_atom_type_source!r} has {actual_num_nodes} atoms."
        )


def _require_condition_key(component: MoEComponentConfig, condition_key: str) -> None:
    if condition_key not in component.condition_keys:
        raise ValueError(
            f"Component {component.name} requires condition key {condition_key!r}, "
            f"but its condition_keys are {component.condition_keys}."
        )


def sample_condition(
    condition: SamplingCondition,
    runtime: SamplingRuntime,
    sampler_cfg: _BaseSamplerConfig,
    save_dir: Path | None = None,
) -> SamplingResult:
    seed_everything(sampler_cfg.seed)

    logger.info("Preparing data and building probability paths...")
    condition_path = build_condition_probability_path(condition, runtime, sampler_cfg)

    logger.info("Running MoE-PDE sampling...")
    with torch.no_grad():
        samples, _, logweight_trajectory, _, choices = MoEPDESampler.sample(
            condition_path.path,
            sampler_cfg,
            sbdd_expert=condition_path.sbdd_expert,
            interleave_fns=condition_path.interleave_fns,
            postprocess_fns=condition_path.postprocess_fns,
        )
    logger.info("Sampling completed. Number of samples: %d", samples.shape[0])

    if save_dir is not None:
        save_sampling_diagnostics(save_dir, logweight_trajectory, choices)

    logger.info("Converting atomic point-cloud features to XYZ blocks and RDKit molecules...")
    num_ligand_atoms = resolve_num_ligand_atoms(condition)
    batch_size = sampler_cfg.batch_size
    ligand_state_mask = condition_path.layout.state_mask_for_scope(NODE_SCOPE_LIGAND)
    xh_lig = samples[:, ligand_state_mask].reshape(batch_size, num_ligand_atoms, condition_path.layout.node_feature_dim)
    output_xyz_blocks = MoleculeBuilder.xyz_blocks_from_batch(
        xh_lig,
        fragment_atom_types=condition_path.fragment_atom_types,
        atom_type_decoder=condition_path.layout.atom_type_decoder,
        atom_type_dim=condition_path.layout.atom_type_dim,
    )
    output_molecules = MoleculeBuilder.build_mols_from_xyz_blocks(output_xyz_blocks)

    return SamplingResult(
        condition=condition,
        xyz_blocks=output_xyz_blocks,
        samples=output_molecules,
        logweight_trajectory=logweight_trajectory,
        choices=choices,
    )


def sample_conditions(
    conditions: list[SamplingCondition],
    runtime: SamplingRuntime,
    sampler_cfg: _BaseSamplerConfig,
    output_root: Path | None = None,
) -> list[SamplingResult]:
    results = []
    for idx, condition in enumerate(conditions):
        condition_name = condition.condition_id or f"condition_{idx:05d}"
        logger.info("Sampling condition %s (%d/%d)", condition_name, idx + 1, len(conditions))
        save_dir = None
        if output_root is not None:
            save_dir = output_root / condition_name
            save_dir.mkdir(parents=True, exist_ok=True)

        result = sample_condition(condition, runtime, sampler_cfg, save_dir=save_dir)
        if save_dir is not None:
            write_sampling_result(result, save_dir)
        results.append(result)

    return results


def resolve_num_ligand_atoms(condition: SamplingCondition) -> int:
    if condition.num_ligand_atoms is None:
        return condition.ref_ligand.GetNumAtoms()

    if condition.num_ligand_atoms <= 0:
        raise ValueError(f"num_ligand_atoms must be positive, got {condition.num_ligand_atoms}.")

    num_fragment_atoms = condition.fragment.GetNumAtoms()
    if condition.num_ligand_atoms < num_fragment_atoms:
        raise ValueError(
            "num_ligand_atoms cannot be smaller than the fragment atom count: "
            f"{condition.num_ligand_atoms} < {num_fragment_atoms}."
        )

    return condition.num_ligand_atoms


def save_sampling_diagnostics(save_dir: Path, logweight_trajectory: torch.Tensor, choices: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "logweight_trajectory.txt", "w") as f_logweight:
        f_logweight.write("Log weight trajectory across sampling steps:\n")
        for step, logweight in enumerate(logweight_trajectory):
            f_logweight.write(f"Step {step}: {logweight.cpu().numpy()}\n")

    with open(save_dir / "choices.txt", "w") as f_choices:
        f_choices.write("Expert choices across sampling steps:\n")
        for step, choice in enumerate(choices):
            f_choices.write(f"Step {step}: {choice}\n")


def write_sampling_result(result: SamplingResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_molecule_topology_png(result.condition.fragment, output_dir / "fragment.png")

    for idx, xyz_block in enumerate(result.xyz_blocks):
        try:
            (output_dir / f"{idx}.xyz").write_text(xyz_block)
        except Exception as e:
            logger.error(f"Failed to save XYZ sample {idx}: {e}")

    for idx, sample in enumerate(result.samples):
        if sample is None:
            continue
        try:
            writer = Chem.SDWriter(str(output_dir / f"{idx}.sdf"))
            writer.write(sample)
            writer.close()
            _write_molecule_topology_png(sample, output_dir / f"{idx}.png")
        except Exception as e:
            logger.error(f"Failed to save sample {idx}: {e}")


def _write_molecule_topology_png(mol: Mol, output_path: Path) -> None:
    try:
        output_path.write_bytes(molecule_to_topology_png(mol))
    except Exception as e:
        logger.warning(f"Failed to save molecule topology image to {output_path}: {e}")
