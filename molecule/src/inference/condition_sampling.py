import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from rdkit import Chem
from rdkit.Chem import Mol

from configs.config_sampler import _BaseSamplerConfig
from inference.sampling_runtime import SamplingRuntime, seed_everything
from postprocessing import MoleculeBuilder
from sampling.moe_layout import (
    ATOM_TYPE_DIM,
    ATOM_TYPE_INDEX,
    NUCLEAR_CHARGE_FEATURE_DIM,
    CrossDockedMoELayout,
    CrossDockedMoEMasks,
)
from sampling.path_factory import (
    build_diffsbdd_ligand_path,
    build_edm_fragment_path,
    build_edm_ligand_path,
    build_geodiff_fragment_path,
    make_zero_auxiliary_point,
)
from sampling.probability_path import MoEProbabilityPath
from sampling.sampler import InterleaveFn, MoEPDESampler, PostprocessFn
from utils.molecule_drawing import molecule_to_topology_png

logger = logging.getLogger(__name__)


def _fragment_atom_type_indices(fragment: Mol, device: str) -> torch.Tensor:
    atom_type_indices = []
    for atom in fragment.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ATOM_TYPE_INDEX:
            supported_atoms = ", ".join(ATOM_TYPE_INDEX)
            raise ValueError(f"Unsupported fragment atom type {symbol!r}. Supported atom types: {supported_atoms}.")
        atom_type_indices.append(ATOM_TYPE_INDEX[symbol])

    return torch.tensor(atom_type_indices, device=device, dtype=torch.long)


def _fragment_atom_feature_point(fragment: Mol, device: str, atom_type_value: float) -> torch.Tensor:
    point = torch.zeros(fragment.GetNumAtoms(), ATOM_TYPE_DIM + NUCLEAR_CHARGE_FEATURE_DIM, device=device)
    atom_type_indices = _fragment_atom_type_indices(fragment, device=device)
    point[torch.arange(fragment.GetNumAtoms(), device=device), atom_type_indices] = atom_type_value

    # The last auxiliary feature is EDM's integer nuclear-charge feature.
    # It is left as zero padding because molecule construction uses atom symbols, not this feature.
    return point


@dataclass(frozen=True)
class SamplingCondition:
    protein_pocket_pdb_path: Path
    fragment: Mol
    ref_ligand: Mol
    condition_id: str | None = None


@dataclass
class ConditionProbabilityPath:
    path: MoEProbabilityPath
    masks: CrossDockedMoEMasks
    fragment_atom_types: torch.Tensor
    prior_sbdd: Float[torch.Tensor, "B data"]
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
    num_ligand_atoms = condition.ref_ligand.GetNumAtoms()
    num_fragment_atoms = condition.fragment.GetNumAtoms()

    masks = CrossDockedMoELayout(
        fragment_size=num_fragment_atoms,
        ligand_size=num_ligand_atoms,
        device=device,
    ).masks()

    # 1. EDM fragment expert
    logger.info("  Building EDM probability path for fragment part generation...")
    runtime.edm_fragment.prepare_data(batch_size, num_fragment_atoms)
    q_edm_fragment_pad = build_edm_fragment_path(
        expert=runtime.edm_fragment,
        scheduler=runtime.scheduler_edm,
        masks=masks,
        padding_point=make_zero_auxiliary_point(num_fragment_atoms, masks.edm_fragment_padding, device=device),
        device=device,
    )

    # 2. EDM ligand expert
    logger.info("  Building EDM probability path for whole molecule generation...")
    runtime.edm_ligand.prepare_data(batch_size, num_ligand_atoms)
    q_edm_ligand_pad = build_edm_ligand_path(
        expert=runtime.edm_ligand,
        scheduler=runtime.scheduler_edm,
        masks=masks,
        padding_point=make_zero_auxiliary_point(num_ligand_atoms, masks.edm_ligand_padding, device=device),
        device=device,
    )

    # 3. GeoDiff fragment conformer expert
    logger.info("  Building GeoDiff probability path for fragment part conformer generation...")
    runtime.geodiff.prepare_data(batch_size, condition.fragment)

    fragment_atom_types = _fragment_atom_type_indices(condition.fragment, device=device)
    atom_type_value = 1.0 / float(runtime.edm_fragment.model.norm_values[1])
    geodiff_atom_feature_point = _fragment_atom_feature_point(
        condition.fragment,
        device=device,
        atom_type_value=atom_type_value,
    )
    q_geodiff_pad = build_geodiff_fragment_path(
        expert=runtime.geodiff,
        scheduler=runtime.scheduler_geodiff,
        masks=masks,
        atom_feature_point=geodiff_atom_feature_point,
        device=device,
    )

    # 4. DiffSBDD pocket-conditioned ligand expert
    logger.info("  Building DiffSBDD probability path for pocket conditioned whole molecule generation...")
    runtime.diffsbdd.prepare_data(
        batch_size,
        num_ligand_atoms,
        condition.protein_pocket_pdb_path,
        condition.ref_ligand,
    )
    q_sbdd_pad = build_diffsbdd_ligand_path(
        expert=runtime.diffsbdd,
        scheduler=runtime.scheduler_sbdd,
        masks=masks,
        padding_point=make_zero_auxiliary_point(num_ligand_atoms, masks.diffsbdd_ligand_padding, device=device),
        device=device,
    )

    moe_probability_path = MoEProbabilityPath(
        scheduler=runtime.scheduler_geodiff,
        q_list=[q_edm_fragment_pad, q_edm_ligand_pad, q_geodiff_pad, q_sbdd_pad],
        mask_list=[
            masks.fragment_state_in_ligand,
            masks.ligand_state,
            masks.fragment_state_in_ligand,
            masks.ligand_state,
        ],
        exponent_list=runtime.exponent_list,
        sample_size=masks.sample_size,
    )

    interleave_fns = [
        runtime.edm_fragment.interleave,
        runtime.edm_ligand.interleave,
        runtime.geodiff.interleave,
        functools.partial(runtime.diffsbdd.interleave, mask=masks.diffsbdd_ligand_xh),
    ]
    postprocess_fns = [
        # FIXME: Passing h_mask through a generic postprocess hook is a layout-specific workaround.
        # Refactor expert postprocessing to declare owned output channels explicitly.
        functools.partial(runtime.edm_fragment.postprocess, h_mask=None),
        functools.partial(runtime.edm_ligand.postprocess, h_mask=masks.edm_ligand_h_atom_type),
        runtime.geodiff.postprocess,
        functools.partial(runtime.diffsbdd.postprocess, mask=masks.diffsbdd_ligand_xh),
    ]

    return ConditionProbabilityPath(
        path=moe_probability_path,
        masks=masks,
        fragment_atom_types=fragment_atom_types,
        prior_sbdd=runtime.diffsbdd._inference_context.z,
        interleave_fns=interleave_fns,
        postprocess_fns=postprocess_fns,
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
            prior_sbdd=condition_path.prior_sbdd,
            interleave_fns=condition_path.interleave_fns,
            postprocess_fns=condition_path.postprocess_fns,
        )
    logger.info("Sampling completed. Number of samples: %d", samples.shape[0])

    if save_dir is not None:
        save_sampling_diagnostics(save_dir, logweight_trajectory, choices)

    logger.info("Converting atomic point-cloud features to XYZ blocks and RDKit molecules...")
    num_ligand_atoms = condition.ref_ligand.GetNumAtoms()
    batch_size = sampler_cfg.batch_size
    xh_lig = samples[:, condition_path.masks.ligand_state].reshape(batch_size, num_ligand_atoms, -1)
    output_xyz_blocks = MoleculeBuilder.xyz_blocks_from_batch(
        xh_lig,
        fragment_atom_types=condition_path.fragment_atom_types,
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
