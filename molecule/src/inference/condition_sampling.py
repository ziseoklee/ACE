import copy
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
from postprocessing.topology import replace_mol_topology_by_fragment
from sampling.moe_layout import _EDM_QM9_ATOM_INDEX_TO_GLOBAL_ATOM_INDEX, CrossDockedMoELayout, CrossDockedMoEMasks
from sampling.path_factory import (
    build_diffsbdd_ligand_path,
    build_edm_fragment_path,
    build_edm_ligand_path,
    build_geodiff_fragment_path,
    make_zero_auxiliary_point,
)
from sampling.probability_path import MoEProbabilityPath
from sampling.sampler import InterleaveFn, MoEPDESampler, PostprocessFn

logger = logging.getLogger(__name__)


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
    original_samples: list[Mol | None]
    samples: list[Mol]
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
    edm2sbdd_atoms = _EDM_QM9_ATOM_INDEX_TO_GLOBAL_ATOM_INDEX

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

    _, fragment_h = runtime.edm_fragment.encode_xh(condition.fragment)
    fragment_atom_types = fragment_h[:, :-1].argmax(dim=-1)
    fragment_atom_types = torch.tensor([edm2sbdd_atoms[int(v.item())] for v in fragment_atom_types], device=device)

    geodiff_atom_feature_point = make_zero_auxiliary_point(
        num_fragment_atoms,
        masks.geodiff_fragment_atom_features,
        device=device,
    )
    geodiff_atom_feature_point[:, list(edm2sbdd_atoms.values()) + [-1]] = fragment_h.to(device=device)
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
        runtime.edm_fragment.postprocess,
        runtime.edm_ligand.postprocess,
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

    if save_dir is not None:
        save_sampling_diagnostics(save_dir, logweight_trajectory, choices)

    num_ligand_atoms = condition.ref_ligand.GetNumAtoms()
    batch_size = sampler_cfg.batch_size
    xh_lig = samples[:, condition_path.masks.diffsbdd_ligand_xh].reshape(batch_size, num_ligand_atoms, -1)
    molecules = MoleculeBuilder.build_batch(
        xh=xh_lig,
        dataset_info=runtime.diffsbdd.model.dataset_info,
        fragment_atom_types=condition_path.fragment_atom_types,
        x_dims=runtime.diffsbdd.model.x_dims,
        add_coords=True,
        add_hydrogens=False,
        sanitize=False,
        relax_iter=0,
        largest_frag=False,
    )

    original_samples: list[Mol | None] = []
    replaced_samples: list[Mol] = []
    for molecule in molecules:
        original_samples.append(copy.deepcopy(molecule))
        if molecule is None:
            continue
        try:
            replaced_sample = replace_mol_topology_by_fragment(
                molecule,
                condition.fragment,
                list(range(condition.fragment.GetNumAtoms())),
            )
            replaced_samples.append(replaced_sample)
        except Exception as e:
            logger.error(f"Replacement failed for a sample: {e}")

    return SamplingResult(
        condition=condition,
        original_samples=copy.deepcopy(original_samples),
        samples=copy.deepcopy(replaced_samples),
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


def write_sampling_result(result: SamplingResult, output_dir: Path, save_original: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(result.samples):
        try:
            writer = Chem.SDWriter(str(output_dir / f"{idx}.sdf"))
            writer.write(sample)
            writer.close()
        except Exception as e:
            logger.error(f"Failed to save sample {idx}: {e}")

    if not save_original:
        return

    for idx, sample in enumerate(result.original_samples):
        if sample is None:
            continue
        try:
            writer = Chem.SDWriter(str(output_dir / f"original_{idx}.sdf"))
            writer.write(sample)
            writer.close()
        except Exception as e:
            logger.error(f"Failed to save original sample {idx}: {e}")
