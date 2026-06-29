import copy
import functools
import logging
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

import hydra
import hydra.core.hydra_config
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from configs import config as _config_registry  # Noqa: F401
from configs.config_sampler import _BaseSamplerConfig
from configs.config_weight import ACEBumpWeightConfig, _BaseWeightConfig
from src.experts import DiffSBDDExpert, EDMExpert, GeoDiffExpert
from src.moe_layout import EDM_ATOM_INDEX_TO_GLOBAL_ATOM_INDEX, CrossDockedMoELayout
from src.path_factory import (
    build_diffsbdd_ligand_path,
    build_edm_fragment_path,
    build_edm_ligand_path,
    build_geodiff_fragment_path,
    make_zero_auxiliary_point,
)
from src.postprocessing import MoleculeBuilder
from src.probability_path import MoEProbabilityPath
from src.sampler import MoEPDESampler
from src.scheduler import DiffSBDDScheduler, EDMScheduler, GeoDiffScheduler
from utils.inference_utils import replace_mol_topology_by_fragment
from utils.postprocess_valfix_utils import postprocess_valfix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_exponent_list(weight_cfg: _BaseWeightConfig) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    # NOTE: For ACEBumpWeightConfig, the bump function is only applied to gamma_4.
    if isinstance(weight_cfg, ACEBumpWeightConfig):
        omega = weight_cfg.omega
        return [
            lambda t: torch.zeros_like(t) + omega,  # gamma_3
            lambda t: torch.zeros_like(t) - omega,  # gamma_1
            lambda t: torch.zeros_like(t) + weight_cfg.weight_function(t),  # gamma_4
            lambda t: torch.zeros_like(t) - (omega - 1),  # gamma_2
        ]

    weight_fn = weight_cfg.weight_function
    return [
        lambda t: torch.zeros_like(t) + weight_fn(t),  # gamma_3
        lambda t: torch.zeros_like(t) - weight_fn(t),  # gamma_1
        lambda t: torch.zeros_like(t) + weight_fn(t),  # gamma_4
        lambda t: torch.zeros_like(t) - (weight_fn(t) - 1),  # gamma_2
    ]


def log_exponent_list(exponent_list: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:
    _exponent_ids = ["gamma_3", "gamma_1", "gamma_4", "gamma_2"]
    logger.info("Using exponent functions:")
    logger.info("-" * 50)
    _exponent_and_id_sorted = sorted(zip(exponent_list, _exponent_ids), key=lambda x: x[1])
    for exponent_fn, exponent_id in _exponent_and_id_sorted:
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.0])).item():.4f} (at t=0.0)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.25])).item():.4f} (at t=0.25)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.5])).item():.4f} (at t=0.5)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.75])).item():.4f} (at t=0.75)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([1.0])).item():.4f} (at t=1.0)")
        logger.info("-" * 50)


def run_single_task_sampling(
    pdb_path: Path,
    fragment: Mol,
    ref_ligand: Mol,
    sampler_cfg: _BaseSamplerConfig,
    exponent_list: list[Callable[[torch.Tensor], torch.Tensor]],
    save_dir: Path,
):
    device = sampler_cfg.device
    batch_size = sampler_cfg.batch_size

    ### GeoDiff Probability Path
    logger.info("Loading schedulers...")
    scheduler_geodiff = GeoDiffScheduler()
    scheduler_edm = EDMScheduler()
    scheduler_sbdd = DiffSBDDScheduler()

    ### Experts
    logger.info("Loading experts...")
    edm_expert_fragment = EDMExpert.from_pretrained(device=device)
    edm_expert_ligand = EDMExpert.from_pretrained(device=device)
    geodiff_expert = GeoDiffExpert.from_pretrained(device=device)
    diffsbdd_expert = DiffSBDDExpert.from_pretrained(device=device)

    seed_everything(sampler_cfg.seed)

    ### Prepare data and build probability paths
    logger.info("Preparing data and building probability paths...")
    num_ligand_atoms = ref_ligand.GetNumAtoms()  # whole molecule size
    num_fragment_atoms = fragment.GetNumAtoms()  # fragment size
    edm2sbdd_atoms = EDM_ATOM_INDEX_TO_GLOBAL_ATOM_INDEX
    masks = CrossDockedMoELayout(
        fragment_size=num_fragment_atoms,
        ligand_size=num_ligand_atoms,
        device=device,
    ).masks()

    # [CONF] GeoDiff score, velocity, probability path p(Msc |Tsc)
    logger.info("  Building GeoDiff probability path for fragment part conformer generation...")
    geodiff_expert.prepare_data(batch_size, fragment)

    _, _h = edm_expert_fragment.encode_xh(fragment)
    h_int = _h[:, :-1].argmax(dim=-1)
    h_int = torch.tensor([edm2sbdd_atoms[v.item()] for v in h_int]).to(device)

    geodiff_atom_type_point = make_zero_auxiliary_point(
        num_fragment_atoms,
        masks.geodiff_fragment_atom_types_and_charge,
        device=device,
    )
    geodiff_atom_type_point[:, list(edm2sbdd_atoms.values()) + [-1]] = _h.to(device=device)
    q_geodiff_pad = build_geodiff_fragment_path(
        expert=geodiff_expert,
        scheduler=scheduler_geodiff,
        masks=masks,
        atom_type_and_charge_point=geodiff_atom_type_point,
        device=device,
    )

    # [DN] EDM score, velocity, probability path for fragment part p(Msc)
    logger.info("  Building EDM probability path for fragment part generation...")
    edm_expert_fragment.prepare_data(batch_size, num_fragment_atoms)
    q_edm_fragment_pad = build_edm_fragment_path(
        expert=edm_expert_fragment,
        scheduler=scheduler_edm,
        masks=masks,
        padding_point=make_zero_auxiliary_point(num_fragment_atoms, masks.edm_fragment_padding, device=device),
        device=device,
    )

    # [DN] EDM score, velocity, probability path for whole molecule (fragment + complement) p(M)
    logger.info("  Building EDM probability path for whole molecule generation...")
    edm_expert_ligand.prepare_data(batch_size, num_ligand_atoms)
    q_edm_ligand_pad = build_edm_ligand_path(
        expert=edm_expert_ligand,
        scheduler=scheduler_edm,
        masks=masks,
        padding_point=make_zero_auxiliary_point(num_ligand_atoms, masks.edm_ligand_padding, device=device),
        device=device,
    )

    # [SBDD] DiffSBDD score, velocity,probability path p(M|P)
    logger.info("  Building DiffSBDD probability path for pocket conditioned whole molecule generation...")
    diffsbdd_expert.prepare_data(batch_size, num_ligand_atoms, pdb_path, ref_ligand)
    q_sbdd_pad = build_diffsbdd_ligand_path(
        expert=diffsbdd_expert,
        scheduler=scheduler_sbdd,
        masks=masks,
        padding_point=make_zero_auxiliary_point(num_ligand_atoms, masks.diffsbdd_ligand_padding, device=device),
        device=device,
    )

    logger.info("  Build MoE probability path for multi-expert sampling...")
    moe_probability_path = MoEProbabilityPath(
        scheduler=scheduler_geodiff,  # for global noise schedule
        q_list=[q_geodiff_pad, q_edm_fragment_pad, q_sbdd_pad, q_edm_ligand_pad],
        mask_list=[
            masks.fragment_state_in_ligand,
            masks.fragment_state_in_ligand,
            masks.ligand_state,
            masks.ligand_state,
        ],
        exponent_list=exponent_list,
        sample_size=masks.sample_size,  # we assume 1-D sample shape
    )

    ### Run MoE-PDE sampling
    prior_sbdd = diffsbdd_expert._inference_context.z
    prior = torch.randn(batch_size, masks.sample_size).to(prior_sbdd.device)
    prior[:, masks.diffsbdd_ligand_xh] = prior_sbdd

    interleave_fns = [
        geodiff_expert.interleave,
        edm_expert_fragment.interleave,
        functools.partial(diffsbdd_expert.interleave, mask=masks.diffsbdd_ligand_xh),
        edm_expert_ligand.interleave,
    ]
    postprocess_fns = [
        geodiff_expert.postprocess,
        edm_expert_fragment.postprocess,
        functools.partial(diffsbdd_expert.postprocess, mask=masks.diffsbdd_ligand_xh),
        edm_expert_ligand.postprocess,
    ]

    with torch.no_grad():
        samples, _, logweight_trajectory, _, choices = MoEPDESampler.sample(
            moe_probability_path,
            sampler_cfg,
            prior_sbdd=prior_sbdd,
            interleave_fns=interleave_fns,
            postprocess_fns=postprocess_fns,
        )

    # Log the log weights and choices across the trajectory for debugging and analysis
    with open(save_dir / "logweight_trajectory.txt", "w") as f_logweight:
        f_logweight.write("Log weight trajectory across sampling steps:\n")
        for step, logweight in enumerate(logweight_trajectory):
            f_logweight.write(f"Step {step}: {logweight.cpu().numpy()}\n")
    with open(save_dir / "choices.txt", "w") as f_choices:
        f_choices.write("Expert choices across sampling steps:\n")
        for step, choice in enumerate(choices):
            f_choices.write(f"Step {step}: {choice}\n")

    xh_lig = samples[:, masks.diffsbdd_ligand_xh].reshape(batch_size, num_ligand_atoms, -1)
    samples = MoleculeBuilder.build_batch(
        xh=xh_lig,
        dataset_info=diffsbdd_expert.model.dataset_info,
        fragment_atom_types=h_int,
        x_dims=diffsbdd_expert.model.x_dims,
        add_coords=True,
        add_hydrogens=False,
        sanitize=False,
        relax_iter=0,
        largest_frag=False,
    )

    replaced_samples = []
    original_samples = []
    for sample in samples:
        original_samples.append(copy.deepcopy(sample))
        try:
            replaced_sample = replace_mol_topology_by_fragment(sample, fragment, list(range(fragment.GetNumAtoms())))
            replaced_samples.append(replaced_sample)
        except Exception as e:
            logger.error(f"Replacement failed for a sample: {e}")
            continue

    return copy.deepcopy(original_samples), copy.deepcopy(replaced_samples)


def run_inference(cfg: DictConfig, output_dir: Path) -> None:
    sampler_cfg = cast(_BaseSamplerConfig, OmegaConf.to_object(cfg.sampler))
    weight_cfg = cast(_BaseWeightConfig, OmegaConf.to_object(cfg.weight))

    # Build and log exponent functions once before entering the inference loop.
    exponent_list = build_exponent_list(weight_cfg)
    log_exponent_list(exponent_list)

    # Load data
    protein_pocket_pdb_path = Path(cfg.data.protein_pocket_pdb_path)
    fragment_sdf_path = Path(cfg.data.fragment_sdf_path)
    ligand_sdf_path = Path(cfg.data.ligand_sdf_path)

    fragment: Mol = Chem.SDMolSupplier(str(fragment_sdf_path), sanitize=False)[0]
    ligand: Mol = Chem.SDMolSupplier(str(ligand_sdf_path), sanitize=False)[0]

    if num_ligand_atoms := ligand.GetNumAtoms() > 29:
        logger.warning(
            f"29 atoms is the maximum supported number of atoms for the ligand in EDM training. However, the ligand in input data has {num_ligand_atoms} atoms, which exceeds the limit. Please note that results may be unreliable."
        )

    # Create output directory for inference results
    output_dir = output_dir / "inference_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference with sampler
    logger.info(
        f"Running inference with sampler {sampler_cfg.name} and weight {weight_cfg.name} on device {sampler_cfg.device}. Output will be saved to {output_dir}"
    )
    original_samples, replaced_samples = run_single_task_sampling(
        pdb_path=protein_pocket_pdb_path,
        fragment=fragment,
        ref_ligand=ligand,
        sampler_cfg=sampler_cfg,
        exponent_list=exponent_list,
        save_dir=output_dir,
    )
    logger.info(f"Generated {len(replaced_samples)} valid samples after replacement.")

    # Save generated samples to output directory
    for j, sample in enumerate(replaced_samples):
        try:
            writer = Chem.SDWriter(str(output_dir / f"{j}.sdf"))
            writer.write(sample)
            writer.close()
        except Exception as e:
            logger.error(f"Failed to save sample {j}: {e}")
            continue
    for j, sample in enumerate(original_samples):
        try:
            writer = Chem.SDWriter(str(output_dir / f"original_{j}.sdf"))
            writer.write(sample)
            writer.close()
        except Exception as e:
            logger.error(f"Failed to save original sample {j}: {e}")
            continue

    logger.info("Starting postprocessing of generated samples with postprocess_valfix...")
    postprocess_valfix(
        protein_pocket_pdb_path=protein_pocket_pdb_path,
        fragment_sdf_path=fragment_sdf_path,
        ref_ligand_sdf_path=ligand_sdf_path,
        output_ligand_dir=output_dir,
        num_samples=sampler_cfg.batch_size,
    )
    logger.info("Inference and postprocessing completed successfully.")


@hydra.main(config_path="../configs", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:

    output_dir: Path
    if cfg.get("output_dir") is not None:
        output_dir = Path(cfg.output_dir)
        logger.info(f"Using output directory specified in config: {output_dir}")
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        logger.info("Output directory not specified in config. Using Hydra's default output directory.")

    run_inference(cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
