import copy
import functools
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, cast

import hydra
import hydra.core.hydra_config
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from torch.distributions import Normal
from tqdm import tqdm

from configs import config as _config_registry  # Noqa: F401
from configs.config_benchmark import CrossDocked2020BenchConfig
from configs.config_sampler import _BaseSamplerConfig
from configs.config_weight import ACEBumpWeightConfig, _BaseWeightConfig
from pretrained_models.export_diffsbdd import interleave_fn as interleave_fn_diffsbdd
from pretrained_models.export_diffsbdd import postprocess_fn as postprocess_fn_diffsbdd
from pretrained_models.export_diffsbdd import prepare_data as prepare_data_diffsbdd
from pretrained_models.export_diffsbdd import score_function as score_function_diffsbdd
from pretrained_models.export_edm import encode_xh
from pretrained_models.export_edm import prepare_data as prepare_data_edm
from pretrained_models.export_edm import score_function as score_function_edm
from pretrained_models.export_geodiff import prepare_data as prepare_data_geodiff
from pretrained_models.export_geodiff import score_function as score_function_geodiff
from src.distributions import FixedPointDistribution
from src.probability_path import MoEProbabilityPath, PaddedProbabilityPath, ProbabilityPath
from src.sampler import MoEPDESampler
from src.scheduler import DiffSBDDScheduler, EDMScheduler, GeoDiffScheduler
from utils.utils_inference import (
    load_diffsbdd,
    load_edm,
    load_geodiff,
    replace_mol_topology_by_fragment,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data_component_mask_extend(num_atoms_subset, num_atoms, type="sbdd", device="cpu"):
    # space allocation: num_atoms_subset x (coord, sbdd_atom_types, include_charge) + (num_atoms - num_atoms_subset) x (coord, edm_atom_types), we include 'H' into 'others' type
    # edm_atoms = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    # sbdd_atoms = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
    assert type == "sbdd"

    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_geodiff_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_geodiff_part1[:, :3] = True
    mask_geodiff_part1 = mask_geodiff_part1.flatten()
    mask_geodiff_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool().flatten()
    mask_geodiff = torch.cat([mask_geodiff_part1, mask_geodiff_part2], dim=0)

    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_geodiff_h_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_geodiff_h_part1[:, 3:] = True
    mask_geodiff_h_part1 = mask_geodiff_h_part1.flatten()
    mask_geodiff_h_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool().flatten()
    mask_geodiff_h = torch.cat([mask_geodiff_h_part1, mask_geodiff_h_part2], dim=0)

    mask_subset = mask_geodiff + mask_geodiff_h

    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_edm_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_edm_part1[:, :3] = True
    mask_edm_part1[:, [3 + edm2sbdd_atoms[i] for i in edm2sbdd_atoms]] = True
    mask_edm_part1[:, -1] = True
    mask_edm_part1 = mask_edm_part1.flatten()

    mask_edm_h_part1 = ~mask_edm_part1

    mask_edm_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool()
    mask_edm_part2[:, :] = True
    mask_edm_part2 = mask_edm_part2.flatten()

    mask_edm = torch.cat([mask_edm_part1, mask_edm_part2], dim=0)

    mask_sbdd_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_sbdd_part1[:, :13] = True
    mask_sbdd_part1 = mask_sbdd_part1.flatten()
    mask_sbdd_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool()
    mask_sbdd_part2[:, :13] = True
    mask_sbdd_part2 = mask_sbdd_part2.flatten()
    mask_sbdd = torch.cat([mask_sbdd_part1, mask_sbdd_part2], dim=0)

    mask_sbdd_hpad_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_sbdd_hpad_part1[:, 13:] = True
    mask_sbdd_hpad_part1 = mask_sbdd_hpad_part1.flatten()
    mask_sbdd_hpad_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool()
    mask_sbdd_hpad_part2[:, 13:] = True
    mask_sbdd_hpad_part2 = mask_sbdd_hpad_part2.flatten()
    mask_sbdd_hpad = torch.cat([mask_sbdd_hpad_part1, mask_sbdd_hpad_part2], dim=0)

    mask = mask_sbdd + mask_sbdd_hpad

    mask_edm2 = torch.zeros(num_atoms, 15).bool()
    mask_edm2[:, :3] = True
    mask_edm2[:, [3 + edm2sbdd_atoms[i] for i in edm2sbdd_atoms]] = True
    mask_edm2[:, -1] = True
    mask_edm2 = mask_edm2.flatten()
    mask_edm2_h = ~mask_edm2
    return (
        mask_geodiff_part1.to(device),
        mask_geodiff_h_part1.to(device),
        mask_edm_part1.to(device),
        mask_edm_h_part1.to(device),
        mask_subset.to(device),
        mask_sbdd.to(device),
        mask_sbdd_hpad.to(device),
        mask.to(device),
        mask_edm2.to(device),
        mask_edm2_h.to(device),
    )


def build_exponent_list(weight_cfg: _BaseWeightConfig) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    # NOTE: For ACEBumpWeightConfig, the bump function is only applied to gamma_4.
    if isinstance(weight_cfg, ACEBumpWeightConfig):
        omega = weight_cfg.omega
        return [
            lambda t: torch.zeros_like(t) + omega,  # gamma_3
            lambda t: torch.zeros_like(t) - omega,  # gamma_1
            lambda t: torch.zeros_like(t) + weight_cfg.weight_function(t),  # gamma_4
            lambda t: torch.zeros_like(t) - omega,  # gamma_2
        ]

    weight_fn = weight_cfg.weight_function
    return [
        lambda t: torch.zeros_like(t) + weight_fn(t),  # gamma_3
        lambda t: torch.zeros_like(t) - weight_fn(t),  # gamma_1
        lambda t: torch.zeros_like(t) + weight_fn(t),  # gamma_4
        lambda t: torch.zeros_like(t) - (weight_fn(t) - 1),  # gamma_2
    ]


def log_exponent_list(exponent_list: List[Callable[[torch.Tensor], torch.Tensor]]) -> None:
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
    sbdd,
    edm,
    geodiff,
    args_edm,
    args_geodiff,
    sampler_cfg: _BaseSamplerConfig,
    exponent_list: List[Callable[[torch.Tensor], torch.Tensor]],
    save_dir: Path,
):
    device = sampler_cfg.device
    batch_size = sampler_cfg.batch_size

    ### GeoDiff Probability Path
    scheduler_geodiff = GeoDiffScheduler()
    scheduler_edm = EDMScheduler()
    scheduler_sbdd = DiffSBDDScheduler()

    num_ligand_atoms = ref_ligand.GetNumAtoms()  # whole molecule size
    num_fragment_atoms = fragment.GetNumAtoms()  # fragment size
    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    (
        mask_geodiff_part1,
        mask_geodiff_h_part1,
        mask_edm_part1,
        mask_edm_h_part1,
        mask_subset,
        mask_sbdd,
        mask_sbdd_hpad,
        mask,
        mask_edm2,
        mask_edm2_h,
    ) = make_data_component_mask_extend(num_fragment_atoms, num_ligand_atoms, type="sbdd", device=device)

    # [CONF] GeoDiff score, velocity, probability path p(Msc |Tsc)
    prepared_data_geodiff = prepare_data_geodiff(args_geodiff, geodiff, fragment, num_samples=batch_size, device=device)
    score_fn = functools.partial(score_function_geodiff, prepared_data=prepared_data_geodiff)
    q_geodiff = ProbabilityPath(scheduler_geodiff, score_fn)

    _, _h = encode_xh(args_edm, edm, fragment)
    h_int = _h[:, :-1].argmax(dim=-1)
    h_int = torch.tensor([edm2sbdd_atoms[v.item()] for v in h_int]).to(device)

    h = torch.zeros(num_fragment_atoms, 12).to(device=device)
    h[:, list(edm2sbdd_atoms.values()) + [-1]] = _h.to(device=device)
    h_dist = FixedPointDistribution(h.flatten(), device=device)
    h_score = h_dist.export_score_function(scheduler_geodiff)
    q_h = ProbabilityPath(scheduler_geodiff, h_score)

    q_geodiff_pad = PaddedProbabilityPath([q_geodiff, q_h], [mask_geodiff_part1, mask_geodiff_h_part1])
    # print("q_geodiff dim:", q_geodiff_pad.dim)  #: should be 15 * n (fragment size)
    # print("g_geodiff_pad.reverse:", q_geodiff_pad.reverse)  # True

    # [DN] EDM score, velocity, probability path for fragment part p(Msc)
    prepared_data_edm = prepare_data_edm(edm, batch_size, num_fragment_atoms, device=device)
    score_fn_edm = functools.partial(score_function_edm, prepared_data=prepared_data_edm)
    q_edm = ProbabilityPath(scheduler_edm, score_fn_edm)

    h = torch.zeros(num_fragment_atoms, 6).to(device=device)
    h_dist = FixedPointDistribution(h.flatten(), device=device)
    h_score = h_dist.export_score_function(scheduler_edm)
    q_h = ProbabilityPath(scheduler_edm, h_score)

    q_edm_pad = PaddedProbabilityPath([q_edm, q_h], [mask_edm_part1, mask_edm_h_part1])
    # print("q_edm dim:", q_edm_pad.dim)  #: should be 15 * n (fragment size)
    # print("g_edm_pad.reverse:", q_edm_pad.reverse)  # True

    # [DN] EDM score, velocity, probability path for whole molecule (fragment + complement) p(M)
    prepared_data_edm2 = prepare_data_edm(edm, batch_size, num_ligand_atoms, device=device)
    score_fn_edm2 = functools.partial(score_function_edm, prepared_data=prepared_data_edm2)
    q_edm2 = ProbabilityPath(scheduler_edm, score_fn_edm2)

    h2 = torch.zeros(num_ligand_atoms, 6).to(device=device)
    h2_dist = FixedPointDistribution(h2.flatten(), device=device)
    h2_score = h2_dist.export_score_function(scheduler_edm)
    q_h2 = ProbabilityPath(scheduler_edm, h2_score)

    q_edm_pad2 = PaddedProbabilityPath([q_edm2, q_h2], [mask_edm2, mask_edm2_h])
    # print("q_edm2 dim:", q_edm_pad2.dim)  #: should be 15 * N (whole molecule size)
    # print("g_edm_pad2.reverse:", q_edm_pad2.reverse)  # True

    # [SBDD] DiffSBDD score, velocity,probability path p(M|P)
    prepared_data_sbdd = prepare_data_diffsbdd(sbdd, pdb_path, ref_ligand, batch_size, num_ligand_atoms, device=device)
    score_fn_sbdd = functools.partial(score_function_diffsbdd, prepared_data=prepared_data_sbdd)
    q_sbdd = ProbabilityPath(scheduler_sbdd, score_fn_sbdd)

    h = torch.zeros(num_ligand_atoms, 2).to(device=device)
    h_dist = FixedPointDistribution(h.flatten(), device=device)
    h_score = h_dist.export_score_function(scheduler_sbdd)
    q_h = ProbabilityPath(scheduler_sbdd, h_score)

    q_sbdd_pad = PaddedProbabilityPath([q_sbdd, q_h], [mask_sbdd, mask_sbdd_hpad])

    prior_sbdd = prepared_data_sbdd["z"]
    prior = torch.randn(batch_size, *mask.shape).to(prior_sbdd.device)
    prior[:, mask_sbdd] = prior_sbdd
    # print("q_sbdd_pad dim:", q_sbdd_pad.dim)  #: should be 15 * N (whole molecule size)
    # print("g_sbdd_pad.reverse:", q_sbdd_pad.reverse)  # True

    # Compute log probability of gaussian prior
    standard_normal_dist = Normal(loc=0.0, scale=1.0)
    log_probs = standard_normal_dist.log_prob(prior)
    log_probs = log_probs.sum(dim=-1).unsqueeze(1).repeat(1, 4).unsqueeze(2)

    interleave_fn_sbdd = functools.partial(interleave_fn_diffsbdd, prepared_data=prepared_data_sbdd, mask=mask_sbdd)

    moe_probability_path = MoEProbabilityPath(
        scheduler_geodiff,  # for global noise schedule
        [q_geodiff_pad, q_edm_pad, q_sbdd_pad, q_edm_pad2],
        mask_list=[mask_subset, mask_subset, mask, mask],
        exponent_list=exponent_list,
        sample_size=tuple(mask.shape)[0],  # we assume 1-D sample shape
    )

    with torch.no_grad():
        samples, _, logweight_trajectory, _, choices = MoEPDESampler.sample(
            moe_probability_path,
            sampler_cfg,
            interleave_fn=interleave_fn_sbdd,
            prior_sbdd=prior_sbdd,
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

    samples = postprocess_fn_diffsbdd(
        samples.to(device=device),
        prepared_data=prepared_data_sbdd,
        mask=mask_sbdd,
        frag_atom_type=h_int,
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


def run_inference_crossdocked(cfg: DictConfig, output_dir: Path) -> None:
    benchmark_cfg = cast(CrossDocked2020BenchConfig, OmegaConf.to_object(cfg.benchmark))
    sampler_cfg = cast(_BaseSamplerConfig, OmegaConf.to_object(cfg.sampler))
    weight_cfg = cast(_BaseWeightConfig, OmegaConf.to_object(cfg.weight))
    exponent_list = build_exponent_list(weight_cfg)

    # setup benchmark configs
    num_trials = benchmark_cfg.num_trials
    data_root = Path(benchmark_cfg.data_root)
    protein_dir = data_root / "crossdocked_pocket10"
    processed_data_dir = data_root / "processed"
    result_dir_base = output_dir / "inference_results"
    seed = benchmark_cfg.seed

    # load pre-trained models and prepare probability paths
    device = sampler_cfg.device
    sbdd = load_diffsbdd(device=device)
    args_edm, edm = load_edm(device=device)
    args_geodiff, geodiff = load_geodiff(device=device)

    # Log exponent functions once before entering the inference loop.
    log_exponent_list(exponent_list)

    ###############################################################################################
    ########## Inference loop over tasks in CrossDocked2020 benchmark                    ##########
    ###############################################################################################
    task_path_list = sorted(processed_data_dir.glob("*.pt"), key=lambda x: int(x.stem))
    for task_path in tqdm(task_path_list):
        logger.info(f"Start task id {task_path.stem} | total {len(task_path_list)}")
        # load data for the task
        if not os.path.exists(task_path):
            continue
        data = torch.load(task_path, weights_only=False)

        pdb_path = protein_dir / f"{data['protein_filename']}"
        fragment: Mol = data["scaffold"]
        ligand: Mol = data["mol"]

        if num_ligand_atoms := ligand.GetNumAtoms() > 29:
            logger.warning(
                f"Task {task_path.stem} skipped due to ligand having {num_ligand_atoms} atoms, which exceeds the limit of 29."
            )
            continue

        # run inference with sampler
        save_dir = result_dir_base / f"{task_path.stem}"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        logger.info(
            f"Running inference for task {task_path.stem} with sampler {sampler_cfg.name} and weight {weight_cfg.name}"
        )
        original_samples_list = []
        replaced_samples_list = []
        for j in range(num_trials):  #!WARNING: we assert num_trials==1
            logger.info(f"Trial {j}")
            try:
                seed_everything(seed + j)
                original_samples, replaced_samples = run_single_task_sampling(
                    pdb_path,
                    fragment,
                    ligand,
                    sbdd,
                    edm,
                    geodiff,
                    args_edm,
                    args_geodiff,
                    sampler_cfg=sampler_cfg,
                    exponent_list=exponent_list,
                    save_dir=save_dir,
                )
            except Exception as e:
                logger.error(f"Trial {j} failed for task {task_path.stem}: {e}")
                continue
            original_samples_list.extend(original_samples)
            replaced_samples_list.extend(replaced_samples)
            logger.info(f"Valid samples: {len(replaced_samples)}")

        output_samples = replaced_samples_list
        outout_original_samples = original_samples_list
        for j, sample in enumerate(output_samples):
            try:
                writer = Chem.SDWriter(str(save_dir / f"{j}.sdf"))
                writer.write(sample)
                writer.close()
            except Exception:
                continue
        for j, sample in enumerate(outout_original_samples):
            try:
                writer = Chem.SDWriter(str(save_dir / f"original_{j}.sdf"))
                writer.write(sample)
                writer.close()
            except Exception:
                continue

    ###############################################################################################
    ########## Postprocessing step over CrossDocked2020 generated samples                ##########
    ###############################################################################################
    batch_size = sampler_cfg.batch_size
    postprocess_command = [
        sys.executable,
        "src/utils/postprocess_valfix.py",
        "--ligand_dir",
        str(result_dir_base),
        "--data_root",
        str(data_root),
        "--num_samples",
        str(batch_size),
    ]

    logger.info(f"Running postprocessing: {' '.join(postprocess_command)}")
    subprocess.run(postprocess_command, cwd=PROJECT_ROOT, check=True, text=True)

    logger.info("Finished inference and postprocessing for CrossDocked2020 benchmark.")


@hydra.main(config_path="../configs", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:

    output_dir: Path
    if cfg.get("output_dir") is not None:
        output_dir = Path(cfg.output_dir)
        logger.info(f"Using output directory specified in config: {output_dir}")
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        logger.info("Output directory not specified in config. Using Hydra's default output directory.")

    run_inference_crossdocked(cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
