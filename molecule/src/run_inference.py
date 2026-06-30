import logging
import shutil
from pathlib import Path
from typing import cast

import hydra
import hydra.core.hydra_config
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from configs import config as _config_registry  # Noqa: F401
from configs.config_sampler import _BaseSamplerConfig
from configs.config_weight import _BaseWeightConfig
from inference.condition_sampling import SamplingCondition, sample_condition, write_sampling_result
from inference.sampling_runtime import load_sampling_runtime, log_exponent_list
from postprocessing.valence_fix import postprocess_valfix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, output_dir: Path) -> None:
    sampler_cfg = cast(_BaseSamplerConfig, OmegaConf.to_object(cfg.sampler))
    weight_cfg = cast(_BaseWeightConfig, OmegaConf.to_object(cfg.weight))

    runtime = load_sampling_runtime(weight_cfg=weight_cfg, device=sampler_cfg.device)
    log_exponent_list(runtime.exponent_list)

    protein_pocket_pdb_path = Path(cfg.data.protein_pocket_pdb_path)
    fragment_sdf_path = Path(cfg.data.fragment_sdf_path)
    ligand_sdf_path = Path(cfg.data.ligand_sdf_path)

    fragment: Mol = Chem.SDMolSupplier(str(fragment_sdf_path), sanitize=False)[0]
    ligand: Mol = Chem.SDMolSupplier(str(ligand_sdf_path), sanitize=False)[0]

    num_ligand_atoms = ligand.GetNumAtoms()
    if num_ligand_atoms > 29:
        logger.warning(
            "29 atoms is the maximum supported number of atoms for the ligand in EDM training. "
            "However, the ligand in input data has %d atoms, which exceeds the limit. "
            "Please note that results may be unreliable.",
            num_ligand_atoms,
        )

    inference_output_dir = output_dir / "inference_output"
    if inference_output_dir.exists():
        shutil.rmtree(inference_output_dir)
    inference_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running inference with sampler %s and weight %s on device %s. Output will be saved to %s",
        sampler_cfg.name,
        weight_cfg.name,
        sampler_cfg.device,
        inference_output_dir,
    )
    condition = SamplingCondition(
        protein_pocket_pdb_path=protein_pocket_pdb_path,
        fragment=fragment,
        ref_ligand=ligand,
        condition_id="single_condition",
    )
    result = sample_condition(
        condition=condition,
        runtime=runtime,
        sampler_cfg=sampler_cfg,
        save_dir=inference_output_dir,
    )
    write_sampling_result(result, inference_output_dir)
    logger.info("Generated %d valid samples after replacement.", len(result.samples))

    logger.info("Starting postprocessing of generated samples with postprocess_valfix...")
    postprocess_valfix(
        protein_pocket_pdb_path=protein_pocket_pdb_path,
        fragment_sdf_path=fragment_sdf_path,
        ref_ligand_sdf_path=ligand_sdf_path,
        output_ligand_dir=inference_output_dir,
        num_samples=sampler_cfg.batch_size,
    )
    logger.info("Inference and postprocessing completed successfully.")


@hydra.main(config_path="configs", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.get("output_dir") is not None:
        output_dir = Path(cfg.output_dir)
        logger.info("Using output directory specified in config: %s", output_dir)
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        logger.info("Output directory not specified in config. Using Hydra's default output directory.")

    run_inference(cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
