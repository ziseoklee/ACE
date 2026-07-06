import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import hydra
import hydra.core.hydra_config
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from configs import config as _config_registry  # Noqa: F401
from configs.config_moe import MoEConfig, MoEExponentConfig
from configs.config_sampler import _BaseSamplerConfig
from inference.condition_sampling import (
    SamplingCondition,
    resolve_num_ligand_atoms,
    sample_condition,
    write_sampling_result,
)
from inference.sampling_runtime import (
    component_configs_from_hydra,
    load_sampling_runtime,
    log_exponent_list,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_inference(cfg: DictConfig, output_dir: Path) -> None:
    sampler_cfg = cast(_BaseSamplerConfig, OmegaConf.to_object(cfg.sampler))
    moe_cfg = cast(MoEConfig, OmegaConf.to_object(cfg.moe))

    component_configs = component_configs_from_hydra(moe_cfg.components)
    runtime = load_sampling_runtime(
        device=sampler_cfg.device,
        component_configs=component_configs,
        global_scheduler_key=moe_cfg.global_scheduler_key,
        exponent_configs=moe_cfg.exponents,
        diffusion_scale=moe_cfg.diffusion_scale,
    )
    log_exponent_list(runtime.exponent_list, [component.component_id for component in runtime.components])

    protein_pocket_pdb_path = Path(cfg.data.protein_pocket_pdb_path)
    fragment_sdf_path = Path(cfg.data.fragment_sdf_path)
    ligand_sdf_path = Path(cfg.data.ligand_sdf_path)

    fragment: Mol = Chem.SDMolSupplier(str(fragment_sdf_path), sanitize=False)[0]
    ligand: Mol = Chem.SDMolSupplier(str(ligand_sdf_path), sanitize=False)[0]
    num_ligand_atoms_override = cfg.data.get("num_ligand_atoms")
    if num_ligand_atoms_override is not None:
        num_ligand_atoms_override = int(num_ligand_atoms_override)

    condition = SamplingCondition(
        protein_pocket_pdb_path=protein_pocket_pdb_path,
        fragment=fragment,
        ref_ligand=ligand,
        num_ligand_atoms=num_ligand_atoms_override,
        condition_id="single_condition",
    )
    num_ligand_atoms = resolve_num_ligand_atoms(condition)
    if num_ligand_atoms_override is None:
        logger.info("Using reference ligand atom count: %d", num_ligand_atoms)
    else:
        logger.info(
            "Using configured ligand atom count: %d (reference ligand has %d atoms)",
            num_ligand_atoms,
            ligand.GetNumAtoms(),
        )

    inference_output_dir = output_dir / "inference_output"
    if inference_output_dir.exists():
        shutil.rmtree(inference_output_dir)
    inference_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running inference with sampler %s and weight %s on device %s. Output will be saved to %s",
        sampler_cfg.name,
        _format_exponent_weight_configs(moe_cfg.exponents),
        sampler_cfg.device,
        inference_output_dir,
    )
    result = sample_condition(
        condition=condition,
        runtime=runtime,
        sampler_cfg=sampler_cfg,
        save_dir=inference_output_dir,
    )
    write_sampling_result(result, inference_output_dir)
    num_valid_samples = sum(sample is not None for sample in result.samples)
    logger.info("Generated %d valid samples.", num_valid_samples)
    logger.info("Inference completed successfully.")


def _format_exponent_weight_configs(exponents: Mapping[str, MoEExponentConfig]) -> str:
    return ", ".join(
        f"{component_id}:{exponent_cfg.weight_fn.name}" for component_id, exponent_cfg in exponents.items()
    )


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
