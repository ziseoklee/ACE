import copy
import logging
import shutil
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import hydra
import hydra.core.hydra_config
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from configs import config as _config_registry  # Noqa: F401
from configs.config_benchmark import CrossDocked2020BenchConfig
from configs.config_sampler import _BaseSamplerConfig
from configs.config_weight import _BaseWeightConfig
from inference.condition_sampling import SamplingCondition, sample_condition
from inference.sampling_runtime import load_sampling_runtime, log_exponent_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_EDM_LIGAND_ATOMS = 29


def run_crossdocked_inference(cfg: DictConfig, output_dir: Path) -> None:
    benchmark_cfg = cast(CrossDocked2020BenchConfig, OmegaConf.to_object(cfg.benchmark))
    sampler_cfg = cast(_BaseSamplerConfig, OmegaConf.to_object(cfg.sampler))
    weight_cfg = cast(_BaseWeightConfig, OmegaConf.to_object(cfg.weight))

    data_root = Path(benchmark_cfg.data_root)
    protein_dir = data_root / "crossdocked_pocket10"
    processed_data_dir = data_root / "processed"
    save_root = Path(benchmark_cfg.save_dir)
    run_save_dir = _make_run_save_dir(save_root, sampler_cfg, weight_cfg)
    inference_save_dir = run_save_dir / "inference"

    logger.info("Hydra output directory: %s", output_dir)
    logger.info("CrossDocked2020 data root: %s", data_root)
    logger.info("CrossDocked2020 save root: %s", save_root)
    logger.info("CrossDocked2020 run save directory: %s", run_save_dir)
    logger.info("CrossDocked2020 inference save directory: %s", inference_save_dir)

    runtime = load_sampling_runtime(weight_cfg=weight_cfg, device=sampler_cfg.device)
    log_exponent_list(runtime.exponent_list)

    task_path_list = _load_task_paths(processed_data_dir)
    logger.info("Loaded %d CrossDocked2020 tasks from %s", len(task_path_list), processed_data_dir)

    inference_save_dir.mkdir(parents=True, exist_ok=False)
    for task_path in tqdm(task_path_list):
        logger.info("Start task id %s | total %d", task_path.stem, len(task_path_list))
        try:
            data = torch.load(task_path, weights_only=False)
        except Exception:
            logger.exception("Failed to load task data from %s", task_path)
            continue

        try:
            pdb_path = protein_dir / data["protein_filename"]
            fragment = cast(Mol, data["scaffold"])
            ligand = cast(Mol, data["mol"])
            condition = SamplingCondition(
                protein_pocket_pdb_path=pdb_path,
                fragment=fragment,
                ref_ligand=ligand,
                condition_id=task_path.stem,
            )
        except Exception:
            logger.exception("Failed to prepare task %s", task_path.stem)
            continue

        num_ligand_atoms = condition.ref_ligand.GetNumAtoms()
        if num_ligand_atoms > MAX_EDM_LIGAND_ATOMS:
            logger.warning(
                "Task %s skipped due to ligand having %d atoms, which exceeds the EDM limit of %d.",
                task_path.stem,
                num_ligand_atoms,
                MAX_EDM_LIGAND_ATOMS,
            )
            continue

        task_save_dir = inference_save_dir / task_path.stem
        if task_save_dir.exists():
            shutil.rmtree(task_save_dir)
        task_save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running inference for task %s with sampler %s and weight %s",
            task_path.stem,
            sampler_cfg.name,
            weight_cfg.name,
        )
        original_samples: list[Mol | None] = []
        replaced_samples: list[Mol] = []
        for trial_idx in range(benchmark_cfg.num_trials):
            logger.info("Trial %d for task %s", trial_idx, task_path.stem)
            trial_sampler_cfg = _sampler_cfg_with_seed(sampler_cfg, benchmark_cfg.seed + trial_idx)
            try:
                result = sample_condition(
                    condition=condition,
                    runtime=runtime,
                    sampler_cfg=trial_sampler_cfg,
                    save_dir=None,
                )
            except Exception:
                logger.exception("Trial %d failed for task %s", trial_idx, task_path.stem)
                continue

            original_samples.extend(result.original_samples)
            replaced_samples.extend(result.samples)
            logger.info("Valid samples in trial %d: %d", trial_idx, len(result.samples))

        _write_molecule_samples(replaced_samples, task_save_dir)
        _write_molecule_samples(original_samples, task_save_dir, prefix="original_")

    logger.info("Done!")


def _load_task_paths(processed_data_dir: Path) -> list[Path]:
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"CrossDocked2020 processed data directory does not exist: {processed_data_dir}")
    return sorted(processed_data_dir.glob("*.pt"), key=lambda path: int(path.stem))


def _sampler_cfg_with_seed(sampler_cfg: _BaseSamplerConfig, seed: int) -> _BaseSamplerConfig:
    if is_dataclass(sampler_cfg):
        return replace(sampler_cfg, seed=seed)

    sampler_cfg_copy = copy.copy(sampler_cfg)
    sampler_cfg_copy.seed = seed
    return sampler_cfg_copy


def _make_run_save_dir(
    save_root: Path,
    sampler_cfg: _BaseSamplerConfig,
    weight_cfg: _BaseWeightConfig,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{_sampler_slug(sampler_cfg)}__{_weight_slug(weight_cfg)}"
    run_save_dir = save_root / f"{prefix}_{timestamp}"

    suffix = 1
    while run_save_dir.exists():
        run_save_dir = save_root / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1
    return run_save_dir


def _sampler_slug(sampler_cfg: _BaseSamplerConfig) -> str:
    parts = [
        sampler_cfg.name,
        f"steps{_path_token(sampler_cfg.num_sampling_steps)}",
        f"ode{_path_token(sampler_cfg.ode_start_t)}",
        f"batch{_path_token(sampler_cfg.batch_size)}",
        "logq" if sampler_cfg.use_logq else "nologq",
        "resample" if sampler_cfg.do_resample else "noresample",
    ]
    return "_".join(parts)


def _weight_slug(weight_cfg: _BaseWeightConfig) -> str:
    parts = [weight_cfg.name]
    if is_dataclass(weight_cfg):
        for field in fields(weight_cfg):
            if not field.init:
                continue
            parts.append(f"{field.name}{_path_token(getattr(weight_cfg, field.name))}")
    return "_".join(parts)


def _path_token(value: object) -> str:
    if value is None:
        text = "none"
    elif isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)

    text = text.replace("-", "m").replace(".", "p")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "-" for char in text).strip("-")


def _write_molecule_samples(samples: Sequence[Mol | None], save_dir: Path, prefix: str = "") -> None:
    for idx, sample in enumerate(samples):
        if sample is None:
            continue

        output_path = save_dir / f"{prefix}{idx}.sdf"
        writer = None
        try:
            writer = Chem.SDWriter(str(output_path))
            writer.write(sample)
        except Exception:
            logger.exception("Failed to save molecule sample to %s", output_path)
        finally:
            if writer is not None:
                writer.close()


@hydra.main(config_path="configs", config_name="crossdocked_inference", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.get("output_dir") is not None:
        output_dir = Path(cfg.output_dir)
        logger.info("Using output directory specified in config: %s", output_dir)
    else:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        logger.info("Output directory not specified in config. Using Hydra's default output directory.")

    run_crossdocked_inference(cfg, output_dir=output_dir)


if __name__ == "__main__":
    main()
