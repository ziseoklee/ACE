import logging
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float

from configs.config_weight import ACEBumpWeightConfig, _BaseWeightConfig
from experts import DiffSBDDExpert, EDMExpert, GeoDiffExpert
from sampling.scheduler import DiffSBDDScheduler, EDMScheduler, GeoDiffScheduler

logger = logging.getLogger(__name__)

ExponentFunctionType = Callable[[Float[torch.Tensor, "B 1"]], Float[torch.Tensor, "B 1"]]


@dataclass
class SamplingRuntime:
    """Long-lived sampling components shared across one or more conditions."""

    edm_fragment: EDMExpert
    edm_ligand: EDMExpert
    geodiff: GeoDiffExpert
    diffsbdd: DiffSBDDExpert
    scheduler_edm: EDMScheduler
    scheduler_geodiff: GeoDiffScheduler
    scheduler_sbdd: DiffSBDDScheduler
    exponent_list: list[ExponentFunctionType]


def seed_everything(seed: int) -> None:
    # FIXME: CRITICAL: These MUST be set at the absolute top of the file,
    # BEFORE importing random, numpy, or torch.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_exponent_list(weight_cfg: _BaseWeightConfig) -> list[ExponentFunctionType]:
    # Expert order: gamma_1=edm_fragment, gamma_2=edm_ligand, gamma_3=geodiff, gamma_4=diffsbdd.
    # NOTE: For ACEBumpWeightConfig, the bump function is only applied to gamma_4.
    if isinstance(weight_cfg, ACEBumpWeightConfig):
        omega = weight_cfg.omega
        return [
            lambda t: torch.zeros_like(t) - omega,  # gamma_1
            lambda t: torch.zeros_like(t) - (omega - 1),  # gamma_2
            lambda t: torch.zeros_like(t) + omega,  # gamma_3
            lambda t: torch.zeros_like(t) + weight_cfg.weight_function(t),  # gamma_4
        ]

    weight_fn = weight_cfg.weight_function
    return [
        lambda t: torch.zeros_like(t) - weight_fn(t),  # gamma_1
        lambda t: torch.zeros_like(t) - (weight_fn(t) - 1),  # gamma_2
        lambda t: torch.zeros_like(t) + weight_fn(t),  # gamma_3
        lambda t: torch.zeros_like(t) + weight_fn(t),  # gamma_4
    ]


def log_exponent_list(exponent_list: list[ExponentFunctionType]) -> None:
    exponent_ids = ["gamma_1", "gamma_2", "gamma_3", "gamma_4"]
    logger.info("Using exponent functions:")
    logger.info("-" * 50)
    for exponent_fn, exponent_id in zip(exponent_list, exponent_ids):
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.0])).item():.4f} (at t=0.0)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.25])).item():.4f} (at t=0.25)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.5])).item():.4f} (at t=0.5)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.75])).item():.4f} (at t=0.75)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([1.0])).item():.4f} (at t=1.0)")
        logger.info("-" * 50)


def load_sampling_runtime(weight_cfg: _BaseWeightConfig, device: str) -> SamplingRuntime:
    logger.info("Loading schedulers...")
    scheduler_geodiff = GeoDiffScheduler()
    scheduler_edm = EDMScheduler()
    scheduler_sbdd = DiffSBDDScheduler()

    logger.info("Loading experts...")
    edm_fragment = EDMExpert.from_pretrained(device=device)
    edm_ligand = EDMExpert.from_pretrained(device=device)
    geodiff = GeoDiffExpert.from_pretrained(device=device)
    diffsbdd = DiffSBDDExpert.from_pretrained(device=device)

    return SamplingRuntime(
        edm_fragment=edm_fragment,
        edm_ligand=edm_ligand,
        geodiff=geodiff,
        diffsbdd=diffsbdd,
        scheduler_edm=scheduler_edm,
        scheduler_geodiff=scheduler_geodiff,
        scheduler_sbdd=scheduler_sbdd,
        exponent_list=build_exponent_list(weight_cfg),
    )
