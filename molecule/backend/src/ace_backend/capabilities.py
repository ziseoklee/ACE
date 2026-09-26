"""Readiness checks without loading model weights or running ACE sampling."""

import importlib
import logging
import pickle
import subprocess
from collections.abc import Callable
from pathlib import Path

from ace_backend.schemas import (
    AvailableFeature,
    Capabilities,
    EvaluationCapabilities,
    FeatureAvailability,
    FeatureName,
    UnavailableFeature,
)
from ace_backend.settings import Settings

logger = logging.getLogger(__name__)


def check_cuda(device: str) -> FeatureAvailability:
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        logger.warning("PyTorch could not be imported", exc_info=True)
        return UnavailableFeature(reason="dependency_unavailable")

    try:
        device_index = int(device.split(":")[1])
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            return UnavailableFeature(reason="cuda_unavailable")
        with torch.cuda.device(device):
            torch.cuda.init()
    except (OSError, RuntimeError):
        logger.warning("Selected CUDA device could not be initialized", exc_info=True)
        return UnavailableFeature(reason="cuda_unavailable")

    try:
        from torch_cluster import radius_graph
        from torch_scatter import scatter_add
    except (ImportError, OSError, RuntimeError):
        logger.warning("CUDA graph extensions could not be imported", exc_info=True)
        return UnavailableFeature(reason="dependency_unavailable")

    try:
        with torch.cuda.device(device):
            matrix = torch.ones((2, 2), device=device)
            if not torch.equal(matrix @ matrix, torch.full_like(matrix, 2.0)):
                return UnavailableFeature(reason="cuda_incompatible")
            source = torch.tensor([1.0, 2.0, 3.0], device=device)
            index = torch.tensor([0, 1, 0], device=device)
            if scatter_add(source, index).cpu().tolist() != [4.0, 2.0]:
                return UnavailableFeature(reason="cuda_incompatible")
            positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
            edges = radius_graph(positions, r=1.5, loop=False)
            if set(map(tuple, edges.t().cpu().tolist())) != {(0, 1), (1, 0)}:
                return UnavailableFeature(reason="cuda_incompatible")
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("Not enough memory for the CUDA readiness check", exc_info=True)
        return UnavailableFeature(reason="cuda_unavailable")
    except (OSError, RuntimeError):
        logger.warning("CUDA readiness kernels could not execute", exc_info=True)
        return UnavailableFeature(reason="cuda_incompatible")
    return AvailableFeature()


def inference_checkpoint_paths() -> tuple[Path, ...]:
    """Use the same four-expert assets shared by all scaffold v1 presets."""
    from experts.diffsbdd_expert import DIFFSBDD_CKPT_PATH
    from experts.edm_expert import EDM_PRETRAINED_GEOM_DRUG, EDM_PRETRAINED_SPECS
    from experts.geodiff_expert import GEODIFF_CKPT_PATH, GEODIFF_CONFIG_PATH

    edm = EDM_PRETRAINED_SPECS[EDM_PRETRAINED_GEOM_DRUG]
    return edm.checkpoint_path, edm.model_config_path, GEODIFF_CKPT_PATH, GEODIFF_CONFIG_PATH, DIFFSBDD_CKPT_PATH


def check_inference(device: str) -> FeatureAvailability:
    cuda = check_cuda(device)
    if not cuda.available:
        return cuda
    try:
        importlib.import_module("inference.condition_sampling")
        paths = inference_checkpoint_paths()
    except (ImportError, OSError, RuntimeError):
        logger.warning("ACE inference dependencies could not be imported", exc_info=True)
        return UnavailableFeature(reason="dependency_unavailable")

    for path in paths:
        try:
            with path.open("rb") as checkpoint:
                if not checkpoint.read(1):
                    return UnavailableFeature(reason="checkpoint_missing")
        except OSError:
            logger.warning("Required model asset is unreadable: %s", path, exc_info=True)
            return UnavailableFeature(reason="checkpoint_missing")
    return AvailableFeature()


def check_druglikeness() -> FeatureAvailability:
    try:
        module = importlib.import_module("evaluation.metrics.druglikeness")
        if module.sascorer is None:
            return UnavailableFeature(reason="dependency_unavailable")
        # Import alone does not check that the SA fragment-score data can be read.
        module.sascorer.readFragmentScores()
    except (ImportError, OSError, RuntimeError, pickle.PickleError, EOFError):
        logger.warning("Druglikeness dependencies are unavailable", exc_info=True)
        return UnavailableFeature(reason="dependency_unavailable")
    return AvailableFeature()


def check_scaffold_preservation() -> FeatureAvailability:
    try:
        importlib.import_module("rdkit.Chem")
    except (ImportError, OSError, RuntimeError):
        logger.warning("RDKit is unavailable", exc_info=True)
        return UnavailableFeature(reason="dependency_unavailable")
    return AvailableFeature()


def check_docking() -> FeatureAvailability:
    try:
        backend = importlib.import_module("evaluation.backends.qvina")
        for command in ([str(backend.PATH_QVINA2), "--version"], ["obabel", "-V"]):
            subprocess.run(command, check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError):
        logger.warning("Docking dependencies are unavailable", exc_info=True)
        return UnavailableFeature(reason="dependency_unavailable")
    return AvailableFeature()


def detect_capabilities(settings: Settings) -> Capabilities:
    def enabled_check(feature: FeatureName, check: Callable[[], FeatureAvailability]) -> FeatureAvailability:
        if feature in settings.disabled_features:
            return UnavailableFeature(reason="feature_not_enabled")
        return check()

    return Capabilities(
        inference=enabled_check("inference", lambda: check_inference(settings.device)),
        evaluation=EvaluationCapabilities(
            druglikeness=enabled_check("druglikeness", check_druglikeness),
            scaffold_preservation=enabled_check("scaffold_preservation", check_scaffold_preservation),
            docking=enabled_check("docking", check_docking),
        ),
        limits=settings.limits,
    )
