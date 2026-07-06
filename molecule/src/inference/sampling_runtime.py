import logging
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from jaxtyping import Float
from omegaconf import DictConfig, ListConfig, OmegaConf

from configs.config_moe import MoEExponentConfig
from configs.config_moe_component import MoEComponentConfig
from configs.config_weight import _BaseWeightConfig
from experts.base_expert import MoEExpertABC, SBDDExpert
from pipelines import ExpertPipeline, get_default_pipeline_registry
from pipelines.registry import ExpertPipelineRegistry
from sampling.path_factory import PaddedPathScheduler

logger = logging.getLogger(__name__)

ExponentFunctionType = Callable[[Float[torch.Tensor, "B 1"]], Float[torch.Tensor, "B 1"]]
ComponentConfigEntry = tuple[str, MoEComponentConfig]


@dataclass
class ComponentRuntime:
    """Loaded expert, scheduler, and config for one MoE component."""

    component_id: str
    config: MoEComponentConfig
    pipeline: ExpertPipeline
    expert: MoEExpertABC
    scheduler: PaddedPathScheduler


@dataclass
class SamplingRuntime:
    """Long-lived sampling components shared across one or more conditions."""

    components: list[ComponentRuntime]
    exponent_list: list[ExponentFunctionType]
    global_scheduler_key: str
    global_scheduler: PaddedPathScheduler

    def get_component(self, name: str) -> ComponentRuntime:
        for component in self.components:
            if component.component_id == name or component.config.name == name:
                return component
        available = ", ".join(f"{component.component_id} ({component.config.name})" for component in self.components)
        raise KeyError(f"MoE component {name!r} was not loaded. Available components: {available}.")

    def sbdd_expert(self) -> SBDDExpert | None:
        for component in self.components:
            sbdd_expert = component.pipeline.sbdd_expert(component)
            if sbdd_expert is not None:
                return sbdd_expert
        return None

    def atom_type_feature_value(self, default: float = 1.0) -> float:
        for component in self.components:
            value = component.pipeline.atom_type_feature_value(component)
            if value is not None:
                return value
        return default


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


def build_exponent_list(
    component_configs: tuple[ComponentConfigEntry, ...],
    exponent_configs: Mapping[str, MoEExponentConfig],
) -> list[ExponentFunctionType]:
    exponent_by_component = dict(exponent_configs)
    exponent_list: list[ExponentFunctionType] = []
    for component_id, component_cfg in component_configs:
        exponent_cfg = exponent_by_component.pop(component_id, None)
        if exponent_cfg is None:
            raise ValueError(f"Missing exponent config for MoE component {component_id!r} ({component_cfg.name}).")

        exponent_list.append(_make_exponent_function(exponent_cfg.weight_fn, exponent_cfg))

    if exponent_by_component:
        unknown_components = ", ".join(sorted(exponent_by_component))
        raise ValueError(f"Exponent configs reference unloaded MoE components: {unknown_components}.")

    return exponent_list


def _make_exponent_function(
    weight_cfg: _BaseWeightConfig,
    exponent_cfg: MoEExponentConfig,
) -> ExponentFunctionType:
    weight_scale = float(exponent_cfg.weight_scale)
    constant = float(exponent_cfg.constant)
    weight_fn = weight_cfg.weight_function

    def _exponent_fn(t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        exponent = weight_scale * weight_fn(t) + constant
        return exponent

    return _exponent_fn


def log_exponent_list(
    exponent_list: list[ExponentFunctionType],
    component_names: list[str],
) -> None:
    if len(exponent_list) != len(component_names):
        raise ValueError(
            "The number of exponent functions must match the number of component names: "
            f"{len(exponent_list)} != {len(component_names)}."
        )

    logger.info("Using exponent functions:")
    logger.info("-" * 50)
    for exponent_fn, exponent_id in zip(exponent_list, component_names):
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.0])).item():.4f} (at t=0.0)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.25])).item():.4f} (at t=0.25)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.5])).item():.4f} (at t=0.5)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([0.75])).item():.4f} (at t=0.75)")
        logger.info(f"{exponent_id}: {exponent_fn(torch.tensor([1.0])).item():.4f} (at t=1.0)")
        logger.info("-" * 50)


def load_sampling_runtime(
    device: str,
    component_configs: list[ComponentConfigEntry],
    global_scheduler_key: str,
    exponent_configs: Mapping[str, MoEExponentConfig],
) -> SamplingRuntime:
    if not component_configs:
        raise ValueError("At least one MoE component must be provided by config.")

    component_entries = tuple(component_configs)
    exponent_list = build_exponent_list(component_entries, exponent_configs)

    logger.info("Loading %d MoE components...", len(component_entries))
    registry = get_default_pipeline_registry()
    components = [
        _load_component_runtime(component_id, component_cfg, device=device, registry=registry)
        for component_id, component_cfg in component_entries
    ]

    return SamplingRuntime(
        components=components,
        exponent_list=exponent_list,
        global_scheduler_key=global_scheduler_key,
        global_scheduler=registry.make_scheduler(global_scheduler_key),
    )


def _load_component_runtime(
    component_id: str,
    component_cfg: MoEComponentConfig,
    device: str,
    registry: ExpertPipelineRegistry,
) -> ComponentRuntime:
    logger.info(
        "Loading MoE component %s (%s, %s)...",
        component_id,
        component_cfg.expert_key,
        component_cfg.scheduler_key,
    )

    pipeline = registry.pipeline_for_expert_key(component_cfg.expert_key)
    return ComponentRuntime(
        component_id=component_id,
        config=component_cfg,
        pipeline=pipeline,
        expert=pipeline.load_expert(device=device, component_config=component_cfg),
        scheduler=registry.make_scheduler(component_cfg.scheduler_key),
    )


def component_configs_from_hydra(
    components_cfg: DictConfig | ListConfig | Mapping[str, object] | list[object],
) -> list[ComponentConfigEntry]:
    components_obj = (
        OmegaConf.to_object(components_cfg) if isinstance(components_cfg, DictConfig | ListConfig) else components_cfg
    )
    if isinstance(components_obj, dict):
        return [
            (str(component_id), _as_component_config(component)) for component_id, component in components_obj.items()
        ]
    if isinstance(components_obj, list):
        components = [_as_component_config(component) for component in components_obj]
        return [(component.name, component) for component in components]

    raise TypeError(f"Unsupported MoE components config type: {type(components_obj).__name__}.")


def _as_component_config(component: object) -> MoEComponentConfig:
    if isinstance(component, MoEComponentConfig):
        return _normalize_component_config(component)
    if isinstance(component, dict):
        return _normalize_component_config(MoEComponentConfig(**component))
    raise TypeError(f"Unsupported MoE component config entry type: {type(component).__name__}.")


def _normalize_component_config(component: MoEComponentConfig) -> MoEComponentConfig:
    return replace(
        component,
        supported_atoms=tuple(component.supported_atoms),
        condition_keys=tuple(component.condition_keys),
        scored_atoms=tuple(component.scored_atoms),
    )
