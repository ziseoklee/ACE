from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool

from experts.base_expert import MoEExpertABC, SBDDExpert
from sampling.path_factory import PaddedPathScheduler, make_zero_auxiliary_point
from sampling.sampler import InterleaveFn, PostprocessFn

if TYPE_CHECKING:
    from configs.config_moe_component import MoEComponentConfig
    from inference.condition_sampling import SamplingCondition
    from inference.sampling_runtime import ComponentRuntime, SamplingRuntime
    from sampling.moe_layout import DynamicMoELayout

DataMask = Bool[torch.Tensor, "data"]  # noqa: F821


class ExpertPipeline(ABC):
    """Adapter that registers one expert family into the MoE sampling runtime."""

    expert_keys: tuple[str, ...]
    scheduler_keys: tuple[str, ...]
    component_configs: tuple[MoEComponentConfig, ...] = ()

    @abstractmethod
    def load_expert(self, *, device: str, component_config: MoEComponentConfig) -> MoEExpertABC:
        """Load the pretrained expert wrapped by this pipeline."""
        ...

    @abstractmethod
    def make_scheduler(self, scheduler_key: str) -> PaddedPathScheduler:
        """Create a scheduler supported by this pipeline."""
        ...

    @abstractmethod
    def prepare_data(
        self,
        *,
        component_runtime: ComponentRuntime,
        condition: SamplingCondition,
        layout: DynamicMoELayout,
        batch_size: int,
        num_ligand_atoms: int,
    ) -> None:
        """Prepare condition-specific expert inputs."""
        ...

    def auxiliary_point(
        self,
        *,
        component_runtime: ComponentRuntime,
        condition: SamplingCondition,
        runtime: SamplingRuntime,
        layout: DynamicMoELayout,
        auxiliary_mask: DataMask,
        device: str,
    ) -> torch.Tensor:
        """Return the fixed auxiliary point used to pad this expert path."""
        component = component_runtime.config
        num_nodes = layout.num_nodes_for_scope(component.node_scope)
        return make_zero_auxiliary_point(num_nodes, auxiliary_mask, device=device)

    def interleave_fn(
        self,
        *,
        component_runtime: ComponentRuntime,
        layout: DynamicMoELayout,
        active_mask: DataMask,
    ) -> InterleaveFn:
        """Return the sampler interleave hook for this component."""
        return component_runtime.expert.interleave

    def postprocess_fn(
        self,
        *,
        component_runtime: ComponentRuntime,
        layout: DynamicMoELayout,
        active_mask: DataMask,
    ) -> PostprocessFn:
        """Return the sampler postprocess hook for this component."""
        return component_runtime.expert.postprocess

    def sbdd_expert(self, component_runtime: ComponentRuntime) -> SBDDExpert | None:
        """Return an SBDD-capable expert when this component conditions on a pocket."""
        return None

    def atom_type_feature_value(self, component_runtime: ComponentRuntime) -> float | None:
        """Return the current MoE atom-type one-hot value if this pipeline owns that scale."""
        return None
