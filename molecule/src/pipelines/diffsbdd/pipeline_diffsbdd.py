from __future__ import annotations

import functools
from typing import cast

import torch

from configs.config_moe_component import CONDITION_POCKET, NODE_SCOPE_LIGAND
from experts.base_expert import SBDDExpert
from experts.diffsbdd_expert import DiffSBDDExpert
from pipelines.base import DataMask, ExpertPipeline
from pipelines.diffsbdd.components import (
    DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
    EXPERT_DIFFSBDD_CROSSDOCKED_FULLATOM_COND,
    SCHEDULER_DIFFSBDD,
)
from pipelines.expert_utils import require_condition_key
from sampling.moe_layout import DynamicMoELayout
from sampling.path_factory import PaddedPathScheduler
from sampling.sampler import InterleaveFn, PostprocessFn
from sampling.scheduler import DiffSBDDScheduler


class DiffSBDDPipeline(ExpertPipeline):
    """Pipeline adapter for the pretrained DiffSBDD CrossDocked expert."""

    expert_keys = (EXPERT_DIFFSBDD_CROSSDOCKED_FULLATOM_COND,)
    scheduler_keys = (SCHEDULER_DIFFSBDD,)
    component_configs = (DIFFSBDD_CROSSDOCKED_FULLATOM_COND,)

    def load_expert(self, *, device: str, component_config) -> DiffSBDDExpert:
        return DiffSBDDExpert.from_pretrained(device=device)

    def make_scheduler(self, scheduler_key: str) -> PaddedPathScheduler:
        if scheduler_key != SCHEDULER_DIFFSBDD:
            raise ValueError(f"DiffSBDDPipeline does not support scheduler_key {scheduler_key!r}.")
        return DiffSBDDScheduler()

    def prepare_data(
        self,
        *,
        component_runtime,
        condition,
        layout: DynamicMoELayout,
        batch_size: int,
        num_ligand_atoms: int,
    ) -> None:
        component = component_runtime.config
        expert = cast(DiffSBDDExpert, component_runtime.expert)
        require_condition_key(component, CONDITION_POCKET)
        if component.node_scope != NODE_SCOPE_LIGAND:
            raise ValueError(f"DiffSBDD component {component.name} must use node_scope={NODE_SCOPE_LIGAND!r}.")
        expert.prepare_data(
            batch_size,
            num_ligand_atoms,
            condition.protein_pocket_pdb_path,
            condition.ref_ligand,
        )

    def interleave_fn(
        self,
        *,
        component_runtime,
        layout: DynamicMoELayout,
        active_mask: DataMask,
    ) -> InterleaveFn:
        expert = cast(DiffSBDDExpert, component_runtime.expert)
        return functools.partial(expert.interleave, mask=active_mask)

    def postprocess_fn(
        self,
        *,
        component_runtime,
        layout: DynamicMoELayout,
        active_mask: DataMask,
    ) -> PostprocessFn:
        expert = cast(DiffSBDDExpert, component_runtime.expert)
        feature_adapter = layout.feature_adapter_for_component(component_runtime.config)

        def _postprocess(x):
            x_out = x.clone()
            x_native = feature_adapter.to_native(x_out[..., active_mask])
            native_mask = torch.ones(x_native.shape[-1], dtype=torch.bool, device=x.device)
            x_native = expert.postprocess(x_native, mask=native_mask)
            x_out[..., active_mask] = feature_adapter.to_global(x_native)
            return x_out

        return _postprocess

    def sbdd_expert(self, component_runtime) -> SBDDExpert | None:
        return cast(DiffSBDDExpert, component_runtime.expert)
