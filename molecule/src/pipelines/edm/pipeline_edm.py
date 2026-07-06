from __future__ import annotations

import functools
from typing import cast

from configs.config_moe_component import NODE_SCOPE_LIGAND
from experts.edm_expert import EDM_PRETRAINED_GEOM_DRUG, EDM_PRETRAINED_QM9, EDMExpert
from pipelines.base import DataMask, ExpertPipeline
from pipelines.edm.components import (
    EDM_GEOM_DRUG_FRAGMENT,
    EDM_GEOM_DRUG_LIGAND,
    EDM_QM9_FRAGMENT,
    EDM_QM9_LIGAND,
    EXPERT_EDM_GEOM_DRUG,
    EXPERT_EDM_QM9,
    SCHEDULER_EDM,
)
from sampling.moe_layout import DynamicMoELayout
from sampling.path_factory import PaddedPathScheduler
from sampling.sampler import InterleaveFn, PostprocessFn
from sampling.scheduler import EDMScheduler


class EDMPipeline(ExpertPipeline):
    """Pipeline adapter for pretrained EDM experts."""

    expert_keys = (EXPERT_EDM_QM9, EXPERT_EDM_GEOM_DRUG)
    scheduler_keys = (SCHEDULER_EDM,)
    component_configs = (EDM_QM9_FRAGMENT, EDM_QM9_LIGAND, EDM_GEOM_DRUG_FRAGMENT, EDM_GEOM_DRUG_LIGAND)

    def load_expert(self, *, device: str, component_config) -> EDMExpert:
        if component_config.expert_key == EXPERT_EDM_GEOM_DRUG:
            return EDMExpert.from_pretrained(device=device, pretrained_model=EDM_PRETRAINED_GEOM_DRUG)
        if component_config.expert_key == EXPERT_EDM_QM9:
            return EDMExpert.from_pretrained(device=device, pretrained_model=EDM_PRETRAINED_QM9)
        raise ValueError(f"EDMPipeline does not support expert_key {component_config.expert_key!r}.")

    def make_scheduler(self, scheduler_key: str) -> PaddedPathScheduler:
        if scheduler_key != SCHEDULER_EDM:
            raise ValueError(f"EDMPipeline does not support scheduler_key {scheduler_key!r}.")
        return EDMScheduler()

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
        expert = cast(EDMExpert, component_runtime.expert)
        num_nodes = layout.num_nodes_for_scope(component.node_scope)
        expert.prepare_data(batch_size, num_nodes)

    def interleave_fn(
        self,
        *,
        component_runtime,
        layout: DynamicMoELayout,
        active_mask: DataMask,
    ) -> InterleaveFn:
        component = component_runtime.config
        expert = cast(EDMExpert, component_runtime.expert)
        coord_mask = layout.coordinate_mask_for_scope(component.node_scope)
        num_nodes = layout.num_nodes_for_scope(component.node_scope)
        return functools.partial(expert.interleave, coord_mask=coord_mask, num_nodes=num_nodes)

    def postprocess_fn(
        self,
        *,
        component_runtime,
        layout: DynamicMoELayout,
        active_mask: DataMask,
    ) -> PostprocessFn:
        component = component_runtime.config
        expert = cast(EDMExpert, component_runtime.expert)
        categorical_mask = layout.atom_type_mask_for_component(component)
        # FIXME: This is a hack to avoid postprocessing categorical features twice when mixing two EDM experts.
        if component.node_scope != NODE_SCOPE_LIGAND:
            categorical_mask = None
        return functools.partial(expert.postprocess, categorical_mask=categorical_mask)

    def atom_type_feature_value(self, component_runtime) -> float | None:
        expert = cast(EDMExpert, component_runtime.expert)
        return 1.0 / float(expert.model.norm_values[1])
