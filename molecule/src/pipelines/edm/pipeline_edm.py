from __future__ import annotations

import functools
from typing import cast

from configs.config_moe_component import EXPERT_EDM_GEOM_DRUG, EXPERT_EDM_QM9, NODE_SCOPE_LIGAND, SCHEDULER_EDM
from experts.edm_expert import EDMExpert
from pipelines.base import DataMask, ExpertPipeline
from sampling.moe_layout import DynamicMoELayout
from sampling.path_factory import PaddedPathScheduler
from sampling.sampler import PostprocessFn
from sampling.scheduler import EDMScheduler


class EDMPipeline(ExpertPipeline):
    """Pipeline adapter for the pretrained EDM-QM9 expert."""

    expert_keys = (EXPERT_EDM_QM9, EXPERT_EDM_GEOM_DRUG)
    scheduler_keys = (SCHEDULER_EDM,)

    # TODO: support both EDM-QM9 and EDM-GEOM-DRUG, but the latter is not yet implemented. For now, we only support EDM-QM9.
    def load_expert(self, *, device: str, component_config) -> EDMExpert:
        if component_config.expert_key == EXPERT_EDM_GEOM_DRUG:
            raise NotImplementedError("EDM_GEOM_DRUG is registered as a component spec, but its loader is not ready.")
        return EDMExpert.from_pretrained(device=device)

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

    def postprocess_fn(
        self,
        *,
        component_runtime,
        layout: DynamicMoELayout,
        active_mask: DataMask,
    ) -> PostprocessFn:
        component = component_runtime.config
        expert = cast(EDMExpert, component_runtime.expert)
        h_mask = layout.h_atom_type_mask_for_scope(component.node_scope)
        # FIXME: This is a hack to avoid prostprocessing hydrogens twice when mixing two EDM experts.
        if component.node_scope != NODE_SCOPE_LIGAND:
            h_mask = None
        return functools.partial(expert.postprocess, h_mask=h_mask)

    def atom_type_feature_value(self, component_runtime) -> float | None:
        expert = cast(EDMExpert, component_runtime.expert)
        return 1.0 / float(expert.model.norm_values[1])
