from __future__ import annotations

from typing import cast

from experts.geodiff_expert import GeoDiffExpert
from pipelines.base import DataMask, ExpertPipeline
from pipelines.expert_utils import (
    condition_mol_for_source,
    mol_atom_feature_point,
    validate_condition_mol_size,
)
from pipelines.geodiff.components import EXPERT_GEODIFF_QM9, GEODIFF_QM9_FRAGMENT, SCHEDULER_GEODIFF
from sampling.moe_layout import DynamicMoELayout
from sampling.path_factory import PaddedPathScheduler
from sampling.scheduler import GeoDiffScheduler


class GeoDiffPipeline(ExpertPipeline):
    """Pipeline adapter for the pretrained GeoDiff-QM9 expert."""

    expert_keys = (EXPERT_GEODIFF_QM9,)
    scheduler_keys = (SCHEDULER_GEODIFF,)
    component_configs = (GEODIFF_QM9_FRAGMENT,)

    def load_expert(self, *, device: str, component_config) -> GeoDiffExpert:
        return GeoDiffExpert.from_pretrained(device=device)

    def make_scheduler(self, scheduler_key: str) -> PaddedPathScheduler:
        if scheduler_key != SCHEDULER_GEODIFF:
            raise ValueError(f"GeoDiffPipeline does not support scheduler_key {scheduler_key!r}.")
        return GeoDiffScheduler()

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
        expert = cast(GeoDiffExpert, component_runtime.expert)
        num_nodes = layout.num_nodes_for_scope(component.node_scope)
        if component.fixed_atom_type_source is None:
            raise ValueError(
                f"GeoDiff component {component.name} must define fixed_atom_type_source because GeoDiff "
                "scores coordinates conditioned on a fixed molecular graph and atom types."
            )
        fixed_mol = condition_mol_for_source(component, condition, component.fixed_atom_type_source)
        validate_condition_mol_size(component, fixed_mol, expected_num_nodes=num_nodes)
        expert.prepare_data(batch_size, fixed_mol)

    def auxiliary_point(
        self,
        *,
        component_runtime,
        condition,
        runtime,
        layout: DynamicMoELayout,
        auxiliary_mask: DataMask,
        device: str,
    ):
        component = component_runtime.config
        num_nodes = layout.num_nodes_for_scope(component.node_scope)
        if component.fixed_atom_type_source is None:
            return super().auxiliary_point(
                component_runtime=component_runtime,
                condition=condition,
                runtime=runtime,
                layout=layout,
                auxiliary_mask=auxiliary_mask,
                device=device,
            )

        fixed_mol = condition_mol_for_source(component, condition, component.fixed_atom_type_source)
        validate_condition_mol_size(component, fixed_mol, expected_num_nodes=num_nodes)
        return mol_atom_feature_point(
            fixed_mol,
            layout=layout,
            auxiliary_mask=auxiliary_mask,
            device=device,
            atom_type_value=runtime.atom_type_feature_value(default=1.0),
        )
