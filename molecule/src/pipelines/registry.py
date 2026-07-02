from __future__ import annotations

from pipelines.base import ExpertPipeline
from sampling.path_factory import PaddedPathScheduler


class ExpertPipelineRegistry:
    """Registry for expert-family pipelines and their scheduler factories."""

    def __init__(self) -> None:
        self._pipelines_by_expert_key: dict[str, ExpertPipeline] = {}
        self._pipelines_by_scheduler_key: dict[str, ExpertPipeline] = {}

    def register(self, pipeline: ExpertPipeline) -> None:
        for expert_key in pipeline.expert_keys:
            if expert_key in self._pipelines_by_expert_key:
                raise ValueError(f"Duplicate ExpertPipeline registration for expert_key {expert_key!r}.")
            self._pipelines_by_expert_key[expert_key] = pipeline

        for scheduler_key in pipeline.scheduler_keys:
            if scheduler_key in self._pipelines_by_scheduler_key:
                raise ValueError(f"Duplicate scheduler registration for scheduler_key {scheduler_key!r}.")
            self._pipelines_by_scheduler_key[scheduler_key] = pipeline

    def pipeline_for_expert_key(self, expert_key: str) -> ExpertPipeline:
        try:
            return self._pipelines_by_expert_key[expert_key]
        except KeyError as exc:
            available = ", ".join(sorted(self._pipelines_by_expert_key))
            raise NotImplementedError(
                f"No ExpertPipeline registered for expert_key {expert_key!r}. Available expert keys: {available}."
            ) from exc

    def make_scheduler(self, scheduler_key: str) -> PaddedPathScheduler:
        try:
            pipeline = self._pipelines_by_scheduler_key[scheduler_key]
        except KeyError as exc:
            available = ", ".join(sorted(self._pipelines_by_scheduler_key))
            raise NotImplementedError(
                f"No ExpertPipeline registered for scheduler_key {scheduler_key!r}. "
                f"Available scheduler keys: {available}."
            ) from exc
        return pipeline.make_scheduler(scheduler_key)


def get_default_pipeline_registry() -> ExpertPipelineRegistry:
    from pipelines.diffsbdd.pipeline_diffsbdd import DiffSBDDPipeline
    from pipelines.edm.pipeline_edm import EDMPipeline
    from pipelines.geodiff.pipeline_geodiff import GeoDiffPipeline

    registry = ExpertPipelineRegistry()
    registry.register(EDMPipeline())
    registry.register(GeoDiffPipeline())
    registry.register(DiffSBDDPipeline())
    return registry
