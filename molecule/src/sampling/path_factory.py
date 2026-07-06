from typing import Protocol

import torch
from jaxtyping import Bool, Float

from sampling.distributions import MarginalScheduler, PointMassDistribution
from sampling.moe_layout import ComponentFeatureAdapter
from sampling.probability_path import PaddedProbabilityPath, ProbabilityPath, SDEScheduler

DataMask = Bool[torch.Tensor, "data"]  # noqa: F821
DataVector = Float[torch.Tensor, "data"]  # noqa: F821
DataFeatureMatrix = Float[torch.Tensor, "num_nodes feature"]
FragmentFeatureMatrix = Float[torch.Tensor, "frag feature"]
LigandFeatureMatrix = Float[torch.Tensor, "lig feature"]
AuxiliaryPoint = DataVector | DataFeatureMatrix | FragmentFeatureMatrix | LigandFeatureMatrix


class ScoringExpert(Protocol):
    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]: ...


class PaddedPathScheduler(SDEScheduler, MarginalScheduler, Protocol):
    pass


class FeatureOrderScoreAdapter:
    """Call an expert in native feature order while exposing global feature order."""

    def __init__(self, expert: ScoringExpert, feature_adapter: ComponentFeatureAdapter):
        self.expert = expert
        self.feature_adapter = feature_adapter

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        x_native = self.feature_adapter.to_native(x)
        score_native = self.expert.score(t, x_native)
        return self.feature_adapter.to_global(score_native)


def make_zero_auxiliary_point(
    num_nodes: int,
    auxiliary_mask: DataMask,
    device: str,
) -> DataFeatureMatrix:
    """Create a per-node zero point for a fixed auxiliary path."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    num_auxiliary_values = int(auxiliary_mask.sum().item())
    if num_auxiliary_values % num_nodes != 0:
        raise ValueError("auxiliary_mask true count must be divisible by num_nodes.")

    return torch.zeros(num_nodes, num_auxiliary_values // num_nodes, device=device)


def build_padded_expert_path(
    *,
    expert: ScoringExpert,
    scheduler: PaddedPathScheduler,
    active_mask: DataMask,
    auxiliary_mask: DataMask,
    auxiliary_point: AuxiliaryPoint,
    device: str,
    feature_adapter: ComponentFeatureAdapter,
    reverse: bool = True,
) -> PaddedProbabilityPath:
    """Build an expert score path padded with a fixed-point auxiliary path."""
    _validate_masks(active_mask, auxiliary_mask)

    auxiliary_point = auxiliary_point.flatten()
    if auxiliary_point.numel() != int(auxiliary_mask.sum().item()):
        raise ValueError("auxiliary_point size must match auxiliary_mask true count.")

    score_expert = FeatureOrderScoreAdapter(expert, feature_adapter)
    main_path = ProbabilityPath(scheduler, score_expert.score, reverse=reverse)
    auxiliary_distribution = PointMassDistribution(auxiliary_point, device=device)
    auxiliary_score = auxiliary_distribution.export_score_function(scheduler)
    auxiliary_path = ProbabilityPath(scheduler, auxiliary_score, reverse=reverse)

    return PaddedProbabilityPath([main_path, auxiliary_path], [active_mask, auxiliary_mask])


def _validate_masks(active_mask: DataMask, auxiliary_mask: DataMask) -> None:
    if active_mask.dtype != torch.bool or auxiliary_mask.dtype != torch.bool:
        raise TypeError("active_mask and auxiliary_mask must be boolean tensors.")
    if active_mask.shape != auxiliary_mask.shape:
        raise ValueError("active_mask and auxiliary_mask must have the same shape.")
    if torch.any(active_mask & auxiliary_mask):
        raise ValueError("active_mask and auxiliary_mask must be disjoint.")
    if not torch.all(active_mask | auxiliary_mask):
        raise ValueError("active_mask and auxiliary_mask must cover the full padded path.")
