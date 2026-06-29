from typing import Protocol

import torch
from jaxtyping import Bool, Float

from sampling.distributions import PointMassDistribution
from sampling.moe_layout import CrossDockedMoEMasks
from sampling.probability_path import PaddedProbabilityPath, ProbabilityPath
from sampling.scheduler import SchedulerABC

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
    scheduler: SchedulerABC,
    active_mask: DataMask,
    auxiliary_mask: DataMask,
    auxiliary_point: AuxiliaryPoint,
    device: str,
    reverse: bool = True,
) -> PaddedProbabilityPath:
    """Build an expert score path padded with a fixed-point auxiliary path."""
    _validate_masks(active_mask, auxiliary_mask)

    auxiliary_point = auxiliary_point.flatten()
    if auxiliary_point.numel() != int(auxiliary_mask.sum().item()):
        raise ValueError("auxiliary_point size must match auxiliary_mask true count.")

    main_path = ProbabilityPath(scheduler, expert.score, reverse=reverse)
    auxiliary_distribution = PointMassDistribution(auxiliary_point, device=device)
    auxiliary_score = auxiliary_distribution.export_score_function(scheduler)
    auxiliary_path = ProbabilityPath(scheduler, auxiliary_score, reverse=reverse)

    return PaddedProbabilityPath([main_path, auxiliary_path], [active_mask, auxiliary_mask])


def build_geodiff_fragment_path(
    *,
    expert: ScoringExpert,
    scheduler: SchedulerABC,
    masks: CrossDockedMoEMasks,
    atom_type_and_charge_point: FragmentFeatureMatrix,
    device: str,
) -> PaddedProbabilityPath:
    """Build the GeoDiff fragment coordinate path with fixed fragment atom types."""
    return build_padded_expert_path(
        expert=expert,
        scheduler=scheduler,
        active_mask=masks.geodiff_fragment_coords,
        auxiliary_mask=masks.geodiff_fragment_atom_types_and_charge,
        auxiliary_point=atom_type_and_charge_point,
        device=device,
    )


def build_edm_fragment_path(
    *,
    expert: ScoringExpert,
    scheduler: SchedulerABC,
    masks: CrossDockedMoEMasks,
    padding_point: FragmentFeatureMatrix,
    device: str,
) -> PaddedProbabilityPath:
    """Build the fragment EDM path with a fixed padding point."""
    return build_padded_expert_path(
        expert=expert,
        scheduler=scheduler,
        active_mask=masks.edm_fragment_xh,
        auxiliary_mask=masks.edm_fragment_padding,
        auxiliary_point=padding_point,
        device=device,
    )


def build_edm_ligand_path(
    *,
    expert: ScoringExpert,
    scheduler: SchedulerABC,
    masks: CrossDockedMoEMasks,
    padding_point: LigandFeatureMatrix,
    device: str,
) -> PaddedProbabilityPath:
    """Build the whole-ligand EDM path with a fixed padding point."""
    return build_padded_expert_path(
        expert=expert,
        scheduler=scheduler,
        active_mask=masks.edm_ligand_xh,
        auxiliary_mask=masks.edm_ligand_padding,
        auxiliary_point=padding_point,
        device=device,
    )


def build_diffsbdd_ligand_path(
    *,
    expert: ScoringExpert,
    scheduler: SchedulerABC,
    masks: CrossDockedMoEMasks,
    padding_point: LigandFeatureMatrix,
    device: str,
) -> PaddedProbabilityPath:
    """Build the DiffSBDD ligand path with a fixed padding point."""
    return build_padded_expert_path(
        expert=expert,
        scheduler=scheduler,
        active_mask=masks.diffsbdd_ligand_xh,
        auxiliary_mask=masks.diffsbdd_ligand_padding,
        auxiliary_point=padding_point,
        device=device,
    )


def _validate_masks(active_mask: DataMask, auxiliary_mask: DataMask) -> None:
    if active_mask.dtype != torch.bool or auxiliary_mask.dtype != torch.bool:
        raise TypeError("active_mask and auxiliary_mask must be boolean tensors.")
    if active_mask.shape != auxiliary_mask.shape:
        raise ValueError("active_mask and auxiliary_mask must have the same shape.")
    if torch.any(active_mask & auxiliary_mask):
        raise ValueError("active_mask and auxiliary_mask must be disjoint.")
    if not torch.all(active_mask | auxiliary_mask):
        raise ValueError("active_mask and auxiliary_mask must cover the full padded path.")
