from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class _BaseWeightConfig:
    """
    Base configuration for weight functions used in sampling.
    It is a time varying function that takes in the current time step and returns a weight value.
    t ∈ [0, 1], where 0 corresponds to the initial noise distribution and 1 corresponds to the data distribution.
    """

    omega: float  # constant weight

    name: str = field(init=False, default="BaseWeight")
    # Keep runtime callable typing while excluding this field from OmegaConf structured schema.
    weight_function: Callable[[torch.Tensor], torch.Tensor] = field(
        init=False,
        metadata={"omegaconf_ignore": True},
    )


@dataclass
class ConstantWeightConfig(_BaseWeightConfig):
    name: str = field(init=False, default="ConstantWeight")

    def __post_init__(self):
        def _constant_weight(t: torch.Tensor) -> torch.Tensor:
            return self.omega * torch.ones_like(t)

        self.weight_function = _constant_weight


@dataclass
class LinearIncreasingWeightConfig(_BaseWeightConfig):
    name: str = field(init=False, default="LinearIncreasingWeight")
    slope: float

    def __post_init__(self):
        def _linear_increasing_weight(t: torch.Tensor) -> torch.Tensor:
            return self.omega * self.slope * t

        self.weight_function = _linear_increasing_weight


@dataclass
class LinearDecreasingWeightConfig(_BaseWeightConfig):
    name: str = field(init=False, default="LinearDecreasingWeight")
    slope: float

    def __post_init__(self):
        def _linear_decreasing_weight(t: torch.Tensor) -> torch.Tensor:
            return self.omega * self.slope * (1 - t)

        self.weight_function = _linear_decreasing_weight


@dataclass
class LambdaBumpWeightConfig(_BaseWeightConfig):
    """0.5 centered bump function that peaks at t=0.5 and goes to 0 at t=0 and t=1"""

    name: str = field(init=False, default="LambdaBumpWeight")
    slope: float

    def __post_init__(self):
        def _lambda_bump_weight(t: torch.Tensor) -> torch.Tensor:
            return self.omega * self.slope * (1 - torch.abs(2 * t - 1))

        self.weight_function = _lambda_bump_weight


@dataclass
class VBumpWeightConfig(_BaseWeightConfig):
    """0.5 centered bump function that 0 at t=0.5 and goes to peak at t=0 and t=1"""

    name: str = field(init=False, default="VBumpWeight")
    slope: float

    def __post_init__(self):
        def _v_bump_weight(t: torch.Tensor) -> torch.Tensor:
            return self.omega * self.slope * torch.abs(2 * t - 1)

        self.weight_function = _v_bump_weight


@dataclass
class QuadraticBumpWeightConfig(_BaseWeightConfig):
    """A quadratic bump function that peaks at t=0.5 and goes to 0 at t=0 and t=1"""

    name: str = field(init=False, default="QuadraticBumpWeight")
    slope: float

    def __post_init__(self):
        def _quadratic_bump_weight(t: torch.Tensor) -> torch.Tensor:
            return self.omega * self.slope * t * (1 - t)

        self.weight_function = _quadratic_bump_weight


@dataclass
class ACEBumpWeightConfig(_BaseWeightConfig):
    """A combination of a linear bump (Lambda bump-like) and a quadratic bump that ACE paper proposes. Two hyperparameters control the shape of the bump as

    b(t) = B1 * Q(t) + B2 * L(t)

    where Q(t) = t(1-t) is the quadratic bump and L(t) = min(t, τ(1-t)) is the linear bump. Here, we choose sufficiently large τ >> 1 that L(t) = t can be ensured for all discretized sampling steps.
    B1 and B2 are the two hyperparameters that control the shape of the bump.
    """

    name: str = field(init=False, default="ACEBumpWeight")
    B1: float
    B2: float

    def __post_init__(self):
        def _ace_bump_weight(t: torch.Tensor) -> torch.Tensor:
            Q_t = t * (1 - t)  # Noqa: N806
            L_t = t  # Noqa: N806
            return self.omega + (self.B1 * Q_t + self.B2 * L_t)

        self.weight_function = _ace_bump_weight


if __name__ == "__main__":
    # --- 사용 예시 ---
    config = ConstantWeightConfig(omega=2.5)

    print(config.name)  # 출력: ConstantWeight_2.5
    print(config.weight_function(torch.tensor([0.1, 0.5])))  # 출력: tensor([2.5000, 2.5000])

    # LinearIncreasingWeightConfig 예시
    linear_config = LinearIncreasingWeightConfig(omega=1.0, slope=2.0)
    print(linear_config.name)  # 출력: LinearIncreasingWeight_1.0*2.0*t
    print(linear_config.weight_function(torch.tensor([0.1, 0.5])))  # 출력: tensor([0.2000, 1.0000]) -> 1.0 * 2.0 * t

    # LinearDecreasingWeightConfig 예시
    decreasing_config = LinearDecreasingWeightConfig(omega=1.0, slope=2.0)
    print(decreasing_config.name)  # 출력: LinearDecreasingWeight_1.0*2.0*(1-t)
    print(
        decreasing_config.weight_function(torch.tensor([0.1, 0.5]))
    )  # 출력: tensor([1.8000, 1.0000]) -> 1.0 * 2.0 * (1 - t)

    # LambdaBumpWeightConfig 예시
    bump_config = LambdaBumpWeightConfig(omega=1.0, slope=2.0)
    print(bump_config.name)  # 출력: LambdaBumpWeight_1.0*2.0*(1-abs(2*t-1))
    print(
        bump_config.weight_function(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    )  # 출력: tensor([0.0000, 1.0000, 2.0000, 1.0000, 0.0000]) -> 1.0 * 2.0 * (1 - abs(2*t-1))
