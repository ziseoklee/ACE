from dataclasses import dataclass

from .config_moe_component import MoEComponentConfig
from .config_weight import _BaseWeightConfig


@dataclass
class MoEExponentConfig:
    """Config-driven exponent rule for one MoE component.

    The runtime uses weight_fn(t), then evaluates:
        weight_scale * weight_fn(t) + constant
    """

    weight_fn: _BaseWeightConfig
    weight_scale: float = 1.0
    constant: float = 0.0


@dataclass
class MoEConfig:
    """Structured config for how component paths are composed into one MoE path."""

    omega: float
    components: dict[str, MoEComponentConfig]
    exponents: dict[str, MoEExponentConfig]
    global_scheduler_key: str
