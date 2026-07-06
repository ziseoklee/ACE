import math
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

    def __post_init__(self) -> None:
        if not isinstance(self.weight_fn, _BaseWeightConfig):
            raise TypeError(f"MoE exponent weight_fn must be a weight config, got {type(self.weight_fn).__name__}.")
        if not math.isfinite(float(self.weight_scale)):
            raise ValueError("MoE exponent weight_scale must be finite.")
        if not math.isfinite(float(self.constant)):
            raise ValueError("MoE exponent constant must be finite.")


@dataclass
class MoEConfig:
    """Structured config for how component paths are composed into one MoE path.

    WARNING: diffusion_scale is an empirical multiplier for the global MoE SDE
    diffusion coefficient. A value of 2.0 works well in DiffSBDD and 4-expert
    MoE inference, but not consistently in EDM-only or GeoDiff-only inference.
    The reason for this scale is not yet fully understood.
    """

    omega: float
    components: dict[str, MoEComponentConfig]
    exponents: dict[str, MoEExponentConfig]
    global_scheduler_key: str
    diffusion_scale: float = 2.0

    def __post_init__(self) -> None:
        self.components = dict(self.components)
        self.exponents = dict(self.exponents)

        if not math.isfinite(float(self.omega)):
            raise ValueError("MoE omega must be finite.")
        if not math.isfinite(float(self.diffusion_scale)) or self.diffusion_scale <= 0:
            raise ValueError("MoE diffusion_scale must be positive and finite.")
        if not self.global_scheduler_key:
            raise ValueError("MoE global_scheduler_key must be non-empty.")
        if not self.components:
            raise ValueError("MoE config must define at least one component.")
        if not self.exponents:
            raise ValueError("MoE config must define at least one exponent.")

        component_keys = set(self.components)
        exponent_keys = set(self.exponents)
        if component_keys != exponent_keys:
            missing_exponents = sorted(component_keys - exponent_keys)
            missing_components = sorted(exponent_keys - component_keys)
            raise ValueError(
                "MoE components and exponents must use identical component ids. "
                f"Missing exponents for components: {missing_exponents}; "
                f"exponents without components: {missing_components}."
            )

        component_names = [component.name for component in self.components.values()]
        duplicate_component_names = sorted(
            {component_name for component_name in component_names if component_names.count(component_name) > 1}
        )
        if duplicate_component_names:
            raise ValueError(f"MoE component names must be unique: {duplicate_component_names}.")

        from pipelines.registry import get_supported_scheduler_keys

        supported_scheduler_keys = get_supported_scheduler_keys()
        if self.global_scheduler_key not in supported_scheduler_keys:
            raise ValueError(
                f"Unsupported MoE global_scheduler_key {self.global_scheduler_key!r}. "
                f"Supported scheduler keys: {sorted(supported_scheduler_keys)}."
            )
