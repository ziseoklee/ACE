from dataclasses import dataclass
from typing import Union

from .config_weight import ACEBumpWeightConfig, ConstantWeightConfig, _BaseWeightConfig


@dataclass
class _BaseSamplerConfig:
    name: str
    use_logq: bool
    do_resample: bool
    omega: float  # guidance scale
    weight_config: _BaseWeightConfig
    num_sampling_steps: int = 500
    dlogq_calc_interval: int = (
        10  # How often to calculate dlogq correction term. This is effectively the step size for logq correction.
    )
    dlogq_noise_scale: float = 3.16227766017  # sqrt(10)
    resampling_step_interval: int = 10
    batch_size: int = 5
    device: str = "cuda:0"
    seed: int = 42  # Random seed for reproducibility


@dataclass(init=False)
class NRSamplerConfig(_BaseSamplerConfig):
    name: str = "NRSampler"
    use_logq: bool = False
    do_resample: bool = False

    def __init__(
        self,
        omega: float,
        weight_config: Union[_BaseWeightConfig, None] = None,
        name: str = "NRSampler",
        use_logq: bool = False,
        do_resample: bool = False,
        num_sampling_steps: int = _BaseSamplerConfig.num_sampling_steps,
        dlogq_calc_interval: int = _BaseSamplerConfig.dlogq_calc_interval,
        dlogq_noise_scale: float = _BaseSamplerConfig.dlogq_noise_scale,
        resampling_step_interval: int = _BaseSamplerConfig.resampling_step_interval,
        batch_size: int = _BaseSamplerConfig.batch_size,
        device: str = _BaseSamplerConfig.device,
    ):
        if weight_config is None:
            weight_config = ConstantWeightConfig(omega=omega)

        super().__init__(
            name=name,
            use_logq=use_logq,
            do_resample=do_resample,
            omega=omega,
            weight_config=weight_config,
            num_sampling_steps=num_sampling_steps,
            dlogq_calc_interval=dlogq_calc_interval,
            dlogq_noise_scale=dlogq_noise_scale,
            resampling_step_interval=resampling_step_interval,
            batch_size=batch_size,
            device=device,
        )


@dataclass(init=False)
class FKCSamplerConfig(_BaseSamplerConfig):
    name: str = "FKCSampler"
    do_resample: bool = True
    use_logq: bool = False

    def __init__(
        self,
        omega: float,
        weight_config: Union[_BaseWeightConfig, None] = None,
        name: str = "FKCSampler",
        use_logq: bool = False,
        do_resample: bool = True,
        num_sampling_steps: int = _BaseSamplerConfig.num_sampling_steps,
        dlogq_calc_interval: int = _BaseSamplerConfig.dlogq_calc_interval,
        dlogq_noise_scale: float = _BaseSamplerConfig.dlogq_noise_scale,
        resampling_step_interval: int = _BaseSamplerConfig.resampling_step_interval,
        batch_size: int = _BaseSamplerConfig.batch_size,
        device: str = _BaseSamplerConfig.device,
    ):
        if weight_config is None:
            weight_config = ConstantWeightConfig(omega=omega)

        super().__init__(
            name=name,
            use_logq=use_logq,
            do_resample=do_resample,
            omega=omega,
            weight_config=weight_config,
            num_sampling_steps=num_sampling_steps,
            dlogq_calc_interval=dlogq_calc_interval,
            dlogq_noise_scale=dlogq_noise_scale,
            resampling_step_interval=resampling_step_interval,
            batch_size=batch_size,
            device=device,
        )


@dataclass(init=False)
class ACESamplerConfig(_BaseSamplerConfig):
    name: str = "ACESampler"
    do_resample: bool = True
    use_logq: bool = True

    def __init__(
        self,
        omega: float,
        weight_config: Union[_BaseWeightConfig, None] = None,
        name: str = "ACESampler",
        use_logq: bool = True,
        do_resample: bool = True,
        num_sampling_steps: int = _BaseSamplerConfig.num_sampling_steps,
        dlogq_calc_interval: int = _BaseSamplerConfig.dlogq_calc_interval,
        dlogq_noise_scale: float = _BaseSamplerConfig.dlogq_noise_scale,
        resampling_step_interval: int = _BaseSamplerConfig.resampling_step_interval,
        batch_size: int = _BaseSamplerConfig.batch_size,
        device: str = _BaseSamplerConfig.device,
        **weight_config_kwargs,
    ):
        if weight_config is None:
            weight_config = ACEBumpWeightConfig(omega=omega, **weight_config_kwargs)

        super().__init__(
            name=name,
            use_logq=use_logq,
            do_resample=do_resample,
            omega=omega,
            weight_config=weight_config,
            num_sampling_steps=num_sampling_steps,
            dlogq_calc_interval=dlogq_calc_interval,
            dlogq_noise_scale=dlogq_noise_scale,
            resampling_step_interval=resampling_step_interval,
            batch_size=batch_size,
            device=device,
        )


@dataclass(init=False)
class ACEliteSamplerConfig(_BaseSamplerConfig):
    name: str = "ACEliteSampler"
    do_resample: bool = False
    use_logq: bool = True

    def __init__(
        self,
        omega: float,
        weight_config: Union[_BaseWeightConfig, None] = None,
        name: str = "ACEliteSampler",
        use_logq: bool = True,
        do_resample: bool = False,
        num_sampling_steps: int = _BaseSamplerConfig.num_sampling_steps,
        dlogq_calc_interval: int = _BaseSamplerConfig.dlogq_calc_interval,
        dlogq_noise_scale: float = _BaseSamplerConfig.dlogq_noise_scale,
        resampling_step_interval: int = _BaseSamplerConfig.resampling_step_interval,
        batch_size: int = _BaseSamplerConfig.batch_size,
        device: str = _BaseSamplerConfig.device,
        **weight_config_kwargs,
    ):
        if weight_config is None:
            weight_config = ACEBumpWeightConfig(omega=omega, **weight_config_kwargs)

        super().__init__(
            name=name,
            use_logq=use_logq,
            do_resample=do_resample,
            omega=omega,
            weight_config=weight_config,
            num_sampling_steps=num_sampling_steps,
            dlogq_calc_interval=dlogq_calc_interval,
            dlogq_noise_scale=dlogq_noise_scale,
            resampling_step_interval=resampling_step_interval,
            batch_size=batch_size,
            device=device,
        )
