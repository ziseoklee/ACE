from dataclasses import dataclass


@dataclass
class _BaseSamplerConfig:
    name: str
    use_logq: bool
    do_resample: bool
    dlogq_calc_interval: int = (
        10  # How often to calculate dlogq correction term. This is effectively the step size for logq correction.
    )
    dlogq_noise_scale: float = 3.16227766017  # sqrt(10)
    resampling_step_interval: int = 10
    # General sampling parameters
    num_sampling_steps: int = 500
    ode_start_t: float = 0.98  # At `ode_start_t` < t <= 1.0, use ODE sampling to avoid endpoint instability.
    batch_size: int = 5
    device: str = "cuda:0"
    seed: int = 42  # Random seed for reproducibility


@dataclass
class NRSamplerConfig(_BaseSamplerConfig):
    name: str = "NRSampler"
    use_logq: bool = False
    do_resample: bool = False


@dataclass
class FKCSamplerConfig(_BaseSamplerConfig):
    name: str = "FKCSampler"
    do_resample: bool = True
    use_logq: bool = False


@dataclass
class ACESamplerConfig(_BaseSamplerConfig):
    name: str = "ACESampler"
    do_resample: bool = True
    use_logq: bool = True


@dataclass
class ACEliteSamplerConfig(_BaseSamplerConfig):
    name: str = "ACEliteSampler"
    do_resample: bool = False
    use_logq: bool = True
