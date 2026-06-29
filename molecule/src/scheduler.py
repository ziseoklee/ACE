from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


class SchedulerABC(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def ddpm_alpha2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """DDPM alpha^2(t) = alpha(t)^2."""
        pass

    @abstractmethod
    def ddpm_sigma2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """DDPM sigma^2(t) = 1 - alpha(t)^2."""
        pass

    @abstractmethod
    def drift_coeff(
        self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B data"]
    ) -> Float[torch.Tensor, "B data"]:
        """VP SDE drift on forward process: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW"""
        pass

    @abstractmethod
    def diffusion_coeff(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """VP SDE diffusion coefficient on forward process"""
        pass


class DiffSBDDScheduler(SchedulerABC):
    """
    DiffSBDD polynomial marginal schedule:

        alpha2_raw(t) = (1 - t**power)**2,  t in [0, 1]

        After clipping:
            per-step alpha2 ratio is clipped to [clip_value, 1].

        After precision offset:
            alpha2 = (1 - 2 * eps) * alpha2_clipped + eps
            sigma2 = 1 - alpha2

    Arguments:
        num_discretization_steps: Number of discretization steps that being used for DiffSBDD training. (default: 500)
        power: Power for the polynomial schedule. (default: 2.0)
        clip_value: Minimum value for the per-step alpha2 ratio. (default: 0.001)
        eps: Precision offset for the alpha2 schedule. (default: 5e-4)
    """

    def __init__(
        self,
        num_discretization_steps: int = 500,
        power: float = 2.0,
        clip_value: float = 0.001,
        eps: float = 5e-4,
    ):
        super().__init__()

        assert isinstance(num_discretization_steps, int) and num_discretization_steps > 0
        assert power > 0, "power must be positive"
        assert 0.0 <= clip_value < 1.0, "clip_value must be in [0, 1)"
        assert 0.0 < eps < 0.5, "eps must be in (0, 0.5)"

        self.num_sampling_steps = num_discretization_steps

        timesteps = torch.linspace(0.0, 1.0, num_discretization_steps + 1)
        ddpm_alphas2 = (1.0 - timesteps.pow(power)).pow(2)
        # EDMScheduler-style precision offset
        ddpm_alphas2 = (1.0 - 2.0 * eps) * ddpm_alphas2 + eps
        ddpm_alphas2 = self._clip_noise_schedule(ddpm_alphas2, clip_value=clip_value)

        ddpm_sigmas2 = 1.0 - ddpm_alphas2

        ddpm_log_alphas2 = torch.log(ddpm_alphas2)
        ddpm_log_sigmas2 = torch.log(ddpm_sigmas2)
        log_alphas2_to_sigmas2 = ddpm_log_alphas2 - ddpm_log_sigmas2

        self.timesteps = timesteps
        # negative log SNR curve defined in EDM paper
        self._ddpm_gammas = -log_alphas2_to_sigmas2

    @staticmethod
    def _clip_noise_schedule(
        ddpm_alphas2: torch.Tensor,
        clip_value: float,
    ) -> torch.Tensor:
        """Clips the per-step ratio alpha2_t / alpha2_{t-1}."""
        ddpm_alphas2_concatenated = torch.cat([ddpm_alphas2.new_ones(1), ddpm_alphas2], dim=0)

        ddpm_alphas2_step_ratio = ddpm_alphas2_concatenated[1:] / ddpm_alphas2_concatenated[:-1]

        ddpm_alphas2_step_ratio_clipped = ddpm_alphas2_step_ratio.clamp(
            min=clip_value,
            max=1.0,
        )

        ddpm_alphas2_clipped = torch.cumprod(ddpm_alphas2_step_ratio_clipped, dim=0)
        return ddpm_alphas2_clipped

    def _index(self, t: Float[torch.Tensor, "B 1"]) -> Int[torch.Tensor, "B 1"]:
        if torch.any((t < 0.0) | (t > 1.0)):
            raise ValueError("t must be in [0, 1]")

        # DiffSBDD-style nearest table lookup.
        index = torch.round(t * self.num_sampling_steps).long()
        index = index.clamp(0, self.num_sampling_steps)
        return index

    def ddpm_alpha2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        index = self._index(t)
        values = self._ddpm_gammas.to(device=t.device)
        return torch.sigmoid(-values[index])

    def ddpm_sigma2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        index = self._index(t)
        values = self._ddpm_gammas.to(device=t.device)
        return torch.sigmoid(values[index])

    def beta(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """
        Piecewise-constant VP-SDE beta rate approximation:

            beta_i ~= - d/dt log alpha2(t)
                ~= T * (log alpha2_{i-1} - log alpha2_i)

        For index 0, use the first interval [0, 1/T].
        """
        index = self._index(t)

        ddpm_gammas = self._ddpm_gammas.to(device=t.device)

        # Use interval i = max(index, 1)
        i = index.clamp(1, self.num_sampling_steps)

        log_prev = torch.log(torch.sigmoid(-ddpm_gammas[i - 1]))
        log_curr = torch.log(torch.sigmoid(-ddpm_gammas[i]))

        beta_rate = self.num_sampling_steps * (log_prev - log_curr)
        return beta_rate.clamp_min(0.0)

    def drift_coeff(
        self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B data"]
    ) -> Float[torch.Tensor, "B data"]:
        """VP-SDE drift: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW."""
        return -0.5 * self.beta(t) * x

    def diffusion_coeff(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """Piecewise-constant VP-like diffusion coefficient induced by the regularized marginal schedule."""
        return torch.sqrt(self.beta(t))


class EDMScheduler(SchedulerABC):
    """
    EDMScheduler polynomial marginal schedule:

        alpha2_raw(t) = (1 - t**power)**2,  t in [0, 1]

        After clipping:
            per-step alpha2 ratio is clipped to [clip_value, 1].

        After precision offset:
            alpha2 = (1 - 2 * eps) * alpha2_clipped + eps
            sigma2 = 1 - alpha2

    Arguments:
        num_discretization_steps: Number of discretization steps that being used for EDM training. (default: 1000)
        power: Power for the polynomial schedule. (default: 2.0)
        clip_value: Minimum value for the per-step alpha2 ratio. (default: 0.001)
        eps: Precision offset for the alpha2 schedule. (default: 1e-5)
    """

    def __init__(
        self,
        num_discretization_steps: int = 1000,
        power: float = 2.0,
        clip_value: float = 0.001,
        eps: float = 1e-5,
    ):
        super().__init__()

        assert isinstance(num_discretization_steps, int) and num_discretization_steps > 0
        assert power > 0, "power must be positive"
        assert 0.0 <= clip_value < 1.0, "clip_value must be in [0, 1)"
        assert 0.0 < eps < 0.5, "eps must be in (0, 0.5)"

        self.num_sampling_steps = num_discretization_steps

        timesteps = torch.linspace(0.0, 1.0, num_discretization_steps + 1)
        ddpm_alphas2 = (1.0 - timesteps.pow(power)).pow(2)
        # EDMScheduler-style precision offset
        ddpm_alphas2 = (1.0 - 2.0 * eps) * ddpm_alphas2 + eps
        ddpm_alphas2 = self._clip_noise_schedule(ddpm_alphas2, clip_value=clip_value)

        ddpm_sigmas2 = 1.0 - ddpm_alphas2

        ddpm_log_alphas2 = torch.log(ddpm_alphas2)
        ddpm_log_sigmas2 = torch.log(ddpm_sigmas2)
        log_alphas2_to_sigmas2 = ddpm_log_alphas2 - ddpm_log_sigmas2

        self.timesteps = timesteps
        # negative log SNR curve defined in EDM paper
        self._ddpm_gammas = -log_alphas2_to_sigmas2

    @staticmethod
    def _clip_noise_schedule(
        ddpm_alphas2: torch.Tensor,
        clip_value: float,
    ) -> torch.Tensor:
        """Clips the per-step ratio alpha2_t / alpha2_{t-1}."""
        ddpm_alphas2_concatenated = torch.cat([ddpm_alphas2.new_ones(1), ddpm_alphas2], dim=0)

        ddpm_alphas2_step_ratio = ddpm_alphas2_concatenated[1:] / ddpm_alphas2_concatenated[:-1]

        ddpm_alphas2_step_ratio_clipped = ddpm_alphas2_step_ratio.clamp(
            min=clip_value,
            max=1.0,
        )

        ddpm_alphas2_clipped = torch.cumprod(ddpm_alphas2_step_ratio_clipped, dim=0)
        return ddpm_alphas2_clipped

    def _index(self, t: Float[torch.Tensor, "B 1"]) -> Int[torch.Tensor, "B 1"]:
        if torch.any((t < 0.0) | (t > 1.0)):
            raise ValueError("t must be in [0, 1]")

        # DiffSBDD-style nearest table lookup.
        index = torch.round(t * self.num_sampling_steps).long()
        index = index.clamp(0, self.num_sampling_steps)
        return index

    def ddpm_alpha2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        index = self._index(t)
        values = self._ddpm_gammas.to(device=t.device)
        return torch.sigmoid(-values[index])

    def ddpm_sigma2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        index = self._index(t)
        values = self._ddpm_gammas.to(device=t.device)
        return torch.sigmoid(values[index])

    def beta(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """
        Piecewise-constant VP-SDE beta rate approximation:

            beta_i ~= - d/dt log alpha2(t)
                ~= T * (log alpha2_{i-1} - log alpha2_i)

        For index 0, use the first interval [0, 1/T].
        """
        index = self._index(t)

        ddpm_gammas = self._ddpm_gammas.to(device=t.device)

        # Use interval i = max(index, 1)
        i = index.clamp(1, self.num_sampling_steps)

        log_prev = torch.log(torch.sigmoid(-ddpm_gammas[i - 1]))
        log_curr = torch.log(torch.sigmoid(-ddpm_gammas[i]))

        beta_rate = self.num_sampling_steps * (log_prev - log_curr)
        return beta_rate.clamp_min(0.0)

    def drift_coeff(
        self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B data"]
    ) -> Float[torch.Tensor, "B data"]:
        """VP-SDE drift: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW."""
        return -0.5 * self.beta(t) * x

    def diffusion_coeff(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """Piecewise-constant VP-like diffusion coefficient induced by the regularized marginal schedule."""
        return torch.sqrt(self.beta(t))


class GeoDiffScheduler(SchedulerABC):
    """
    betas = np.linspace(-6, 6, num_diffusion_timesteps)
    betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

    Arguments:
        num_discretization_steps: Number of discretization steps that being used for GeoDiff training. (default: 5000)
        beta_start: Lower bound of the discrete DDPM sigmoid beta schedule. (default: 1e-7)
        beta_end: Upper bound of the discrete DDPM sigmoid beta schedule. (default: 2e-3)
        eps: Precision offset for the beta schedule
    """

    def __init__(
        self,
        num_discretization_steps: int = 5000,
        beta_start: float = 1e-7,
        beta_end: float = 2e-3,
        eps: float = 1e-12,
    ):
        super().__init__()
        # GeoDiff's beta_start/beta_end are per-step DDPM variances. Convert them to
        # normalized-time VP-SDE rates by dividing by dt = 1 / num_discretization_steps.
        self.beta_start = float(beta_start) * num_discretization_steps
        self.beta_end = float(beta_end) * num_discretization_steps

        self.beta_delta = self.beta_end - self.beta_start
        self.eps = float(eps)

    def beta(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        # beta(t) = (beta_end - beta_start) * sigmoid(t) + beta_start
        return self.beta_delta * torch.sigmoid((t - 0.5) * 12.0) + self.beta_start

    def log_mean_coeff(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        # log alpha(t) = -1/2 [ beta_delta * (softplus(t) - log 2) + beta_start * t ]
        return -0.5 * (self.beta_delta / 12.0 * F.softplus((t - 0.5) * 12.0) + self.beta_start * t)

    def ddpm_alpha2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        return torch.exp(2.0 * self.log_mean_coeff(t))

    def ddpm_sigma2(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        return 1.0 - self.ddpm_alpha2(t)

    def drift_coeff(
        self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B data"]
    ) -> Float[torch.Tensor, "B data"]:
        """VP SDE drift on forward process: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW"""
        return -0.5 * self.beta(t) * x

    def diffusion_coeff(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """VP SDE diffusion coefficient on forward process"""
        return torch.sqrt(self.beta(t).clamp(min=self.eps))
