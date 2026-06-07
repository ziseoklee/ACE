from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class SchedulerABC(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """alpha(t) in VP SDE"""
        pass

    @abstractmethod
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """VP SDE drift on forward process: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW"""
        pass

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """VP SDE diffusion coefficient on forward process"""
        pass

    @abstractmethod
    def h(self, t: torch.Tensor) -> torch.Tensor:
        """marginal noise variance for VP SDE: 1 - alpha(t)^2"""
        pass


class DiffSBDDScheduler(SchedulerABC):
    """
    VP-SDE schedule consistent with:
        alpha^2_raw(t) = (1 - t**power)**2,  t in [0, 1]
    Then:
        log_mean_coeff(t) = 0.5 * log(alpha^2_clipped(t))
        beta(t) = - d/dt log(alpha^2_clipped(t))
                = 2*power*t**(power-1) / (1 - t**power)  (where not clipped)
    Clipping saturates both log-mean and beta (beta=0 once clipped).
    """

    def __init__(self, power: float = 2.0, clip_value: float = 1e-3, eps: float = 1e-12):
        super().__init__()
        assert power > 0, "power must be positive"
        assert 0.0 < clip_value <= 1.0, "clip_value must be in (0, 1]"
        self.power = float(power)
        self.clip_value = float(clip_value)
        self.eps = float(eps)

    def _t01(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(0.0, 1.0)

    def _alpha2_raw(self, t: torch.Tensor) -> torch.Tensor:
        t = self._t01(t)
        return (1.0 - t.pow(self.power)).pow(2)

    def _alpha2_clipped(self, t: torch.Tensor) -> torch.Tensor:
        return self._alpha2_raw(t).clamp(min=self.clip_value, max=1.0) * (1 - 2e-4) + 1e-4

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # log(alpha(t)) = 0.5 * log(alpha^2_clipped(t))
        a2 = self._alpha2_clipped(t)
        return 0.5 * torch.log(a2.clamp(min=self.eps))

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # alpha(t)
        return torch.exp(self.log_mean_coeff(t))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        t = self._t01(t)
        tp = t.pow(self.power)
        denom = (1.0 - tp).clamp(min=self.eps)

        # raw beta from alpha^2_raw
        with torch.no_grad():
            # handle t^(p-1) safely, especially when p<1 near t=0
            safe_t_pow = torch.where(t > 0, t.pow(self.power - 1.0), torch.zeros_like(t))
        beta_raw = (2.0 * self.power) * safe_t_pow / denom

        # zero beta where clipping is active (alpha^2 is saturated)
        a2_raw = self._alpha2_raw(t)
        not_clipped = a2_raw > self.clip_value
        return torch.where(not_clipped, beta_raw, torch.zeros_like(beta_raw))

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """VP SDE drift on forward process: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW"""
        return -0.5 * self.beta(t) * x

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """VP SDE diffusion coefficient on forward process"""
        return torch.sqrt(self.beta(t).clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        """marginal noise variance for VP SDE: 1 - alpha(t)^2"""
        return 1.0 - self._alpha2_clipped(t)


class EDMScheduler(SchedulerABC):
    """
    VP-SDE schedule consistent with:
        alpha^2_raw(t) = (1 - t**power)**2,  t in [0, 1]
    Then:
        log_mean_coeff(t) = 0.5 * log(alpha^2_clipped(t))
        beta(t) = - d/dt log(alpha^2_clipped(t))
                = 2*power*t**(power-1) / (1 - t**power)  (where not clipped)
    Clipping saturates both log-mean and beta (beta=0 once clipped).
    """

    def __init__(self, power: float = 2.0, clip_value: float = 1e-3, eps: float = 1e-12):
        super().__init__()
        assert power > 0, "power must be positive"
        assert 0.0 < clip_value <= 1.0, "clip_value must be in (0, 1]"
        self.power = float(power)
        self.clip_value = float(clip_value)
        self.eps = float(eps)

    def _t01(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(0.0, 1.0)

    def _alpha2_raw(self, t: torch.Tensor) -> torch.Tensor:
        t = self._t01(t)
        return (1.0 - t.pow(self.power)).pow(2)

    def _alpha2_clipped(self, t: torch.Tensor) -> torch.Tensor:
        return self._alpha2_raw(t).clamp(min=self.clip_value, max=1.0) * (1 - 2e-4) + 1e-4

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # log(alpha(t)) = 0.5 * log(alpha^2_clipped(t))
        a2 = self._alpha2_clipped(t)
        return 0.5 * torch.log(a2.clamp(min=self.eps))

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # alpha(t)
        return torch.exp(self.log_mean_coeff(t))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        t = self._t01(t)
        tp = t.pow(self.power)
        denom = (1.0 - tp).clamp(min=self.eps)

        # raw beta from alpha^2_raw
        with torch.no_grad():
            # handle t^(p-1) safely, especially when p<1 near t=0
            safe_t_pow = torch.where(t > 0, t.pow(self.power - 1.0), torch.zeros_like(t))
        beta_raw = (2.0 * self.power) * safe_t_pow / denom

        # zero beta where clipping is active (alpha^2 is saturated)
        a2_raw = self._alpha2_raw(t)
        not_clipped = a2_raw > self.clip_value
        return torch.where(not_clipped, beta_raw, torch.zeros_like(beta_raw))

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """VP SDE drift on forward process: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW"""
        return -0.5 * self.beta(t) * x

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """VP SDE diffusion coefficient on forward process"""
        return torch.sqrt(self.beta(t).clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        """marginal noise variance for VP SDE: 1 - alpha(t)^2"""
        return 1.0 - self._alpha2_clipped(t)


class GeoDiffScheduler(SchedulerABC):
    """
    betas = np.linspace(-6, 6, num_diffusion_timesteps)
    betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    """

    def __init__(self, beta_start: float = 1e-12, beta_end: float = 10.0, eps: float = 1e-12):
        super().__init__()
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_delta = self.beta_end - self.beta_start
        self.eps = float(eps)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # beta(t) = (beta_end - beta_start) * sigmoid(t) + beta_start
        return self.beta_delta * torch.sigmoid((t - 0.5) * 12.0) + self.beta_start

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # log alpha(t) = -1/2 [ beta_delta * (softplus(t) - log 2) + beta_start * t ]
        return -0.5 * (self.beta_delta / 12.0 * F.softplus((t - 0.5) * 12.0) + self.beta_start * t)

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_mean_coeff(t))

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """VP SDE drift on forward process: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW"""
        return -0.5 * self.beta(t) * x

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """VP SDE diffusion coefficient on forward process"""
        return torch.sqrt(self.beta(t).clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        """marginal variance for VP SDE: 1 - alpha(t)^2"""
        return 1.0 - torch.exp(2.0 * self.log_mean_coeff(t))
