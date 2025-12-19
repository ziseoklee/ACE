import torch
import functools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import qmc

### scheduler ###

class DiffSBDDSchedule:
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
        return self._alpha2_raw(t).clamp(min=self.clip_value, max=1.0)

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
            safe_t_pow = torch.where(
                t > 0, t.pow(self.power - 1.0), torch.zeros_like(t)
            )
        beta_raw = (2.0 * self.power) * safe_t_pow / denom

        # zero beta where clipping is active (alpha^2 is saturated)
        a2_raw = self._alpha2_raw(t)
        not_clipped = (a2_raw > self.clip_value)
        return torch.where(not_clipped, beta_raw, torch.zeros_like(beta_raw))

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # VP SDE drift: dx = -0.5 * beta(t) * x dt + g(t) dW
        return -0.5 * self.beta(t) * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        # diffusion coefficient
        return torch.sqrt(self.beta(t).clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        # marginal noise variance for VP SDE: 1 - alpha(t)^2
        return 1.0 - self._alpha2_clipped(t)

class EdmSchedule:
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
        return self._alpha2_raw(t).clamp(min=self.clip_value, max=1.0)

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
            safe_t_pow = torch.where(
                t > 0, t.pow(self.power - 1.0), torch.zeros_like(t)
            )
        beta_raw = (2.0 * self.power) * safe_t_pow / denom

        # zero beta where clipping is active (alpha^2 is saturated)
        a2_raw = self._alpha2_raw(t)
        not_clipped = (a2_raw > self.clip_value)
        return torch.where(not_clipped, beta_raw, torch.zeros_like(beta_raw))

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # VP SDE drift: dx = -0.5 * beta(t) * x dt + g(t) dW
        return -0.5 * self.beta(t) * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        # diffusion coefficient
        return torch.sqrt(self.beta(t).clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        # marginal noise variance for VP SDE: 1 - alpha(t)^2
        return 1.0 - self._alpha2_clipped(t)


class GeodiffSchedule:
    """
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    """
    def __init__(self, beta_start=0.001, beta_end=20.0, eps=1e-12):
        self.beta_start = float(beta_start)
        self.beta_end   = float(beta_end)
        self.beta_delta = self.beta_end - self.beta_start
        self.eps = eps

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # beta(t) = (beta_end - beta_start) * sigmoid(t) + beta_start
        return self.beta_delta * torch.sigmoid((t - 0.5) * 12.0) + self.beta_start

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # log alpha(t) = -1/2 [ beta_delta * (softplus(t) - log 2) + beta_start * t ]
        return -0.5 * ( self.beta_delta / 12.0 * F.softplus((t - 0.5) * 12.0)
                        + self.beta_start * t )

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_mean_coeff(t))

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # VP drift
        return -0.5 * self.beta(t) * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        # diffusion coeff
        return torch.sqrt(self.beta(t).clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        # marginal variance for VP SDE: 1 - alpha(t)^2
        return 1.0 - torch.exp(2.0 * self.log_mean_coeff(t))

class TargetDiffPosSchedule:
    """
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    """
    def __init__(self, beta_start=0.001, beta_end=20.0, eps=1e-12):
        self.beta_start = float(beta_start)
        self.beta_end   = float(beta_end)
        self.beta_delta = self.beta_end - self.beta_start
        self.eps = eps

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # beta(t) = (beta_end - beta_start) * sigmoid(t) + beta_start
        return self.beta_delta * torch.sigmoid((t - 0.5) * 12.0) + self.beta_start

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # log alpha(t) = -1/2 [ beta_delta * (softplus(t) - log 2) + beta_start * t ]
        return -0.5 * ( self.beta_delta / 12.0 * F.softplus((t - 0.5) * 12.0)
                        + self.beta_start * t )

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_mean_coeff(t))

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # VP drift
        return -0.5 * self.beta(t) * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        # diffusion coeff
        return torch.sqrt(self.beta(t).clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        # marginal variance for VP SDE: 1 - alpha(t)^2
        return 1.0 - torch.exp(2.0 * self.log_mean_coeff(t))

class TargetDiffAtomSchedule:
    """
    Cosine scheduler with T=1000 timesteps
    alpha_t = exp(-PI/(2T) * tan(PI/2 * t))
    sigma_t = sqrt(1 - exp(-PI/T * tan(PI/2 * t)))
    """
    def __init__(self, T=1000, eps=1e-12):
        self.T = T
        self.eps = eps
        self.pi = torch.tensor(torch.pi)

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # log alpha(t) = -PI/(2T) * tan(PI/2 * t)
        return -self.pi / (2 * self.T) * torch.tan(self.pi / 2 * t)

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # alpha(t) = exp(-PI/(2T) * tan(PI/2 * t))
        return torch.exp(self.log_mean_coeff(t))

    def variance_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # sigma(t) = sqrt(1 - exp(-PI/T * tan(PI/2 * t)))
        # This is equivalent to sqrt(1 - alpha(t)^2)
        alpha_squared = torch.exp(2 * self.log_mean_coeff(t))
        return torch.sqrt(1.0 - alpha_squared + self.eps)

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # VP drift: v(t,x) = -0.5 * beta(t) * x
        # For cosine schedule, we need to derive beta(t) from the alpha schedule
        # beta(t) = -2 * d/dt log(alpha(t))
        # d/dt log(alpha(t)) = d/dt [-PI/(2T) * tan(PI/2 * t)]
        #                   = -PI/(2T) * PI/2 * sec^2(PI/2 * t)
        #                   = -PI^2/(4T) * sec^2(PI/2 * t)
        sec_squared = 1.0 / (torch.cos(self.pi / 2 * t) ** 2 + self.eps)
        beta_t = self.pi ** 2 / (2 * self.T) * sec_squared
        return -0.5 * beta_t * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        # diffusion coeff: g(t) = sqrt(beta(t))
        sec_squared = 1.0 / (torch.cos(self.pi / 2 * t) ** 2 + self.eps)
        beta_t = self.pi ** 2 / (2 * self.T) * sec_squared
        return torch.sqrt(beta_t.clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        # marginal variance for VP SDE: 1 - alpha(t)^2
        return self.variance_coeff(t) ** 2


class CosineSchedule:
    """
    Cosine scheduler with T=1000 timesteps
    alpha_t = exp(-PI/(2T) * tan(PI/2 * t))
    sigma_t = sqrt(1 - exp(-PI/T * tan(PI/2 * t)))
    """
    def __init__(self, T=1000, eps=1e-12):
        self.T = T
        self.eps = eps
        self.pi = torch.tensor(torch.pi)

    def log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # log alpha(t) = -PI/(2T) * tan(PI/2 * t)
        return -self.pi / (2 * self.T) * torch.tan(self.pi / 2 * t)

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # alpha(t) = exp(-PI/(2T) * tan(PI/2 * t))
        return torch.exp(self.log_mean_coeff(t))

    def variance_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # sigma(t) = sqrt(1 - exp(-PI/T * tan(PI/2 * t)))
        # This is equivalent to sqrt(1 - alpha(t)^2)
        alpha_squared = torch.exp(2 * self.log_mean_coeff(t))
        return torch.sqrt(1.0 - alpha_squared + self.eps)

    def v(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # VP drift: v(t,x) = -0.5 * beta(t) * x
        # For cosine schedule, we need to derive beta(t) from the alpha schedule
        # beta(t) = -2 * d/dt log(alpha(t))
        # d/dt log(alpha(t)) = d/dt [-PI/(2T) * tan(PI/2 * t)]
        #                   = -PI/(2T) * PI/2 * sec^2(PI/2 * t)
        #                   = -PI^2/(4T) * sec^2(PI/2 * t)
        sec_squared = 1.0 / (torch.cos(self.pi / 2 * t) ** 2 + self.eps)
        beta_t = self.pi ** 2 / (2 * self.T) * sec_squared
        return -0.5 * beta_t * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        # diffusion coeff: g(t) = sqrt(beta(t))
        sec_squared = 1.0 / (torch.cos(self.pi / 2 * t) ** 2 + self.eps)
        beta_t = self.pi ** 2 / (2 * self.T) * sec_squared
        return torch.sqrt(beta_t.clamp(min=self.eps))

    def h(self, t: torch.Tensor) -> torch.Tensor:
        # marginal variance for VP SDE: 1 - alpha(t)^2
        return self.variance_coeff(t) ** 2


class VPSchedule:
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        # beta = -2 * d/dt log_alpha
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def log_mean_coeff(self, t):  # log_alpha
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def v(self, t, x):
        beta = self.beta(t)
        return -0.5 * beta * x

    def g(self, t):
        return self.beta(t) ** 0.5
    
    def mean_coeff(self, t):
        return torch.exp(self.log_mean_coeff(t))

    def h(self, t):  # variance
        return 1.0 - torch.exp(2.0 * self.log_mean_coeff(t))

class BaseVPSchedule:
    """Base class for Variance Preserving (VP) schedulers with different beta scheduling schemes."""
    
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t):
        """Compute beta(t) for the given time t."""
        raise NotImplementedError("Subclasses must implement beta(t)")
    
    def log_mean_coeff(self, t):
        """Compute log(alpha(t)) = -0.5 * integral_0^t beta(s) ds."""
        raise NotImplementedError("Subclasses must implement log_mean_coeff(t)")
    
    def v(self, t, x):
        """Compute drift coefficient v(t, x) = -0.5 * beta(t) * x."""
        beta = self.beta(t)
        return -0.5 * beta * x
    
    def g(self, t):
        """Compute diffusion coefficient g(t) = sqrt(beta(t))."""
        return self.beta(t) ** 0.5
    
    def mean_coeff(self, t):
        """Compute mean coefficient alpha(t) = exp(log_mean_coeff(t))."""
        return torch.exp(self.log_mean_coeff(t))
    
    def h(self, t):
        """Compute variance h(t) = 1 - alpha(t)^2."""
        return 1.0 - torch.exp(2.0 * self.log_mean_coeff(t))
    

class DerivedVPSchedule(BaseVPSchedule):
    def __init__(self, mean_coeff_square_t, d_mean_coeff_square_t):
        """
        Initialize with h(t) function (variance function).
        h(t) = 1 - alpha(t)^2, so alpha(t) = sqrt(1 - h(t))
        """
        super().__init__()
        self.mean_coeff_square_t = lambda t: mean_coeff_square_t(t)
        self.d_mean_coeff_square_t = lambda t: d_mean_coeff_square_t(t)
    
    def h(self, t):
        """Compute variance h(t) using the provided function."""
        return 1 - self.mean_coeff_square_t(t)
    
    def mean_coeff(self, t):
        """Compute alpha(t) = sqrt(1 - h(t))."""
        return torch.exp(0.5 * torch.log(self.mean_coeff_square_t(t)))
    
    def log_mean_coeff(self, t):
        """Compute log(alpha(t))."""
        return 0.5 * torch.log(self.mean_coeff_square_t(t))
    
    def beta(self, t):
        """Compute beta(t) = -2 * dlog(alpha_t) / dt using automatic differentiation."""
        return ((self.d_mean_coeff_square_t(t) / self.mean_coeff_square_t(t).clamp(min=1e-4)))

    
    def v(self, t, x):
        """Compute drift coefficient v(t, x) = -0.5 * beta(t) * x."""
        beta = self.beta(t)
        return -0.5 * beta * x
    
    def g(self, t):
        """Compute diffusion coefficient g(t) = sqrt(beta(t))."""
        return torch.sqrt(self.beta(t))
            


class PolynomialVPSchedule(BaseVPSchedule):
    """Polynomial VP scheduler: beta(t) = beta_min + (beta_max - beta_min) * t^n."""
    
    def __init__(self, beta_min=0.1, beta_max=20, n=2):
        super().__init__(beta_min, beta_max)
        self.n = n

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t**self.n
    
    def log_mean_coeff(self, t):
        return -0.5 * (self.beta_min * t + (self.beta_max - self.beta_min) * t**(self.n + 1) / (self.n + 1))

class LinearVPSchedule(BaseVPSchedule):
    """Linear VP scheduler: beta(t) = beta_min + (beta_max - beta_min) * t."""
    
    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t
    
    def log_mean_coeff(self, t):
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min


class CosineVPSchedule(BaseVPSchedule):
    """Cosine VP scheduler: beta(t) = beta_min + 0.5 * (beta_max - beta_min) * (1 - cos(pi * t))."""
    
    def beta(self, t):
        return self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 - torch.cos(math.pi * t))
    
    def log_mean_coeff(self, t):
        # Integral of beta(s) from 0 to t
        # beta(s) = beta_min + 0.5 * (beta_max - beta_min) * (1 - cos(pi * s))
        # integral = beta_min * t + 0.5 * (beta_max - beta_min) * (t - sin(pi * t) / pi)
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t - torch.sin(math.pi * t) / math.pi)
        return -0.5 * integral


class ExponentialVPSchedule(BaseVPSchedule):
    """Exponential VP scheduler: beta(t) = beta_min * (beta_max / beta_min)^t."""
    
    def beta(self, t):
        return self.beta_min * (self.beta_max / self.beta_min) ** t
    
    def log_mean_coeff(self, t):
        # Integral of beta(s) from 0 to t
        # beta(s) = beta_min * (beta_max / beta_min)^s
        # integral = beta_min * ((beta_max / beta_min)^t - 1) / log(beta_max / beta_min)
        if abs(self.beta_max - self.beta_min) < 1e-8:
            integral = self.beta_min * t
        else:
            log_ratio = math.log(self.beta_max / self.beta_min)
            integral = self.beta_min * ((self.beta_max / self.beta_min) ** t - 1) / log_ratio
        return -0.5 * integral


class QuadraticVPSchedule(BaseVPSchedule):
    """Quadratic VP scheduler: beta(t) = beta_min + (beta_max - beta_min) * t^2."""
    
    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t**2
    
    def log_mean_coeff(self, t):
        return -0.5 * (self.beta_min * t + (self.beta_max - self.beta_min) * t**3 / 3)


class SigmoidVPSchedule(BaseVPSchedule):
    """Sigmoid VP scheduler: beta(t) = beta_min + (beta_max - beta_min) * sigmoid(10 * (t - 0.5))."""
    
    def beta(self, t):
        sigmoid_t = torch.sigmoid(10 * (t - 0.5))
        return self.beta_min + (self.beta_max - self.beta_min) * sigmoid_t
    
    def log_mean_coeff(self, t):
        # For sigmoid, we need to compute the integral numerically or use approximation
        # Here we use a numerical approximation for the integral
        # This is a simplified version - in practice, you might want to use more sophisticated integration
        sigmoid_t = torch.sigmoid(10 * (t - 0.5))
        # Approximate integral using trapezoidal rule approximation
        integral = self.beta_min * t + (self.beta_max - self.beta_min) * sigmoid_t * t * 0.5
        return -0.5 * integral


class CyclicalVPSchedule(BaseVPSchedule):
    """Cyclical VP scheduler: beta(t) = beta_min + 0.5 * (beta_max - beta_min) * (1 + cos(2π * n * t))."""
    
    def __init__(self, beta_min=0.1, beta_max=20, cycles=2):
        super().__init__(beta_min, beta_max)
        self.cycles = cycles
    
    def beta(self, t):
        return self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 + torch.cos(2 * math.pi * self.cycles * t))
    
    def log_mean_coeff(self, t):
        # Integral of beta(s) from 0 to t
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t + torch.sin(2 * math.pi * self.cycles * t) / (2 * math.pi * self.cycles))
        return -0.5 * integral

### Probability Path ###
# The following class implements the probability path attained from the Fokker-Planck equation scheduling given data distribution and prior (Gaussian) distribution.

class ReversedProbabilityPath():
    def __init__(self, scheduler, score_model, gamma=1.0):
        self.scheduler = scheduler
        self.score_model = score_model
        self.gamma = gamma
        
    def drift_coeff(self, t, x):
        v = self.scheduler.v(t, x)
        g = self.scheduler.g(t)
        return -v + 0.5 * g ** 2 * (1 + self.gamma ** 2) * self.score_model(t, x)

    def diffusion_coeff(self, t):
        return self.gamma * self.scheduler.g(t)
        # return self.gamma * self.scheduler.g(t)

    def v(self, t, x):  
        v = self.scheduler.v(t, x)
        return -v

    def score(self, t, x):
        return self.score_model(t, x)

    def g(self, t):
        return self.scheduler.g(t)    

        

class ProbabilityPath():
    def __init__(self, scheduler, score_model, reverse=True, gamma=1.0):
        self.scheduler = scheduler
        self.score_model = score_model
        self.reverse = reverse
        self.gamma = gamma
        
    def drift_coeff(self, t, x):
        if not self.reverse:
            v = self.scheduler.v(t, x)
            return v
        else:
            v = self.scheduler.v(1 - t, x)
            g = self.scheduler.g(1 - t)
            return -v + 0.5 * g ** 2 * (1 + self.gamma ** 2) * self.score_model(1 - t, x)

    def diffusion_coeff(self, t):
        if not self.reverse:
            return self.gamma * self.scheduler.g(t)
        else:
            return self.gamma * self.scheduler.g(1 - t)
        # return self.gamma * self.scheduler.g(t)

    def v(self, t, x):  
        if not self.reverse:
            v = self.scheduler.v(t, x)
            return v
        else:
            v = self.scheduler.v(1 - t, x)
            return -v

    def score(self, t, x):
        if not self.reverse:
            return self.score_model(t, x)
        else:
            return self.score_model(1 - t, x)

    def g(self, t):
        if not self.reverse:
            return self.scheduler.g(t)
        else:
            return self.scheduler.g(1 - t)    

    def reversed_(self, value):
        self.reverse = value

class ConcatenatedProbabilityPath():
    def __init__(self, paths, mask_list):
        self.paths = paths
        self.mask_list = mask_list
        self.dim = mask_list[0].shape[0]
        self.reverse = self.check_reverse()
        self.scheduler = self.paths[0].scheduler

    def check_reverse(self):
        for path in self.paths:
            if path.reverse != self.paths[0].reverse:
                raise ValueError("All paths must have the same reverse value")
        return self.paths[0].reverse

    def drift_coeff(self, t, x):
        out = torch.zeros(x.shape[0], self.dim, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].drift_coeff(t, x[:, self.mask_list[i]])
        return out
    
    def diffusion_coeff(self, t):
        return self.paths[0].diffusion_coeff(t)
    
    def v(self, t, x):
        out = torch.zeros(x.shape[0], self.dim, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].v(t, x[:, self.mask_list[i]])

        return out
    
    def g(self, t):
        return self.paths[0].g(t)

    def score(self, t, x):
        with torch.autograd.set_detect_anomaly(True):
            out = torch.zeros(x.shape[0], self.dim, device=x.device)

            for i in range(len(self.paths)):
                out[:, self.mask_list[i]] = self.paths[i].score(t, x[:, self.mask_list[i]])
                # grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
                # print(f'grad: {grad.shape}')
        return out

    def reversed_(self, value):
        for path in self.paths:
            path.reversed_(value)





