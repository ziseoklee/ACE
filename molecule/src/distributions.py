import functools
from typing import Protocol

import torch
import torch.nn as nn
from jaxtyping import Float


class Scheduler(Protocol):
    def ddpm_alpha2(self, t: Float[torch.Tensor, " B"]) -> Float[torch.Tensor, " B"]: ...

    def ddpm_sigma2(self, t: Float[torch.Tensor, " B"]) -> Float[torch.Tensor, " B"]: ...


class MultivariateGaussian(nn.Module):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, normalize=True, device="cpu"):
        super().__init__()
        self.device = device
        # breakpoint()
        self.mean = mean.clone().detach().to(device=device, dtype=torch.float32)
        self.cov = cov.clone().detach().to(device=device, dtype=torch.float32)
        self.normalize_mode = normalize

        self.N = self.mean.shape[0]
        self.cov_inv = torch.linalg.inv(self.cov)
        self.log_det_cov = torch.logdet(self.cov)
        self.log_coeff = -0.5 * (self.N * torch.log(torch.tensor(2 * torch.pi).to(device)) + self.log_det_cov)

    def __add__(self, other):
        if isinstance(other, MultivariateGaussian):
            return MultivariateGaussian(
                self.mean + other.mean,
                self.cov + other.cov,
                normalize=self.normalize_mode,
                device=self.device,
            )
        elif isinstance(other, torch.Tensor):
            return MultivariateGaussian(
                self.mean + other,
                self.cov,
                normalize=self.normalize_mode,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

    def __rmul__(self, scalar):
        return MultivariateGaussian(
            self.mean * scalar,
            self.cov * scalar**2,
            normalize=self.normalize_mode,
            device=self.device,
        )

    def forward(self, x):
        diff = x - self.mean
        quad_form = torch.einsum("...i,ij,...j->...", diff, self.cov_inv, diff)
        if self.normalize_mode:
            # Only quadratic part, no normalization constant
            return -0.5 * quad_form
        else:
            # Full log-probability
            return self.log_coeff - 0.5 * quad_form

    def log_prob(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        return self.log_coeff - 0.5 * torch.einsum("...i,ij,...j->...", x - self.mean, self.cov_inv, x - self.mean)

    def convolve(self, alpha, sigma, prior):
        # assert that alpha, sigma is consistent over a batch (= t is same for all items)
        mean_tensor = alpha * self.mean + sigma * prior.mean
        cov_tensor = (alpha**2).unsqueeze(2) * self.cov + (sigma**2).unsqueeze(2) * prior.cov
        return MultivariateGaussian(
            mean_tensor[0],
            cov_tensor[0],
            normalize=self.normalize_mode,
            device=self.device,
        )

    def grad_log_prob(self, x):
        return -torch.einsum("ij,bj->bi", self.cov_inv, (x - self.mean))


class FixedPointDistribution(nn.Module):
    def __init__(self, point: torch.Tensor, device: str = "cpu"):
        super().__init__()
        assert len(point.shape) == 1
        self.point = point.to(device)
        self.device = device

    def export_score_function(self, scheduler: Scheduler):
        prior = MultivariateGaussian(
            torch.zeros(self.point.shape[0]),
            torch.eye(self.point.shape[0]),
            device=self.device,
        )

        def score(
            t: Float[torch.Tensor, "B 1"],
            x: Float[torch.Tensor, "B D"],
            prior: MultivariateGaussian,
            point: torch.Tensor,
        ):
            alpha = scheduler.ddpm_alpha2(t.squeeze()).sqrt()
            sigma = scheduler.ddpm_sigma2(t.squeeze()).sqrt()

            convolved: MultivariateGaussian = sigma[0] * prior + alpha[0] * point  # type: ignore
            return convolved.grad_log_prob(x)

        score_fn = functools.partial(score, prior=prior, point=self.point)

        return score_fn
