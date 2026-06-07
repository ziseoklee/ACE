import functools

import torch
import torch.nn as nn


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

    def score(self, x, alpha, sigma, prior):
        mean_tensor = alpha * self.mean + sigma * prior.mean
        cov_tensor = (alpha**2).unsqueeze(2) * self.cov + (sigma**2).unsqueeze(2) * prior.cov
        cov_inv_tensor = torch.linalg.inv(cov_tensor)
        # print(f'x shape: {x.shape}, mean_tensor shape: {mean_tensor.shape}, cov_tensor shape: {cov_tensor.shape}, cov_inv_tensor shape: {cov_inv_tensor.shape}')
        score = -torch.einsum("ij,ijk->ik", x - mean_tensor, cov_inv_tensor)
        return score

    def sample_test_set(self, M):
        L = torch.linalg.cholesky(self.cov)
        z = torch.randn(M, self.N, dtype=self.mean.dtype, device=self.device)
        samples = self.mean + torch.matmul(z, L.T)
        return samples

    def conditional_sample_test_set(self, conditioned_indices, x):
        # x: M x N
        conditioned_indices = torch.tensor(conditioned_indices, dtype=torch.long, device=self.device)
        all_indices = torch.arange(self.mean.size(0), dtype=torch.long, device=self.device)

        # Find unconditioned indices
        unconditioned_indices = all_indices[~torch.isin(all_indices, conditioned_indices)]

        # Use .detach() to ensure no gradients are tracked for mean and covariance
        mu = self.mean
        Sigma = self.cov

        # Split the mean and covariance into blocks
        mu_A = mu[unconditioned_indices]
        mu_B = mu[conditioned_indices]

        Sigma_AA = Sigma[unconditioned_indices[:, None], unconditioned_indices]
        Sigma_AB = Sigma[unconditioned_indices[:, None], conditioned_indices]
        Sigma_BA = Sigma[conditioned_indices[:, None], unconditioned_indices]
        Sigma_BB = Sigma[conditioned_indices[:, None], conditioned_indices]

        Sigma_BB_inv = torch.linalg.inv(Sigma_BB)

        mu_A_star = mu_A + torch.einsum("ij,kj->ki", Sigma_AB @ Sigma_BB_inv, (x - mu_B))
        Sigma_A_star = Sigma_AA - torch.einsum("ij,kj->ki", Sigma_AB @ Sigma_BB_inv, Sigma_BA)

        L = torch.linalg.cholesky(Sigma_A_star)
        z = torch.randn(
            mu_A_star.shape[0],
            mu_A_star.shape[1],
            dtype=self.mean.dtype,
            device=self.mean.device,
        )
        samples = mu_A_star + torch.matmul(z, L.T)

        return samples

    def marginal_dist(self, indices):
        # return new MultivariateGaussian for only the conditional indices
        return MultivariateGaussian(
            self.mean[torch.tensor(indices)],
            self.cov[torch.tensor(indices)[:, None], torch.tensor(indices)],
            normalize=self.normalize_mode,
            device=self.device,
        )


class FixedPointDistribution(nn.Module):
    def __init__(self, point: torch.Tensor, device: str = "cpu"):
        super().__init__()
        assert len(point.shape) == 1
        self.point = point.to(device)
        self.device = device

    def score(self, x, alpha, sigma, prior):
        convolved = sigma[0] * prior + alpha[0] * self.point
        return convolved.grad_log_prob(x)

    def export_score_function(self, scheduler):
        prior = MultivariateGaussian(
            torch.zeros(self.point.shape[0]),
            torch.eye(self.point.shape[0]),
            device=self.device,
        )
        score_fn = functools.partial(self.score, prior=prior)

        def score_function(t, x):
            alpha = scheduler.mean_coeff(t)
            sigma = scheduler.h(t) ** 0.5
            return score_fn(x, alpha, sigma)

        return score_function
