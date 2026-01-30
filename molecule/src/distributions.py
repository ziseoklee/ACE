import torch
import functools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


class MultivariateGaussian(nn.Module):
    def __init__(self, mean, cov, normalize=True, device='cpu'):
        super().__init__()
        self.device = device
        # breakpoint()
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.cov = torch.tensor(cov, dtype=torch.float32).to(device)
        self.normalize_mode = normalize

        self.N = self.mean.shape[0]
        self.cov_inv = torch.linalg.inv(self.cov)
        self.log_det_cov = torch.logdet(self.cov)
        self.log_coeff = -0.5 * (self.N * torch.log(torch.tensor(2 * torch.pi).to(device)) + self.log_det_cov)

    def __add__(self, other):
        if isinstance(other, MultivariateGaussian):
            return MultivariateGaussian(self.mean + other.mean, self.cov + other.cov, normalize=self.normalize_mode, device=self.device)
        elif isinstance(other, torch.Tensor):
            return MultivariateGaussian(self.mean + other, self.cov, normalize=self.normalize_mode, device=self.device)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")
        
    def __rmul__(self, scalar):
        return MultivariateGaussian(self.mean * scalar, self.cov * scalar ** 2, normalize=self.normalize_mode, device=self.device)

    def forward(self, x):
        diff = x - self.mean
        quad_form = torch.einsum('...i,ij,...j->...', diff, self.cov_inv, diff)
        if self.normalize_mode:
            # Only quadratic part, no normalization constant
            return -0.5 * quad_form
        else:
            # Full log-probability
            return self.log_coeff - 0.5 * quad_form

    def log_prob(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        return self.log_coeff - 0.5 * torch.einsum('...i,ij,...j->...', x - self.mean, self.cov_inv, x - self.mean)

    def convolve(self, alpha, sigma, prior):
        # assert that alpha, sigma is consistent over a batch (= t is same for all items)
        mean_tensor = alpha * self.mean + sigma * prior.mean
        cov_tensor = (alpha ** 2).unsqueeze(2) * self.cov + (sigma ** 2).unsqueeze(2) * prior.cov
        return MultivariateGaussian(mean_tensor[0], cov_tensor[0], normalize=self.normalize_mode, device=self.device)
    
    def grad_log_prob(self, x):
        return - torch.einsum('ij,bj->bi', self.cov_inv, (x - self.mean))
    
    def score(self, x, alpha, sigma, prior):
        mean_tensor = alpha * self.mean + sigma * prior.mean
        cov_tensor = (alpha ** 2).unsqueeze(2) * self.cov + (sigma ** 2).unsqueeze(2) * prior.cov
        cov_inv_tensor = torch.linalg.inv(cov_tensor)
        # print(f'x shape: {x.shape}, mean_tensor shape: {mean_tensor.shape}, cov_tensor shape: {cov_tensor.shape}, cov_inv_tensor shape: {cov_inv_tensor.shape}')
        score = -torch.einsum('ij,ijk->ik', x - mean_tensor, cov_inv_tensor)
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

        mu_A_star = mu_A + torch.einsum('ij,kj->ki', Sigma_AB @ Sigma_BB_inv, (x - mu_B))
        Sigma_A_star = Sigma_AA - torch.einsum('ij,kj->ki', Sigma_AB @ Sigma_BB_inv, Sigma_BA)

        L = torch.linalg.cholesky(Sigma_A_star)
        z = torch.randn(mu_A_star.shape[0], mu_A_star.shape[1], dtype=self.mean.dtype, device=self.mean.device)
        samples = mu_A_star + torch.matmul(z, L.T)
        
        return samples
        
    def marginal_dist(self, indices):
        # return new MultivariateGaussian for only the conditional indices
        return MultivariateGaussian(self.mean[torch.tensor(indices)], self.cov[torch.tensor(indices)[:, None], torch.tensor(indices)], normalize=self.normalize_mode, device=self.device)



class MultivariateGaussianMixture(nn.Module):
    @staticmethod
    def from_config(means, covs, log_weights, device='cpu'):
        gaussians = []
        for mean, cov, log_weight in zip(means, covs, log_weights):
            gaussians.append(MultivariateGaussian(mean, cov, normalize=True, device=device))    
        return MultivariateGaussianMixture(gaussians, log_weights, device=device)

    def __init__(self, components, log_weights, device='cpu'):
        super().__init__()
        self.device = device
        self.components = components
        self.log_weights = torch.tensor(log_weights, device=device)

    def forward(self, x):
        log_probs = torch.stack([component.log_prob(x) for component in self.components], dim=1)
        return torch.logsumexp(log_probs + self.log_weights, dim=1)

    def log_prob(self, x):
        log_probs = torch.stack([component.log_prob(x) for component in self.components], dim=1)
        return torch.logsumexp(log_probs + self.log_weights, dim=1)
        
    def convolve(self, alpha, sigma, prior):
        convolved_components = [alpha[0] * comp + sigma[0] * prior for comp in self.components]
        convolution_log_weights = self.log_weights
        convolved = MultivariateGaussianMixture(convolved_components, convolution_log_weights, device=self.device)
        return convolved
    
    def score(self, x, alpha, sigma, prior):
        convolved_components = [alpha[0] * comp + sigma[0] * prior for comp in self.components]
        convolution_log_weights = self.log_weights
        convolved = MultivariateGaussianMixture(convolved_components, convolution_log_weights, device=self.device)
        
        score = torch.func.jacfwd(convolved.log_prob)(x)[torch.arange(x.shape[0]), torch.arange(x.shape[0])]
        return score
    
    def sample_test_set(self, M):
        samples = []
        for component in self.components:
            samples.append(component.sample_test_set(M))
        samples = torch.stack(samples, dim=1)
        
        which_component_to_use = torch.distributions.categorical.Categorical(logits=self.log_weights).sample((M,)).to(self.device)
        samples = samples[torch.arange(M).to(self.device), which_component_to_use]
        return samples

    def conditional_sample_test_set(self, conditional_indices, x):
        samples = []
        likelihoods = []
        for component in self.components:
            likelihoods.append(component.marginal_dist(conditional_indices).log_prob(x))
            samples.append(component.conditional_sample_test_set(conditional_indices, x))
        samples = torch.stack(samples, dim=1)
        likelihoods = torch.stack(likelihoods, dim=1)
        posterior_log_weights = likelihoods + self.log_weights
        which_component_to_use = torch.distributions.categorical.Categorical(logits=posterior_log_weights).sample().to(self.device)
        samples = samples[torch.arange(x.shape[0]).to(self.device), which_component_to_use]
        return samples

    def marginal_dist(self, indices):
        return MultivariateGaussianMixture([component.marginal_dist(indices) for component in self.components], self.log_weights, device=self.device)


class FixedPointDistribution(nn.Module):
    def __init__(self, point, device='cpu'):
        super().__init__()
        assert len(point.shape) == 1
        self.point = point.to(device)
        self.device = device

    def sample_test_set(self, num_samples):
        return self.point.repeat(num_samples, 1)
    
    def score(self, x, alpha, sigma, prior):
        convolved =  sigma[0] * prior + alpha[0] * self.point
        return convolved.grad_log_prob(x)

    def export_score_function(self, scheduler):
        prior = MultivariateGaussian(torch.zeros(self.point.shape[0]), torch.eye(self.point.shape[0]), device=self.device)
        score_fn = functools.partial(self.score, prior=prior)
        def score_function(t, x):   
            alpha = scheduler.mean_coeff(t)
            sigma = scheduler.h(t) ** 0.5
            return score_fn(x, alpha, sigma)
        return score_function

class JointSampler(nn.Module):
    def __init__(self, dist_1, dist_2, indice_1, indice_2):
        super().__init__()
        self.dist_1 = dist_1
        self.dist_2 = dist_2
        self.indice_1 = indice_1
        self.indice_2 = indice_2

    def sample_test_set(self, num_samples):
        x = self.dist_1.sample_test_set(num_samples)
        y = self.dist_2.conditional_sample_test_set(self.indice_1, x)
        out = torch.zeros(num_samples, x.shape[1] + y.shape[1], device=x.device)
        out[:, self.indice_1] = x
        out[:, self.indice_2] = y
        return out


def sample_product_masked_diag(
    mu1, sigma1, mask, mu2, sigma2, n_samples=1
):
    """
    Sample from the (unnormalized) product N1(x_S | mu1, sigma1) * N2(x | mu2, sigma2),
    where N1 is defined only on a subset S of dimensions of N2, indicated by `mask`.
    All covariances are diagonal (given as stds).

    Args:
        mu1:    (..., K) tensor — mean of N1 on the masked dims (K = mask.sum()).
        sigma1: (..., K) tensor — std of N1 on the masked dims.
        mask:   (D,) or (..., D) bool tensor — which dims of N2 are covered by N1.
        mu2:    (..., D) tensor — mean of N2 on all D dims.
        sigma2: (..., D) tensor — std of N2 on all D dims.
        n_samples: int — number of samples to draw.

    Returns:
        samples: (n_samples, ..., D) tensor
        prod_mu: (..., D) tensor — mean of the product Gaussian
        prod_std:(..., D) tensor — std of the product Gaussian
    """
    # Basic checks
    if mu2.shape != sigma2.shape:
        raise ValueError("mu2 and sigma2 must have the same shape (..., D).")
    if mask.shape[-1] != mu2.shape[-1]:
        raise ValueError("mask last dim must equal D (the last dim of mu2).")
    K = mask[..., :].to(torch.bool).sum(dim=-1)
    if torch.any(K != mu1.shape[-1]):
        raise ValueError("Number of True in mask must equal mu1/sigma1 size on last dim.")

    # Ensure tensors
    mu2 = torch.as_tensor(mu2)
    sigma2 = torch.as_tensor(sigma2)
    mask  = torch.as_tensor(mask, dtype=torch.bool, device=mu2.device)

    # Broadcast mu1/sigma1 onto D dims using the mask
    # Build empty (..., D) and fill the masked positions
    prod_shape = mu2.shape
    device, dtype = mu2.device, mu2.dtype

    mu1_full    = torch.zeros(prod_shape, device=device, dtype=dtype)
    sigma1_full = torch.ones (prod_shape, device=device, dtype=dtype)  # placeholder; we'll invert below

    # We need to place mu1/sigma1 values into masked positions along the last dim.
    # Support batch masks: for generality, align shapes by flattening leading dims.
    lead_shape = prod_shape[:-1]
    D = prod_shape[-1]
    num_leads = int(torch.tensor(lead_shape).numel()) if len(lead_shape) > 0 else 1

    mu2_flat = mu2.reshape(num_leads, D)
    mu1_full_flat = mu1_full.reshape(num_leads, D)
    sigma1_full_flat = sigma1_full.reshape(num_leads, D)

    mask_flat = mask.reshape(num_leads, D)
    mu1_flat  = torch.as_tensor(mu1, device=device, dtype=dtype).reshape(num_leads, -1)
    sigma1_flat = torch.as_tensor(sigma1, device=device, dtype=dtype).reshape(num_leads, -1)

    # Fill per row
    for i in range(num_leads):
        m = mask_flat[i]
        k = int(m.sum().item())
        mu1_full_flat[i, m]    = mu1_flat[i, :k]
        sigma1_full_flat[i, m] = sigma1_flat[i, :k]

    mu1_full    = mu1_full_flat.reshape(prod_shape)
    sigma1_full = sigma1_full_flat.reshape(prod_shape)

    # Precisions (diagonal): tau = 1/var
    var2 = sigma2 ** 2
    tau2 = 1.0 / var2

    # For dims not covered by N1, set tau1 = 0 so product reduces to N2 there
    var1_full = sigma1_full ** 2
    tau1_full = torch.zeros_like(var2, device=device, dtype=dtype)
    tau1_full[mask] = (1.0 / var1_full[mask])

    # Product precision and natural mean
    tau = tau1_full + tau2                          # (..., D)
    eta = tau2 * mu2 + tau1_full * mu1_full         # (..., D)

    # Product parameters
    prod_var = 1.0 / tau
    prod_mu  = eta * prod_var
    prod_std = torch.sqrt(prod_var).clamp_min(1e-12)  # numerical safety

    # Sampling
    eps = torch.randn((n_samples,) + prod_mu.shape, device=device, dtype=dtype)
    samples = prod_mu.unsqueeze(0) + prod_std.unsqueeze(0) * eps

    return samples, prod_mu, prod_std