import torch
import functools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
# local imports (should import from root directory)
from utils.utils import pad_tensor
import torch.cuda as cuda
### HCG Class ###
# The following class implements the feymann-kac equation
# for pairs of conditional / unconditional distribution of subset for conditioning and 
# global distribution for their coherence.


def divergence_hutchinson(
    score_fn,
    t,
    x,
    n_probe: int = 1,
    rademacher: bool = True,
    create_graph: bool = False,   # sampling usually doesn't need higher-order grads
    t_eps: float = 1e-4,
    jitter: float = 1e-6,
    grad_clip: float = 1e6
):
    """
    Safe Hutchinson trace estimator: E_eps[ eps^T (∂ score / ∂x) eps ].

    - clamps t to (t_eps, 1 - t_eps)
    - jitters x -> x + jitter * eps to avoid 1/0 singularities inside score_fn
    - nan_to_num on score and grads
    - clips extreme grads
    - retries once with 10x jitter if backward still fails
    """
    # Ensure t is batch-shaped and safe
    if t.dim() == 0:
        t = t * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    else:
        # broadcast to (..., 1) if needed
        if t.shape[-1] != 1:
            t = t.view(-1, 1)
    t = t.clamp(t_eps, 1.0 - t_eps)

    # We need grads w.r.t x
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    div = x.new_zeros((x.shape[0], 1))
    for k in range(n_probe):
        eps = (
            torch.randint_like(x, low=0, high=2).float().mul_(2).sub_(1)
            if rademacher else torch.randn_like(x)
        )

        def one_pass(curr_jitter: float):
            with torch.enable_grad():
                xj = x + curr_jitter * eps  # break exact coincidences / zeros
                s = score_fn(t, xj)
                # sanitize forward values (don’t crash on inf/nan)
                s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                phi = (s * eps).sum(dim=-1)
                (g,) = torch.autograd.grad(
                    phi.sum(), x,
                    retain_graph=(k + 1 < n_probe),
                    create_graph=create_graph,
                    allow_unused=True
                )
                if g is None:
                    g = torch.zeros_like(x)
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                if grad_clip is not None:
                    g = torch.clamp(g, -grad_clip, grad_clip)
                return (g * eps).sum(dim=-1, keepdim=True)

        try:
            contrib = one_pass(jitter)
        except RuntimeError as e:
            print(f'RuntimeError: {e}')
            # Often "ReciprocalBackward0 returned nan"; try with a bigger jitter once.
            contrib = one_pass(jitter * 10)

        div = div + contrib
    # print(f'div: {div}')
    return div / n_probe

class HcgSingleCondition1(torch.nn.Module):
    def __init__(self, noise_schedule, q1, q2, q3, cond_dim=1, dim=2, gamma=1.0):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.gamma = gamma

        self.cond_dim = cond_dim
        self.dim = dim

    def f(self, t, x, resampling_interval=None):        
        v_1 = self.q1.v(t, x[:, :self.cond_dim])
        v_2 = self.q2.v(t, x[:, :self.cond_dim])
        v_3 = self.q3.v(t, x)

        s_1 = self.q1.score(t, x[:, :self.cond_dim])
        s_2 = self.q2.score(t, x[:, :self.cond_dim])
        s_3 = self.q3.score(t, x)

        sigma = self.q1.g(t)

        x_subset = x[:, :self.cond_dim]
        div_v_2 = torch.func.jacfwd(self.q2.v, argnums=1)(t, x_subset).reshape(x_subset.shape[0] * x_subset.shape[1], x_subset.shape[0] * x_subset.shape[1]).diag().reshape(x_subset.shape[0], x_subset.shape[1]).sum(dim=1).unsqueeze(1)
        v_star = pad_tensor(v_1 + v_2, self.dim, dim=1) + v_3 + 0.5 * (1 + self.gamma ** 2) * sigma ** 2 * (pad_tensor(s_1 - s_2, self.dim, dim=1) + s_3)

        g_bar = -v_1.unsqueeze(1) @ s_2.unsqueeze(2) + v_2.unsqueeze(1) @ s_1.unsqueeze(2) + 2 * div_v_2.unsqueeze(2) + pad_tensor(s_1 - s_2, self.dim, dim=1).unsqueeze(1) @ v_3.unsqueeze(2) + pad_tensor(v_1 + v_2, self.dim, dim=1).unsqueeze(1) @ s_3.unsqueeze(2) - sigma.unsqueeze(2) ** 2 * (s_2.unsqueeze(1) @ s_2.unsqueeze(2) - s_1.unsqueeze(1) @ s_2.unsqueeze(2) + pad_tensor(s_1 - s_2, self.dim, dim=1).unsqueeze(1) @ s_3.unsqueeze(2))

        return v_star, g_bar.squeeze(2).squeeze(1)

    def g(self, t):
        return self.q1.g(t)

    def diffusion(self, t, x):
        return self.gamma * self.g(t) * torch.randn_like(x).to(x.device)


class HcgSingleCondition2(torch.nn.Module):
    def __init__(self, noise_schedule, q1, q2, q3, cond_dim=1, dim=2, gamma=1.0):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.gamma = gamma

        self.cond_dim = cond_dim
        self.dim = dim

    def f(self, t, x, resampling_interval=None):   
        v_1 = pad_tensor(self.q1.v(t, x[:, :self.cond_dim]), self.dim, dim=1)
        v_2 = pad_tensor(self.q2.v(t, x[:, :self.cond_dim]), self.dim, dim=1)
        v_3 = pad_tensor(self.q3.v(t, x), self.dim, dim=1)

        s_1 = pad_tensor(self.q1.score(t, x[:, :self.cond_dim]), self.dim, dim=1)
        s_2 = pad_tensor(self.q2.score(t, x[:, :self.cond_dim]), self.dim, dim=1)
        s_3 = pad_tensor(self.q3.score(t, x), self.dim, dim=1)

        x_subset = x[:, :self.cond_dim]
        div_v_1 = torch.func.jacfwd(self.q1.v, argnums=1)(t, x_subset).reshape(x_subset.shape[0] * x_subset.shape[1], x_subset.shape[0] * x_subset.shape[1]).diag().reshape(x_subset.shape[0], x_subset.shape[1]).sum(dim=1).unsqueeze(1)
        div_v_2 = torch.func.jacfwd(self.q2.v, argnums=1)(t, x_subset).reshape(x_subset.shape[0] * x_subset.shape[1], x_subset.shape[0] * x_subset.shape[1]).diag().reshape(x_subset.shape[0], x_subset.shape[1]).sum(dim=1).unsqueeze(1)

        sigma = self.q3.g(t)

        A = s_1 - s_2 + s_3 
        B = (s_1.unsqueeze(1) @ s_1.unsqueeze(2) - s_2.unsqueeze(1) @ s_2.unsqueeze(2) + s_3.unsqueeze(1) @ s_3.unsqueeze(2)).squeeze()
        C = - ((div_v_1 - div_v_2).unsqueeze(2) + s_1.unsqueeze(1) @ (v_1 - v_3).unsqueeze(2) - s_2.unsqueeze(1) @ (v_2 - v_3).unsqueeze(2)).squeeze()

        v_star = v_3 + 0.5 * (1 + self.gamma ** 2) * sigma ** 2 * A
        g_bar = 0.5 * sigma.squeeze(1) ** 2 * ((A.unsqueeze(1) @ A.unsqueeze(2)).squeeze() - B) + C

        return v_star, g_bar

    def g(self, t):
        return self.q1.g(t)

    def diffusion(self, t, x):
        return self.gamma * self.g(t) * torch.randn_like(x).to(x.device)


class HcgSingleCondition2_noReweight(torch.nn.Module):
    def __init__(self, noise_schedule, q1, q2, q3, cond_dim=1, dim=2, gamma=1.0):
        self.noise_schedule = noise_schedule
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.gamma = gamma

        self.cond_dim = cond_dim
        self.dim = dim

    def f(self, t, x, resampling_interval=None):   
        v_1 = pad_tensor(self.q1.v(t, x[:, :self.cond_dim]), self.dim, dim=1)
        v_2 = pad_tensor(self.q2.v(t, x[:, :self.cond_dim]), self.dim, dim=1)
        v_3 = pad_tensor(self.q3.v(t, x), self.dim, dim=1)

        s_1 = pad_tensor(self.q1.score(t, x[:, :self.cond_dim]), self.dim, dim=1)
        s_2 = pad_tensor(self.q2.score(t, x[:, :self.cond_dim]), self.dim, dim=1)
        s_3 = pad_tensor(self.q3.score(t, x), self.dim, dim=1)


        x_subset = x[:, :self.cond_dim]
        div_v_1 = torch.func.jacfwd(self.q1.v, argnums=1)(t, x_subset).reshape(x_subset.shape[0] * x_subset.shape[1], x_subset.shape[0] * x_subset.shape[1]).diag().reshape(x_subset.shape[0], x_subset.shape[1]).sum(dim=1).unsqueeze(1)
        div_v_2 = torch.func.jacfwd(self.q2.v, argnums=1)(t, x_subset).reshape(x_subset.shape[0] * x_subset.shape[1], x_subset.shape[0] * x_subset.shape[1]).diag().reshape(x_subset.shape[0], x_subset.shape[1]).sum(dim=1).unsqueeze(1)

        sigma = self.q3.g(t)

        A = s_1 - s_2 + s_3 
        B = (s_1.unsqueeze(1) @ s_1.unsqueeze(2) - s_2.unsqueeze(1) @ s_2.unsqueeze(2) + s_3.unsqueeze(1) @ s_3.unsqueeze(2)).squeeze()
        C = - ((div_v_1 - div_v_2).unsqueeze(2) + s_1.unsqueeze(1) @ (v_1 - v_3).unsqueeze(2) - s_2.unsqueeze(1) @ (v_2 - v_3).unsqueeze(2)).squeeze()

        v_star = v_3 + 0.5 * (1 + self.gamma ** 2) * sigma ** 2 * A
        g_bar = torch.zeros_like(0.5 * sigma.squeeze(1) ** 2 * ((A.unsqueeze(1) @ A.unsqueeze(2)).squeeze() - B) + C)

        return v_star, g_bar

    def g(self, t):
        return self.q1.g(t)

    def diffusion(self, t, x):
        return self.gamma * self.g(t) * torch.randn_like(x).to(x.device)

class HcgSingleCondition2_noC(torch.nn.Module):
    def __init__(self, noise_schedule, q1, q2, q3, cond_dim=1, dim=2, gamma=1.0):
        self.noise_schedule = noise_schedule
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.gamma = gamma

        self.cond_dim = cond_dim
        self.dim = dim

    def f(self, t, x, resampling_interval=None):   
        v_1 = pad_tensor(self.q1.v(t, x[:, :self.cond_dim]), self.dim, dim=1)
        v_2 = pad_tensor(self.q2.v(t, x[:, :self.cond_dim]), self.dim, dim=1)
        v_3 = pad_tensor(self.q3.v(t, x), self.dim, dim=1)

        s_1 = pad_tensor(self.q1.score(t, x[:, :self.cond_dim]), self.dim, dim=1)
        s_2 = pad_tensor(self.q2.score(t, x[:, :self.cond_dim]), self.dim, dim=1)
        s_3 = pad_tensor(self.q3.score(t, x), self.dim, dim=1)


        x_subset = x[:, :self.cond_dim]
        div_v_1 = torch.func.jacfwd(self.q1.v, argnums=1)(t, x_subset).reshape(x_subset.shape[0] * x_subset.shape[1], x_subset.shape[0] * x_subset.shape[1]).diag().reshape(x_subset.shape[0], x_subset.shape[1]).sum(dim=1).unsqueeze(1)
        div_v_2 = torch.func.jacfwd(self.q2.v, argnums=1)(t, x_subset).reshape(x_subset.shape[0] * x_subset.shape[1], x_subset.shape[0] * x_subset.shape[1]).diag().reshape(x_subset.shape[0], x_subset.shape[1]).sum(dim=1).unsqueeze(1)

        sigma = self.q3.g(t)

        A = s_1 - s_2 + s_3 
        B = (s_1.unsqueeze(1) @ s_1.unsqueeze(2) - s_2.unsqueeze(1) @ s_2.unsqueeze(2) + s_3.unsqueeze(1) @ s_3.unsqueeze(2)).squeeze()
        C = - ((div_v_1 - div_v_2).unsqueeze(2) + s_1.unsqueeze(1) @ (v_1 - v_3).unsqueeze(2) - s_2.unsqueeze(1) @ (v_2 - v_3).unsqueeze(2)).squeeze()

        v_star = v_3 + 0.5 * (1 + self.gamma ** 2) * sigma ** 2 * A
        g_bar = 0.5 * sigma.squeeze(1) ** 2 * ((A.unsqueeze(1) @ A.unsqueeze(2)).squeeze() - B) 

        return v_star, g_bar

    def g(self, t):
        return self.q1.g(t)

    def diffusion(self, t, x):
        return self.gamma * self.g(t) * torch.randn_like(x).to(x.device)

class HcgSingleConditionMask(torch.nn.Module):
    def __init__(self, noise_schedule, q1, q2, q3, mask_list, dim=2, gamma=1.0, reg_score=None):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.gamma = gamma

        self.mask_list = mask_list
        self.dim = dim
        self.reg_score = reg_score

    def f(self, t, x, resampling_interval=None):   
        v_1 = pad_tensor(self.q1.v(t, x[:, self.mask_list[0]]), self.dim, dim=1)
        v_2 = pad_tensor(self.q2.v(t, x[:, self.mask_list[1]]), self.dim, dim=1)
        v_3 = pad_tensor(self.q3.v(t, x[:, self.mask_list[2]]), self.dim, dim=1)

        s_1 = pad_tensor(self.q1.score(t, x[:, self.mask_list[0]]), self.dim, dim=1).clamp(-20, 20)   
        s_2 = pad_tensor(self.q2.score(t, x[:, self.mask_list[1]]), self.dim, dim=1).clamp(-20, 20)
        s_3 = pad_tensor(self.q3.score(t, x[:, self.mask_list[2]]), self.dim, dim=1).clamp(-20, 20)
        if self.reg_score is not None:
            score_reg = self.reg_score(x)
        else:
            score_reg = torch.zeros_like(x)


        x_subset_1 = x[:, self.mask_list[0]]
        x_subset_2 = x[:, self.mask_list[1]]
        x_subset_3 = x[:, self.mask_list[2]]

        v = v_1 - v_2 + v_3

        sigma = self.noise_schedule.g(1 - t)

        A = s_1 - s_2 + s_3 + score_reg
        B = (s_1.unsqueeze(1) @ s_1.unsqueeze(2) - s_2.unsqueeze(1) @ s_2.unsqueeze(2) + s_3.unsqueeze(1) @ s_3.unsqueeze(2) + score_reg.unsqueeze(1) @ score_reg.unsqueeze(2)).squeeze()
        C = - (s_1.unsqueeze(1) @ (v_1 - v).unsqueeze(2) - s_2.unsqueeze(1) @ (v_2 - v).unsqueeze(2) + s_3.unsqueeze(1) @ (v_3 - v).unsqueeze(2) + score_reg.unsqueeze(1) @ (-v).unsqueeze(2)).squeeze() 

        v_star = v + 0.5 * (1 + self.gamma ** 2) * sigma ** 2 * A
        g_bar = 0.5 * sigma.squeeze(1) ** 2 * ((A.unsqueeze(1) @ A.unsqueeze(2)).squeeze() - B) + C

        return v_star, g_bar

    def g(self, t):
        return self.q1.g(t)

    def diffusion(self, t, x):
        return self.gamma * self.g(t) * torch.randn_like(x).to(x.device)


class HcgSingleConditionMaskExpanded(torch.nn.Module):
    def __init__(self, noise_schedule, q_list, mask_list, exponent_list, dim=2, gamma=2.5, reg_score=None, use_bump=False):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.q_list = q_list
        self.exponent_list = exponent_list
        self.gamma = gamma

        self.mask_list = mask_list
        self.dim = dim
        self.reg_score = reg_score
        self.use_bump = use_bump
        
    def f(self, t, x, logq_tensor, resampling_interval=None, calc_dlogq_tensor=True):   

        v_tensor = torch.stack([pad_tensor(q.v(t, x[:, mask]), self.dim, dim=1) for q, mask in zip(self.q_list, self.mask_list)], dim=1)
        sigma_tensor = torch.stack([q.g(t) for q in self.q_list], dim=1)

        s_tensor = torch.stack([pad_tensor(q.score(t, x[:, mask]), self.dim, dim=1).clamp(-20, 20) for q, mask in zip(self.q_list, self.mask_list)], dim=1)
        v_star = v_tensor.sum(dim=1)
        e_tensor = torch.stack([exponent_fn(t) for exponent_fn in self.exponent_list], dim=1)
        de_tensor = torch.stack([torch.func.jacfwd(exponent_fn, argnums=0)(t).squeeze().diag().unsqueeze(1) for exponent_fn in self.exponent_list], dim=1)
        s_star = (e_tensor * s_tensor).sum(dim=1)
        sigma = 2.5 * self.noise_schedule.g(1 - t)

        v_tensor = v_tensor + sigma_tensor ** 2 / 2 * s_tensor
        v_star = (e_tensor * v_tensor).sum(dim=1)
        
        if self.use_bump:
            v_net = v_star + sigma ** 2 / 2 * s_star

            if calc_dlogq_tensor:
                x_subset_list = [x[:, mask] for mask in self.mask_list]
                div_v_tensor = torch.stack([divergence_hutchinson(q.v, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)], dim=1)
                div_s_tensor = torch.stack([divergence_hutchinson(q.score, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)], dim=1)
                div_v_tensor = div_v_tensor + sigma_tensor ** 2 / 2 * div_s_tensor
        
                dlogq_tensor_A = (-div_v_tensor + torch.einsum('bij,bij->bi', v_net.unsqueeze(1) - v_tensor, s_tensor).unsqueeze(2) + sigma.unsqueeze(2) ** 2 / 2 * (torch.einsum('bj,bij->bi', s_star, s_tensor).unsqueeze(2) + div_s_tensor))
                dlogq_tensor_B = sigma.unsqueeze(2) * s_tensor
                dlogq_tensor_A = dlogq_tensor_A * resampling_interval
                dlogq_tensor_B = dlogq_tensor_B * resampling_interval
            else:
                dlogq_tensor_A = None
                dlogq_tensor_B = None

            g_net = (e_tensor * torch.einsum('bij,bij->bi', v_star.unsqueeze(1) - v_tensor, s_tensor).unsqueeze(2)).sum(dim=1).squeeze(1) + (de_tensor * logq_tensor).sum(dim=1).squeeze(1)
        
        else:
            v_net = v_star + sigma ** 2 / 2 * s_star
            g_net = (e_tensor * torch.einsum('bij,bij->bi', v_star.unsqueeze(1) - v_tensor, s_tensor).unsqueeze(2)).sum(dim=1).squeeze(1)
            dlogq_tensor_A = None
            dlogq_tensor_B = None
            
        return v_net, g_net, dlogq_tensor_A, dlogq_tensor_B

    def g(self, t):
        return self.gamma * self.noise_schedule.g(1 - t)

    def diffusion(self, t, x):
        return self.gamma * self.g(t) * torch.randn_like(x).to(x.device)

class HcgSingleConditionMaskExpandedODE(torch.nn.Module):
    def __init__(self, noise_schedule, q_list, mask_list, exponent_list, dim=2, gamma=1.0, reg_score=None, use_logq=False):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.q_list = q_list
        self.exponent_list = exponent_list
        self.gamma = gamma

        self.mask_list = mask_list
        self.dim = dim
        self.reg_score = reg_score
        self.use_logq = use_logq
        
    def f(self, t, x, logq_tensor, resampling_interval=None):   
        print(f't: {t[0].item()}')
        v_tensor = torch.stack([pad_tensor(q.v(t, x[:, mask]), self.dim, dim=1) for q, mask in zip(self.q_list, self.mask_list)], dim=1)
        sigma_tensor = torch.stack([q.g(t) for q in self.q_list], dim=1)
        s_tensor = torch.stack([pad_tensor(q.score(t, x[:, mask]), self.dim, dim=1).clamp(-20, 20) for q, mask in zip(self.q_list, self.mask_list)], dim=1)
        v_star = v_tensor.sum(dim=1)
        e_tensor = torch.stack([exponent_fn(t) for exponent_fn in self.exponent_list], dim=1)
        de_tensor = torch.stack([torch.func.jacfwd(exponent_fn, argnums=0)(t).squeeze().diag().unsqueeze(1) for exponent_fn in self.exponent_list], dim=1)
        s_star = (e_tensor * s_tensor).sum(dim=1)
        sigma = self.noise_schedule.g(1 - t)

        v_tensor = v_tensor + sigma_tensor ** 2 / 2 * s_tensor
        v_star = (e_tensor * v_tensor).sum(dim=1)
        
        # print(f'log q tensor: {logq_tensor}')
        if self.use_logq:
            x_subset_list = [x[:, mask] for mask in self.mask_list]
            div_v_tensor = torch.stack([divergence_hutchinson(q.v, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)], dim=1)
            div_s_tensor = torch.stack([divergence_hutchinson(q.score, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)], dim=1)
            div_v_tensor = div_v_tensor + sigma_tensor ** 2 / 2 * div_s_tensor
            dlogq_tensor_A = (-div_v_tensor + torch.einsum('bij,bij->bi', v_star.unsqueeze(1) - v_tensor, s_tensor).unsqueeze(2))
            dlogq_tensor_B = 0

            v_net = v_star
            g_net = (e_tensor * torch.einsum('bij,bij->bi', v_star.unsqueeze(1) - v_tensor, s_tensor).unsqueeze(2)).sum(dim=1) + (de_tensor * logq_tensor).sum(dim=1)
        
        else:
            assert False
            
        return v_net, g_net, dlogq_tensor_A, dlogq_tensor_B

    def g(self, t):
        return self.noise_schedule.g(1 - t)


# ProbabilityPath-based SDE 

class ProbabilityPathHcgWraper(torch.nn.Module):
    def __init__(self, probability_path):
        assert probability_path.reverse == True, "ProbabilityPath must be reversed"
        self.probability_path = probability_path
        self.scheduler = probability_path.scheduler

    def f(self, t, x):
        drift_coeff = self.probability_path.drift_coeff(t, x)
        reweight_coeff = torch.zeros(len(x)).to(x.device)

        return drift_coeff, reweight_coeff

    def g(self, t):
        return self.probability_path.g(t)

    def diffusion(self, t, x):
        return self.probability_path.diffusion_coeff(t) * torch.randn_like(x).to(x.device)

### Resampling functions ###

def sample_cat(d, bs, next_u, logits):
    device = logits.device
    sampler = qmc.Sobol(d=d, scramble=False)
    u = sampler.random(bs).squeeze()
    bins = torch.cumsum(torch.softmax(logits, dim=-1), dim=-1)
    ids = np.digitize(u, bins.cpu().numpy())
    ids = torch.tensor(ids, dtype=torch.long).to(device)
    return ids, next_u


def sample_cat_sys(bs, logits):
    device = logits.device
    u = torch.rand(size=(1,), device=device)
    u = (u + 1 / bs * torch.arange(bs, device=device)) % 1
    probs = torch.softmax(logits, dim=-1)
    bins = torch.cumsum(probs, dim=-1)
    ids = np.digitize(u.cpu().numpy(), bins.cpu().numpy(), right=True)

    ids[ids == logits.shape[-1]] = ids[ids == logits.shape[-1]] - 1

    return ids, None

def sample_cat_sys_safe(bs: int, logits: torch.Tensor, tol: float = 1e-6, stratified: bool = True):
    """
    Draw one categorical sample per row of `logits` using stratified uniforms.
    - Collapses tiny probs (<= tol) to 0 and renormalizes.
    - Uses torch.searchsorted on the CDF to avoid np.digitize monotonicity errors.

    Args:
        bs: number of samples (and/or batch size). If logits is 1D, we sample `bs` times from it.
        logits: shape (B, K) or (K,)
        tol: probabilities <= tol are collapsed to zero
        stratified: use stratified uniforms across [0,1)

    Returns:
        ids: LongTensor of shape (B,) with chosen category per row
        None: placeholder to match your original signature
    """
    device = logits.device
    dtype = logits.dtype

    # Make logits 2D: (B, K)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0).expand(bs, -1)
        B, K = logits.shape
    else:
        B, K = logits.shape
        assert B == bs, f"bs ({bs}) must match logits batch size ({B})"

    # Stable softmax, then collapse tiny probs
    probs = torch.softmax(logits, dim=-1)
    probs = torch.where(probs <= tol, torch.zeros_like(probs), probs)

    # Renormalize (rows that got fully zeroed become uniform)
    row_sum = probs.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(probs, 1.0 / K)
    probs = torch.where(row_sum > 0, probs / row_sum.clamp_min(torch.finfo(dtype).eps), uniform)

    # CDF (non-decreasing; duplicates OK)
    cdf = torch.cumsum(probs, dim=-1).clamp(max=1.0)

    # Stratified uniforms u in [0,1)
    if stratified:
        base = torch.rand((), device=device, dtype=dtype)
        # center within each stratum (optional but nice)
        u = (base + (torch.arange(B, device=device, dtype=dtype) + 0.5) / B) % 1.0
    else:
        u = torch.rand(B, device=device, dtype=dtype)
    # Search rightmost CDF edge strictly greater than u
    ids = torch.searchsorted(cdf, u.unsqueeze(-1), right=True).squeeze(-1)
    ids = ids.clamp_(0, K - 1)

    return ids.cpu().numpy(), None

### Rolling out Feynmann-Kac SDE ###

def euler_maruyama_step_coupled(
    sde,
    x,
    t,
    a,
    dt,
    step,
    resampling_interval,
    resampling_strategy,
):
    drift_Xt, drift_At = sde.f(t, x)
    diffusion = sde.diffusion(t, x)
    dx = drift_Xt * dt + diffusion * np.sqrt(dt)


    # Update the state
    x_next = x + dx                                                                    
    a_next = a + drift_At * dt
    if resampling_interval is None or step % resampling_interval != 0:
        return x_next, a_next, torch.arange(x.shape[0]).numpy()


    if resampling_strategy == "systematic":
        choice, _ = sample_cat_sys_safe(x.shape[0], a_next)
        a_next = torch.zeros_like(a_next)
        x_next = x_next[choice]


    return (
        x_next,
        a_next,
        choice,
    )


def euler_maruyama_step_coupled_expanded_ode(
    sde,
    x,
    t,
    a,
    logq_tensor,
    dt,
    step,
    resampling_interval,
    resampling_strategy,
):
    drift_Xt, drift_At, dlogq_tensor_A, dlogq_tensor_B = sde.f(t, x, logq_tensor)
    dx = drift_Xt * dt


    # Update the state
    x_next = x + dx                                                                    
    a_next = a + drift_At * dt
    if dlogq_tensor_A is not None:
        logq_tensor_next = logq_tensor + dlogq_tensor_A * dt
    else:
        logq_tensor_next = logq_tensor

    # if True:
    if resampling_interval is None or step % resampling_interval != 0:
        return x_next, a_next, logq_tensor_next, torch.arange(x.shape[0]).numpy()


    if resampling_strategy == "systematic":
        choice, _ = sample_cat_sys_safe(x.shape[0], a_next)
        a_next = torch.zeros_like(a_next)
        x_next = x_next[choice]
        logq_tensor_next = logq_tensor_next[choice]


    return (
        x_next,
        a_next,
        logq_tensor_next,
        choice,
    )
    
    
def euler_maruyama_step_coupled_expanded(
    sde,
    x,
    t,
    a,
    logq_tensor,
    dt,
    step,
    resampling_interval,
    resampling_strategy,
):
    drift_Xt, drift_At, dlogq_tensor_A, dlogq_tensor_B = sde.f(t, x, logq_tensor, resampling_interval=resampling_interval, calc_dlogq_tensor=step % resampling_interval == 0)
    dW = torch.randn_like(x) * np.sqrt(dt)
    sigma = sde.g(t)
    dx = drift_Xt * dt + sigma * dW

    # Update the state
    x_next = x + dx                                                                    
    a_next = a + drift_At * dt

    if dlogq_tensor_A is not None:
        logq_tensor_next = logq_tensor + dlogq_tensor_A * dt
    else:
        logq_tensor_next = logq_tensor
        
    if resampling_interval is None or step % resampling_interval != 1:
        return x_next, a_next, logq_tensor_next, torch.arange(x.shape[0]).numpy()


    if resampling_strategy == "systematic":
        choice, _ = sample_cat_sys_safe(x.shape[0], a_next)
        a_next = torch.zeros_like(a_next)
        x_next = x_next[choice]
        logq_tensor_next = logq_tensor_next[choice]


    return (
        x_next,
        a_next,
        logq_tensor_next,
        choice,
    )


def integrate_sde_coupled(
    sde, x0, t_span, dt, resampling_interval, resampling_strategy, interleave_fn=None
):
    device = x0.device
    times = torch.arange(t_span[0], t_span[1], dt).to(device)
    x = x0
    x0.requires_grad = True
    samples = []
    logweights = []
    choices = []
    a = torch.zeros(x.shape[0], device=device)

    with torch.no_grad():
        for step, t in enumerate(times):
            if t.dim() == 0:
                t = t * torch.ones(x.shape[0], 1).to(x.device)
            x, a, choice = (
                euler_maruyama_step_coupled(
                    sde,
                    x,
                    t,
                    a,
                    dt,
                    step + 1,
                    resampling_interval,
                    resampling_strategy=resampling_strategy,
                )
            )
            if interleave_fn is not None:
                x = interleave_fn(x, choice=choice)

            samples.append(x)
            logweights.append(a)
            choices.append(choice)

    return torch.stack(samples), torch.stack(logweights), np.array(choices)


def integrate_sde_coupled_expanded(
    sde, x0, logq_tensor, t_span, dt, resampling_interval, resampling_strategy, interleave_fn=None
):
    device = x0.device
    times = torch.arange(t_span[0], t_span[1], dt).to(device)
    x = x0
    x0.requires_grad = True
    samples = []
    logweights = []
    logq_tensor_list = []
    choices = []
    a = torch.zeros(x.shape[0], device=device)

    with torch.torch.no_grad():
        for step, t in enumerate(times):
            if t.dim() == 0:
                t = t * torch.ones(x.shape[0], 1).to(x.device)
            x, a, logq_tensor, choice = (
                euler_maruyama_step_coupled_expanded(
                    sde,
                    x,
                    t,
                    a,
                    logq_tensor,
                    dt,
                    step + 1,
                    resampling_interval,
                    resampling_strategy=resampling_strategy,
                )
            )
            if interleave_fn is not None:
                x = interleave_fn(x, choice=choice)

            samples.append(x.detach().cpu())
            logweights.append(a.detach().cpu())
            choices.append(choice)
            logq_tensor_list.append(logq_tensor.detach().cpu())
            cuda.empty_cache()

    return torch.stack(samples), torch.stack(logweights), torch.stack(logq_tensor_list), np.array(choices)





def integrate_sde_coupled_expanded_ode(
    sde, x0, logq_tensor, t_span, dt, resampling_interval, resampling_strategy, interleave_fn=None
):
    device = x0.device
    times = torch.arange(t_span[0], t_span[1], dt).to(device)
    x = x0
    x0.requires_grad = True
    samples = []
    logweights = []
    logq_tensor_list = []
    choices = []
    a = torch.zeros(x.shape[0], device=device)

    with torch.torch.no_grad():
        for step, t in enumerate(times):
            if t.dim() == 0:
                t = t * torch.ones(x.shape[0], 1).to(x.device)
            x, a, logq_tensor, choice = (
                euler_maruyama_step_coupled_expanded_ode(
                    sde,
                    x,
                    t,
                    a,
                    logq_tensor,
                    dt,
                    step + 1,
                    resampling_interval,
                    resampling_strategy=resampling_strategy,
                )
            )
            if interleave_fn is not None:
                x = interleave_fn(x, choice=choice)

            samples.append(x)
            logweights.append(a)
            choices.append(choice)
            logq_tensor_list.append(logq_tensor)

    return torch.stack(samples), torch.stack(logweights), torch.stack(logq_tensor_list), np.array(choices)


### Run Sampling ###
def generate_samples_weighted(
    reverse_sde,
    t_span=(0, 1),
    num_integration_steps=100,
    samples=None,
    num_samples=200,
    resampling_interval=None,
    prior=None,
    resampling_strategy="systematic",
    interleave_fn=None,
    logq_tensor=None,
    ode=False,
    return_intermediate=False,
):
    if samples is None:
        if prior is None:
            raise ValueError("Either samples or prior distribution should be provided")
        samples = prior.sample(num_samples)

    dt = 1 / num_integration_steps

    # breakpoint()
    if logq_tensor is not None:
        if ode:
            samples, weights, logq_tensor, choices = integrate_sde_coupled_expanded_ode(
                sde=reverse_sde,
                x0=samples,
                logq_tensor=logq_tensor,
                t_span=t_span,
                dt=dt,
                resampling_interval=resampling_interval,
                resampling_strategy=resampling_strategy,
                interleave_fn=interleave_fn,
            )
        else:
            print(f'integrate_sde_coupled_expanded')
            samples, weights, logq_tensor, choices = integrate_sde_coupled_expanded(
                sde=reverse_sde,
                x0=samples,
                logq_tensor=logq_tensor,
                t_span=t_span,
                dt=dt,
                resampling_interval=resampling_interval,
                resampling_strategy=resampling_strategy,
                interleave_fn=interleave_fn,
            )
    else:
        print(f'integrate_sde_coupled')
        samples, weights, choices = integrate_sde_coupled(
        sde=reverse_sde,
        x0=samples,
        t_span=t_span,
        dt=dt,
        resampling_interval=resampling_interval,
        resampling_strategy=resampling_strategy,
        interleave_fn=interleave_fn,
    )
    if return_intermediate:
        if logq_tensor is not None:
            return samples, weights, choices
        else:
            return samples, weights, choices
    else:
        if logq_tensor is not None:
            return samples[-1], weights, choices
        else:
            return samples[-1], weights, choices
