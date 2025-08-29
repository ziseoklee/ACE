# code for class and function definitions to import

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import math, time, random, itertools, os, gc, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.func import vmap, jvp
from IPython.display import clear_output

# !pip install --quiet torch torchvision matplotlib tqdm  # Colab: remove "!" locally

import math, time, random, itertools, os, gc, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.func import vmap, jvp
from IPython.display import clear_output

from datetime import datetime

import math, time, random, itertools, os, gc, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.func import vmap, jvp
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import math, time, random, itertools, os, gc, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.func import vmap, jvp
from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(0); random.seed(0); np.random.seed(0)
# print("device →", device)

class MLPInstFlexible(nn.Module):
    """
    Input: 
        z: tensor of shape [..., z_dim]
        t: scalar or tensor of shape [...]  (time scalar)
        c: optional conditioning tensor of shape [..., cond_dim] or None
    Output:
        tensor of shape [..., output_dim] (default 2)
    """
    def __init__(self, z_dim=2, cond_dim=0, width=256, depth=4, output_dim=2):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.width = width

        # Time embedding network (from scalar t to width-dim vector)
        self.time_proj = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU()
        )

        input_dim = z_dim + width + cond_dim  # total input dim

        layers = [nn.Linear(input_dim, width)]
        for _ in range(depth - 1):
            layers += [nn.SiLU(), nn.Linear(width, width)]
        self.net = nn.Sequential(*layers, nn.SiLU(), nn.Linear(width, output_dim))

    def forward(self, z, t, c=None):
        """
        z: tensor, shape [..., z_dim]
        t: scalar or tensor, shape [...] (same batch shape as z except last dim)
        c: tensor or None, shape [..., cond_dim]
        """
        device = next(self.parameters()).device  # 모델 파라미터가 있는 디바이스 추출
        z = z.to(device)
        t = t.to(device)
        if c is not None:
            c = c.to(device)

        if t.dim() == 0:
            t = t.unsqueeze(0)  # scalar to 1D
        if t.dim() == 2:
            t = t.squeeze(-1) # 2D to 1D

        t = t.unsqueeze(-1)  # [..., 1]
        t_emb = self.time_proj(t)  # [..., width]
        z = z.unsqueeze(0) if z.dim() == 1 else z  # [..., z_dim]        
        if c is not None:
            c = c.unsqueeze(-1) if c.dim() == 1 else c # [..., cond_dim]
            inputs = [z, t_emb, c]
        else:
            inputs = [z, t_emb]
                
        # print(f"torch.cat dimensions: {[inp.shape for inp in inputs]}")  # Debugging line
        h = torch.cat(inputs, dim=-1)  # [..., z_dim + width + cond_dim]
        return self.net(h)
    
def loss_per_sample_FM(u_theta, x, eps, t, cond=None):
    device = x.device
    eps = eps.to(device)  # <- ensure eps is on the same device
    t = t[:, None] if t.dim() == 1 else t
    t = t.to(device)  # <- ensure t is on the same device
    v_t = x - eps
    x_t = (1 - t) * eps + t * x

    if cond is not None:
        cond = cond.to(device)
        out = u_theta(x_t, t, cond).to(device)
    else:
        out = u_theta(x_t, t).to(device)

    return 0.5 * torch.sum(out.square()) - torch.sum(out * v_t)


def loss_per_sample_SM(s_theta, x, eps, t, cond=None):
    device = x.device
    eps = eps.to(device)  # <- ensure eps is on the same device
    t = t[:, None] if t.dim() == 1 else t
    t = t.to(device)
    alpha = 1. - t

    x_t = (1 - t) * eps + t * x
    if cond is not None:
        cond = cond.to(device)
        out = s_theta(x_t, t, cond).to(device)
    else:
        out = s_theta(x_t, t).to(device)

    loss = 0.5 * torch.sum(out.square()) + (1 / alpha) * torch.sum(out * eps)

    # Antisymmetric term
    eps = -eps
    x_t = (1 - t) * eps + t * x
    if cond is not None:
        out = s_theta(x_t, t, cond).to(device)
    else:
        out = s_theta(x_t, t).to(device)

    loss += 0.5 * torch.sum(out.square()) + (1 / alpha) * torch.sum(out * eps)
    return loss



def make_batch_loss(loss_sample):
    return vmap(loss_sample, in_dims=(None,0,0,0), randomness='different')

def train_step_FM_SM(u_theta, s_theta, opt_u, opt_s, sched_u, sched_s, batch_size, clip=1.0, sample_fn=None):
    # x, cond ← sample_data() must return both
    if sample_fn is None:
        raise ValueError("Must provide sample_fn(x, cond)")

    x, cond = sample_fn(batch_size)
    x, cond = x.to(device), cond.to(device)

    eps = torch.randn_like(x)
    t = torch.rand(size=(batch_size,), device=device)

    opt_u.zero_grad()
    opt_s.zero_grad()

    loss_u = make_batch_loss(loss_per_sample_FM)(u_theta, x, eps, t, cond).mean()
    loss_s = make_batch_loss(loss_per_sample_SM)(s_theta, x, eps, t, cond).mean()
    loss = loss_u + loss_s

    loss_u.backward()
    loss_s.backward()

    torch.nn.utils.clip_grad_norm_(u_theta.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(s_theta.parameters(), clip)

    opt_u.step()
    opt_s.step()
    sched_u.step()
    sched_s.step()

    return loss.item()

# @torch.no_grad()
# def n_step_sample_FM_SM(u_theta, s_theta, num, device, n_step=10, noise_level=1.0, cond_val=None):
#     z = torch.randn(num, u_theta.z_dim).to(device)  # generalize for 1D/2D
#     t_vals = torch.linspace(0.0, 1.0, n_step + 1, device=device)
#     sigma  = torch.tensor(noise_level, device=device)

#     if cond_val is not None:
#         cond = torch.full((num, u_theta.cond_dim), fill_value=cond_val, device=device)
#     else:
#         cond = None

#     for i in range(n_step):
#         t = t_vals[i].expand(num)
#         dt = t_vals[i + 1] - t_vals[i]
#         noise = torch.randn_like(z, device=device)

#         v = u_theta(z, t, cond).to(device) if cond is not None else u_theta(z, t).to(device)
#         s = s_theta(z, t, cond).to(device) if cond is not None else s_theta(z, t).to(device)

#         drift = v + sigma * s
#         diffusion = torch.sqrt(2 * sigma * dt)

#         z = z + drift * dt + diffusion * noise

#     return z

@torch.no_grad()
def n_step_sample_FM_SM(u_theta, s_theta, num, device, n_step=10, noise_level=1.0, cond_val=None, z_dim=2):
    z = torch.randn(num, z_dim).to(device)  # generalize for 1D/2D
    t_vals = torch.linspace(0.0, 1.0, n_step + 1, device=device)
    sigma  = torch.tensor(noise_level, device=device)

    if cond_val is not None:
        cond = torch.full((num, u_theta.cond_dim), fill_value=cond_val, device=device)
    else:
        cond = None

    for i in range(n_step):
        t = t_vals[i].expand(num)
        dt = t_vals[i + 1] - t_vals[i]
        noise = torch.randn_like(z, device=device)

        v = u_theta(z, t, cond).to(device) if cond is not None else u_theta(z, t).to(device)
        s = s_theta(z, t, cond).to(device) if cond is not None else s_theta(z, t).to(device)

        drift = v + sigma * s
        diffusion = torch.sqrt(2 * sigma * dt)

        z = z + drift * dt + diffusion * noise

    return z

@torch.no_grad()
def plot_state_FM_SM(u_theta, s_theta, data_fn, history, step, show_samples=4096, suptitle=True,
                     n_step=1, noise_level=1.0, cond_val=None):
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history, lw=2)
    axes[0].set_title("FM_SM training loss")
    axes[0].set_xlabel("Iteration")

    # Real vs. Generated Samples
    if cond_val is not None:
        real = data_fn(show_samples, cond_val=cond_val)[0].cpu()
    else:
        real = data_fn(show_samples)[0].cpu()

    fake = n_step_sample_FM_SM(u_theta, s_theta, show_samples, device,
                               n_step=n_step, noise_level=noise_level, cond_val=cond_val, z_dim=u_theta.z_dim).cpu()

    if real.shape[1] == 1:
        axes[1].hist(real.numpy(), bins=50, alpha=0.5, label="real", density=True)
        axes[1].hist(fake[:, 0].numpy(), bins=50, alpha=0.5, label="fake", density=True)
        axes[1].set_xlim(-5, 5)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("Value")
    else:
        axes[1].scatter(real[:, 0], real[:, 1], s=5, alpha=0.3, label="real")
        axes[1].scatter(fake[:, 0], fake[:, 1], s=5, alpha=0.3, label="fake")
        axes[1].set_xlim(-5, 5)
        axes[1].set_ylim(-5, 5)
        axes[1].set_aspect("equal", "box")

    axes[1].legend()
    axes[1].set_title(f"samples @ step {step}")

    if suptitle:
        fig.suptitle("FM_SM Visualization", fontsize=14)

    plt.show()


def run_training_v_s(velocity_theta, score_theta, sample_data_fn, plot_cond_val=None, n_iters=2000):
    batch_size   = 1024
    iters        = n_iters
    log_every    = 250
    lr           = 2e-3

    opt_velocity = AdamW(velocity_theta.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=0.)
    sched_velocity = CosineAnnealingLR(opt_velocity, T_max=iters, eta_min=1e-5)

    opt_score = AdamW(score_theta.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=0.)
    sched_score = CosineAnnealingLR(opt_score, T_max=iters, eta_min=1e-5)

    loss_hist = []

    start_t = time.time()
    for step in range(1, iters+1):
        # Call sample_data with cond_val if needed
        x, cond = sample_data_fn(batch_size)

        eps = torch.randn_like(x)
        t = torch.rand(size=(batch_size,), device=device)

        opt_velocity.zero_grad()
        opt_score.zero_grad()

        if cond is not None:
            loss_u = vmap(lambda m, x, e, t, c: loss_per_sample_FM(m, x, e, t, c),
                        in_dims=(None, 0, 0, 0, 0), randomness='different')(
                velocity_theta, x, eps, t, cond
            )

            loss_s = vmap(lambda m, x, e, t, c: loss_per_sample_SM(m, x, e, t, c),
                        in_dims=(None, 0, 0, 0, 0), randomness='different')(
                score_theta, x, eps, t, cond
            )

        else:
            loss_u = make_batch_loss(loss_per_sample_FM)(velocity_theta, x, eps, t)
            loss_s = make_batch_loss(loss_per_sample_SM)(score_theta, x, eps, t)


        loss = loss_u.mean() + loss_s.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(velocity_theta.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_theta.parameters(), 1.0)

        opt_velocity.step()
        opt_score.step()
        sched_velocity.step()
        sched_score.step()

        loss_hist.append(loss.item())

        if step % log_every == 0 or step == 1:
            dt = time.time() - start_t
            print(f"[{step:>6}/{iters}]  loss={loss:.4e}  ({dt/step:.3f}s/it)  lr={sched_velocity.get_last_lr()[0]:.2e}")
            plot_state_FM_SM(velocity_theta, score_theta, sample_data_fn, loss_hist, step,
                             suptitle=True, n_step=100, noise_level=0.5, show_samples=5000,
                             cond_val=plot_cond_val)


@torch.no_grad()
def n_step_sample_sde(
    z0,                        # [bs, d]
    v_list, s_list,            # list of functions: [v1_fn, v2_fn, ...], [s1_fn, s2_fn, ...]
    sigma_fn,                  # function of t → [bs, 1]
    t0=0.0, t1=1.0, n_steps=1000,
    device="cuda"
):
    z = z0.clone().to(device)
    bs = z.size(0)
    times = torch.linspace(t0, t1, n_steps + 1, device=device)
    dt = (t1 - t0) / n_steps

    for i in range(n_steps):
        t = times[i].expand(bs, 1)
        sigma_t = sigma_fn(t)  # [..., 1]

        # Compute total drift: v1 + v2 + ... + sigma^2 (s1 + s2 + ...)
        total_v = sum(v(z, t) for v in v_list)
        total_s = sum(s(z, t) for s in s_list)
        drift = total_v + sigma_t**2 * total_s

        # Add diffusion noise
        noise = torch.randn_like(z)
        z = z + drift * dt + torch.sqrt(2 * sigma_t * dt) * noise

    return z

















# ChatGPT-generated examples, may not be right for this case
def example_mode_suppression():  # Example 1 — vertical columns
    def pX(bs):
        mix = torch.bernoulli(0.5*torch.ones(bs,1,device=device))
        left  = torch.randn(bs,1,device=device)*0.4 + (-2.0)
        right = torch.randn(bs,1,device=device)*0.4 + (+2.0)
        return mix*right + (1-mix)*left
    def pX_givenA(bs):
        return torch.randn(bs,1,device=device)*0.2 + (-2.0)
    def pXY_givenB(bs):
        which = torch.bernoulli(0.5*torch.ones(bs,1,device=device))
        x_left  = torch.randn(bs,1,device=device)*0.12 + (-2.0)
        x_right = torch.randn(bs,1,device=device)*0.12 + (+2.0)
        x = which * x_right + (1-which) * x_left
        y = which * (torch.randn(bs,1,device=device)*0.5 + 1.0) + (1-which)*(torch.randn(bs,1,device=device)*0.5 - 1.0)
        return torch.cat([x,y], dim=1)
    return pX, pX_givenA, pXY_givenB

def example_branch_selection():  # Example 2 — branch selection Y depends on sign(X)
    def pX(bs):
        return torch.randn(bs,1,device=device)*1.5
    def pX_givenA(bs):
        return torch.randn(bs,1,device=device)*0.2 + 1.0   # concentrate positive X
    def pXY_givenB(bs):
        x = torch.randn(bs,1,device=device)*1.5
        y = torch.randn(bs,1,device=device)*0.5 + (torch.sign(x)*2.0)
        return torch.cat([x,y], dim=1)
    return pX, pX_givenA, pXY_givenB

def example_density_modulation():  # Example 3 — rectangle with X-weighting
    def pX(bs):
        return (torch.rand(bs,1,device=device)*6.0 - 3.0)
    def pX_givenA(bs):
        return torch.randn(bs,1,device=device)*0.3 + 0.0   # bump near 0
    def pXY_givenB(bs):
        x = torch.rand(bs,1,device=device)*6.0 - 3.0
        y = torch.rand(bs,1,device=device)*4.0 - 2.0
        return torch.cat([x,y], dim=1)
    return pX, pX_givenA, pXY_givenB

def example_manifold_segment():  # Example 4 — S-curve with X gating
    def pX(bs):
        return torch.randn(bs,1,device=device)*2.0
    def pX_givenA(bs):
        return torch.randn(bs,1,device=device)*0.1 + 1.5  # concentrate X~1.5
    def pXY_givenB(bs):
        x = torch.rand(bs,1,device=device)*6.0 - 3.0
        y = torch.sin(x) * 1.5 + torch.randn(bs,1,device=device)*0.12
        return torch.cat([x,y], dim=1)
    return pX, pX_givenA, pXY_givenB

# Cell 7 — Utility that wraps each sampler into the (x,cond) signature used by trainer
def make_data_fns(pX, pX_givenA, pXY_givenB):
    # q1: X | A  (we treat cond unused; return cond tensor for API)
    def sample_data_q1(batch_size, cond_val=None):
        if cond_val is None:
            return pX(batch_size), None
        else:
            return pX_givenA(batch_size), None

    # q2: X (unconditional marginal)
    def sample_data_q2(batch_size, cond_val=None):
        return pX(batch_size), None

    # q3: (X,Y) | B  (cond unused; returns XY)
    def sample_data_q3(batch_size, cond_val=None):
        return pXY_givenB(batch_size), None

    return sample_data_q1, sample_data_q2, sample_data_q3
