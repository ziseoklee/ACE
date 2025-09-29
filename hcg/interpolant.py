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






import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.func import vmap
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import math

# --- Helper Functions and Default Schedules ---

def default_alpha_t(t):
    """Default alpha_t schedule: 1 - t"""
    return 1.0 - t

def default_beta_t(t):
    """Default beta_t schedule: t"""
    return t

def default_d_alpha_t(t):
    """Derivative of default alpha_t: -1"""
    return -torch.ones_like(t)

def default_d_beta_t(t):
    """Derivative of default beta_t: 1"""
    return torch.ones_like(t)

# --- Interpolant Class for Scheduling ---

class Interpolant:
    """
    Manages the interpolation schedule (alpha_t, beta_t) and their derivatives.
    This allows for easy swapping of different schedules for flow matching.
    """
    def __init__(self, alpha_t=None, beta_t=None, d_alpha_t=None, d_beta_t=None, name=None):
        """
        Initializes the interpolant with custom or default schedules.

        Args:
            alpha_t (callable, optional): Function for alpha schedule. Defaults to 1-t.
            beta_t (callable, optional): Function for beta schedule. Defaults to t.
            d_alpha_t (callable, optional): Derivative of alpha_t. Defaults to -1.
            d_beta_t (callable, optional): Derivative of beta_t. Defaults to 1.
        """
        self.alpha_t = alpha_t if alpha_t is not None else default_alpha_t
        self.beta_t = beta_t if beta_t is not None else default_beta_t
        self.d_alpha_t = d_alpha_t if d_alpha_t is not None else default_d_alpha_t
        self.d_beta_t = d_beta_t if d_beta_t is not None else default_d_beta_t
        self.name = name if name is not None else "default"


# --- Main Flow Matcher Class ---

# class FlowMatcher:
#     """
#     A class to encapsulate the training, loss calculation, and sampling for
#     Flow Matching (FM) and Score Matching (SM) models with customizable interpolants.
#     """
#     def __init__(self, u_theta, s_theta, interpolant=None):
#         """
#         Initializes the FlowMatcher.

#         Args:
#             u_theta (nn.Module): The velocity model (u_theta).
#             s_theta (nn.Module): The score model (s_theta).
#             interpolant (Interpolant, optional): An instance of the Interpolant class.
#                                                 If None, uses the default linear schedule.
#         """
#         self.device = next(u_theta.parameters()).device
#         self.u_theta = u_theta.to(self.device)
#         self.s_theta = s_theta.to(self.device)
#         self.interpolant = interpolant if interpolant is not None else Interpolant()
#         self.loss_history = []

#     def _loss_per_sample_fm(self, x, eps, t, cond=None):
#         """Calculates the Flow Matching loss for a single sample."""
#         # Ensure all tensors are on the correct device
#         x, eps, t = x.to(self.device), eps.to(self.device), t.to(self.device)
#         t = t.view(-1, 1) if t.dim() == 1 else t

#         # Get schedule values from interpolant
#         alpha_t_val = self.interpolant.alpha_t(t)
#         beta_t_val = self.interpolant.beta_t(t)
#         d_alpha_t_val = self.interpolant.d_alpha_t(t)
#         d_beta_t_val = self.interpolant.d_beta_t(t)

#         # Calculate interpolated path x_t and velocity v_t
#         x_t = alpha_t_val * eps + beta_t_val * x
#         v_t = d_alpha_t_val * eps + d_beta_t_val * x

#         # Get model output
#         out = self.u_theta(x_t, t, cond.to(self.device) if cond is not None else None)

#         # Return MSE loss components for FM
#         return 0.5 * torch.sum(out.square()) - torch.sum(out * v_t)

#     def _loss_per_sample_sm(self, x, eps, t, cond=None):
#         """Calculates the Score Matching loss for a single sample."""
#         x, eps, t = x.to(self.device), eps.to(self.device), t.to(self.device)
#         t = t.view(-1, 1) if t.dim() == 1 else t
        
#         alpha_t_val = self.interpolant.alpha_t(t)
#         beta_t_val = self.interpolant.beta_t(t)

#         # Symmetric term
#         x_t_pos = alpha_t_val * eps + beta_t_val * x
#         out_pos = self.s_theta(x_t_pos, t, cond.to(self.device) if cond is not None else None)
#         loss = 0.5 * torch.sum(out_pos.square()) + (1 / alpha_t_val) * torch.sum(out_pos * eps)

#         # Antisymmetric term
#         eps_neg = -eps
#         x_t_neg = alpha_t_val * eps_neg + beta_t_val * x
#         out_neg = self.s_theta(x_t_neg, t, cond.to(self.device) if cond is not None else None)
#         loss += 0.5 * torch.sum(out_neg.square()) + (1 / alpha_t_val) * torch.sum(out_neg * eps_neg)
        
#         return loss
#     def train(self, sample_fn, n_iters=2000, batch_size=1024, lr=2e-3, clip=1.0, plot_cond_val=None, log_every=250):
#         """
#         Main training loop for the FM and SM models.

#         Args:
#             sample_fn (callable): Function that returns a batch of data (x, cond).
#             n_iters (int): Number of training iterations.
#             batch_size (int): The size of each batch.
#             lr (float): Learning rate.
#             clip (float): Gradient clipping value.
#             plot_cond_val: Conditional value to use for plotting.
#             log_every (int): How often to log progress and plot samples.
#         """
#         opt_u = AdamW(self.u_theta.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
#         sched_u = CosineAnnealingLR(opt_u, T_max=n_iters, eta_min=1e-5)

#         opt_s = AdamW(self.s_theta.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
#         sched_s = CosineAnnealingLR(opt_s, T_max=n_iters, eta_min=1e-5)

#         # Vectorize loss functions for batching
#         # The `in_dims` for `cond` is set to None so vmap doesn't try to batch it if it's None.
#         batch_loss_fm = vmap(self._loss_per_sample_fm, in_dims=(0, 0, 0, None), randomness='different')
#         batch_loss_sm = vmap(self._loss_per_sample_sm, in_dims=(0, 0, 0, None), randomness='different')

#         start_t = time.time()
#         for step in range(1, n_iters + 1):
#             opt_u.zero_grad()
#             opt_s.zero_grad()

#             x, cond = sample_fn(batch_size)
#             eps = torch.randn_like(x)
#             t = torch.rand(size=(batch_size,), device=self.device)

#             loss_u = batch_loss_fm(x, eps, t, cond).mean()
#             loss_s = batch_loss_sm(x, eps, t, cond).mean()
#             loss = loss_u + loss_s
            
#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(self.u_theta.parameters(), clip)
#             torch.nn.utils.clip_grad_norm_(self.s_theta.parameters(), clip)

#             opt_u.step()
#             opt_s.step()
#             sched_u.step()
#             sched_s.step()

#             self.loss_history.append(loss.item())

#             if step % log_every == 0 or step == 1:
#                 dt = time.time() - start_t
#                 print(f"[{step:>6}/{n_iters}]  loss={loss.item():.4e}  ({dt/step:.3f}s/it)  lr={sched_u.get_last_lr()[0]:.2e}")
#                 self.plot_state(sample_fn, self.loss_history, step, suptitle=True,
#                                 n_step=100, noise_level=0.5, show_samples=5000,
#                                 cond_val=plot_cond_val)
#     # def train(self, sample_fn, n_iters=2000, batch_size=1024, lr=2e-3, clip=1.0, plot_cond_val=None, log_every=250):
#     #     """
#     #     Main training loop for the FM and SM models.

#     #     Args:
#     #         sample_fn (callable): Function that returns a batch of data (x, cond).
#     #         n_iters (int): Number of training iterations.
#     #         batch_size (int): The size of each batch.
#     #         lr (float): Learning rate.
#     #         clip (float): Gradient clipping value.
#     #         plot_cond_val: Conditional value to use for plotting.
#     #         log_every (int): How often to log progress and plot samples.
#     #     """
#     #     opt_u = AdamW(self.u_theta.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
#     #     sched_u = CosineAnnealingLR(opt_u, T_max=n_iters, eta_min=1e-5)

#     #     opt_s = AdamW(self.s_theta.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
#     #     sched_s = CosineAnnealingLR(opt_s, T_max=n_iters, eta_min=1e-5)

#     #     # Vectorize loss functions for batching
#     #     batch_loss_fm = vmap(self._loss_per_sample_fm, in_dims=(0, 0, 0, 0), randomness='different')
#     #     batch_loss_sm = vmap(self._loss_per_sample_sm, in_dims=(0, 0, 0, 0), randomness='different')

#     #     start_t = time.time()
#     #     for step in range(1, n_iters + 1):
#     #         opt_u.zero_grad()
#     #         opt_s.zero_grad()

#     #         x, cond = sample_fn(batch_size)
#     #         eps = torch.randn_like(x)
#     #         t = torch.rand(size=(batch_size,), device=self.device)

#     #         loss_u = batch_loss_fm(x, eps, t, cond).mean()
#     #         loss_s = batch_loss_sm(x, eps, t, cond).mean()
#     #         loss = loss_u + loss_s
            
#     #         loss.backward()

#     #         torch.nn.utils.clip_grad_norm_(self.u_theta.parameters(), clip)
#     #         torch.nn.utils.clip_grad_norm_(self.s_theta.parameters(), clip)

#     #         opt_u.step()
#     #         opt_s.step()
#     #         sched_u.step()
#     #         sched_s.step()

#     #         self.loss_history.append(loss.item())

#     #         if step % log_every == 0 or step == 1:
#     #             dt = time.time() - start_t
#     #             print(f"[{step:>6}/{n_iters}]  loss={loss.item():.4e}  ({dt/step:.3f}s/it)  lr={sched_u.get_last_lr()[0]:.2e}")
#     #             self.plot_state(sample_fn, self.loss_history, step, suptitle=True,
#     #                             n_step=100, noise_level=0.5, show_samples=5000,
#     #                             cond_val=plot_cond_val)

#     @torch.no_grad()
#     def sample(self, num_samples, n_step=10, noise_level=1.0, cond_val=None):
#         """
#         Generate samples from the learned models.

#         Args:
#             num_samples (int): Number of samples to generate.
#             n_step (int): Number of steps in the sampling process.
#             noise_level (float): The level of noise for the diffusion term.
#             cond_val: Conditional value for sampling.

#         Returns:
#             torch.Tensor: The generated samples.
#         """
#         z_dim = self.u_theta.z_dim
#         z = torch.randn(num_samples, z_dim, device=self.device)
#         t_vals = torch.linspace(0.0, 1.0, n_step + 1, device=self.device)
#         sigma = torch.tensor(noise_level, device=self.device)

#         cond = torch.full((num_samples, self.u_theta.cond_dim), fill_value=cond_val, device=self.device) if cond_val is not None else None

#         for i in range(n_step):
#             t = t_vals[i].expand(num_samples)
#             dt = t_vals[i + 1] - t_vals[i]
#             noise = torch.randn_like(z)

#             v = self.u_theta(z, t, cond)
#             s = self.s_theta(z, t, cond)

#             drift = v + sigma * s
#             diffusion = torch.sqrt(2 * sigma * dt)

#             z = z + drift * dt + diffusion * noise
        
#         return z

#     @torch.no_grad()
#     def plot_state(self, data_fn, history, step, show_samples=4096, suptitle=True, n_step=1, noise_level=1.0, cond_val=None):
#         """Plots the current state of training (loss curve and sample comparison)."""
#         clear_output(wait=True)
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#         # Plot loss curve
#         axes[0].plot(history, lw=2)
#         axes[0].set_title("Training Loss")
#         axes[0].set_xlabel("Iteration")
#         axes[0].grid(True)

#         # Plot real vs. generated samples
#         real_data, _ = data_fn(show_samples, cond_val=cond_val)
#         real = real_data.cpu()
#         fake = self.sample(show_samples, n_step=n_step, noise_level=noise_level, cond_val=cond_val).cpu()

#         if real.shape[1] == 1: # 1D Data
#             axes[1].hist(real.numpy(), bins=50, alpha=0.5, label="real", density=True)
#             axes[1].hist(fake.numpy(), bins=50, alpha=0.5, label="fake", density=True)
#             axes[1].set_xlim(-5, 5)
#             axes[1].set_ylim(0, 1)
#             axes[1].set_xlabel("Value")
#         else: # 2D Data
#             axes[1].scatter(real[:, 0], real[:, 1], s=5, alpha=0.3, label="real")
#             axes[1].scatter(fake[:, 0], fake[:, 1], s=5, alpha=0.3, label="fake")
#             axes[1].set_aspect("equal", "box")
#             axes[1].set_xlim(-5, 5)
#             axes[1].set_ylim(-5, 5)
#         axes[1].legend()
#         axes[1].set_title(f"Samples @ Step {step}")

#         if suptitle:
#             fig.suptitle("Flow Matcher Training Visualization", fontsize=14)

#         plt.show()


class FlowMatcher:
    """
    A class to encapsulate the training, loss calculation, and sampling for
    Flow Matching (FM) and Score Matching (SM) models with customizable interpolants.
    """
    def __init__(self, u_theta, s_theta, interpolant=None):
        """
        Initializes the FlowMatcher.

        Args:
            u_theta (nn.Module): The velocity model (u_theta).
            s_theta (nn.Module): The score model (s_theta).
            interpolant (Interpolant, optional): An instance of the Interpolant class.
                                                If None, uses the default linear schedule.
        """
        self.device = next(u_theta.parameters()).device
        self.u_theta = u_theta.to(self.device)
        self.s_theta = s_theta.to(self.device)
        self.interpolant = interpolant if interpolant is not None else Interpolant()
        self.loss_history = []

    def _loss_per_sample_fm(self, x, eps, t, cond=None):
        """Calculates the Flow Matching loss for a single sample."""
        # Ensure all tensors are on the correct device
        x, eps, t = x.to(self.device), eps.to(self.device), t.to(self.device)
        t = t.view(-1, 1) if t.dim() == 1 else t

        # Get schedule values from interpolant
        alpha_t_val = self.interpolant.alpha_t(t)
        beta_t_val = self.interpolant.beta_t(t)
        d_alpha_t_val = self.interpolant.d_alpha_t(t)
        d_beta_t_val = self.interpolant.d_beta_t(t)

        # Calculate interpolated path x_t and velocity v_t
        x_t = alpha_t_val * eps + beta_t_val * x
        v_t = d_alpha_t_val * eps + d_beta_t_val * x

        # Get model output
        out = self.u_theta(x_t, t, cond.to(self.device) if cond is not None else None)

        # Return MSE loss components for FM
        return 0.5 * torch.sum(out.square()) - torch.sum(out * v_t)

    def _loss_per_sample_sm(self, x, eps, t, cond=None):
        """Calculates the Score Matching loss for a single sample."""
        x, eps, t = x.to(self.device), eps.to(self.device), t.to(self.device)
        t = t.view(-1, 1) if t.dim() == 1 else t
        
        alpha_t_val = self.interpolant.alpha_t(t)
        beta_t_val = self.interpolant.beta_t(t)

        # Symmetric term
        x_t_pos = alpha_t_val * eps + beta_t_val * x
        out_pos = self.s_theta(x_t_pos, t, cond.to(self.device) if cond is not None else None)
        loss = 0.5 * torch.sum(out_pos.square()) + (1 / alpha_t_val) * torch.sum(out_pos * eps)

        # Antisymmetric term
        eps_neg = -eps
        x_t_neg = alpha_t_val * eps_neg + beta_t_val * x
        out_neg = self.s_theta(x_t_neg, t, cond.to(self.device) if cond is not None else None)
        loss += 0.5 * torch.sum(out_neg.square()) + (1 / alpha_t_val) * torch.sum(out_neg * eps_neg)
        
        return loss

    def train(self, sample_fn, n_iters=2000, batch_size=1024, lr=2e-3, clip=1.0, plot_cond_val=None, log_every=250):
        """
        Main training loop for the FM and SM models.

        Args:
            sample_fn (callable): Function that returns a batch of data (x, cond).
            n_iters (int): Number of training iterations.
            batch_size (int): The size of each batch.
            lr (float): Learning rate.
            clip (float): Gradient clipping value.
            plot_cond_val: Conditional value to use for plotting.
            log_every (int): How often to log progress and plot samples.
        """
        opt_u = AdamW(self.u_theta.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
        sched_u = CosineAnnealingLR(opt_u, T_max=n_iters, eta_min=1e-5)

        opt_s = AdamW(self.s_theta.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
        sched_s = CosineAnnealingLR(opt_s, T_max=n_iters, eta_min=1e-5)

        # --- FIX: Determine vmap configuration before the loop ---
        # We sample one batch to check if the data loader is conditional.
        # This allows us to configure vmap correctly and efficiently once.
        _, cond_sample = sample_fn(1)
        is_conditional = cond_sample is not None
        cond_in_dim = 0 if is_conditional else None

        # Vectorize loss functions with the correct in_dims for the `cond` argument.
        batch_loss_fm = vmap(self._loss_per_sample_fm, in_dims=(0, 0, 0, cond_in_dim), randomness='different')
        batch_loss_sm = vmap(self._loss_per_sample_sm, in_dims=(0, 0, 0, cond_in_dim), randomness='different')

        start_t = time.time()
        for step in range(1, n_iters + 1):
            opt_u.zero_grad()
            opt_s.zero_grad()

            x, cond = sample_fn(batch_size)
            eps = torch.randn_like(x)
            t = torch.rand(size=(batch_size,), device=self.device)

            loss_u = batch_loss_fm(x, eps, t, cond).mean()
            loss_s = batch_loss_sm(x, eps, t, cond).mean()
            loss = loss_u + loss_s
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.u_theta.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(self.s_theta.parameters(), clip)

            opt_u.step()
            opt_s.step()
            sched_u.step()
            sched_s.step()

            self.loss_history.append(loss.item())

            if step % log_every == 0 or step == 1:
                dt = time.time() - start_t
                print(f"[{step:>6}/{n_iters}]  loss={loss.item():.4e}  ({dt/step:.3f}s/it)  lr={sched_u.get_last_lr()[0]:.2e}")
                self.plot_state(sample_fn, self.loss_history, step, suptitle=True,
                                n_step=100, noise_level=0.5, show_samples=5000,
                                cond_val=plot_cond_val)
                plt.close('all')

    @torch.no_grad()
    def sample(self, num_samples, n_step=10, noise_level=1.0, cond_val=None):
        """
        Generate samples from the learned models.

        Args:
            num_samples (int): Number of samples to generate.
            n_step (int): Number of steps in the sampling process.
            noise_level (float): The level of noise for the diffusion term.
            cond_val: Conditional value for sampling.

        Returns:
            torch.Tensor: The generated samples.
        """
        z_dim = self.u_theta.z_dim
        z = torch.randn(num_samples, z_dim, device=self.device)
        t_vals = torch.linspace(0.0, 1.0, n_step + 1, device=self.device)
        sigma = torch.tensor(noise_level, device=self.device)

        cond = torch.full((num_samples, self.u_theta.cond_dim), fill_value=cond_val, device=self.device) if cond_val is not None else None

        for i in range(n_step):
            t = t_vals[i].expand(num_samples)
            dt = t_vals[i + 1] - t_vals[i]
            noise = torch.randn_like(z)

            # v = self.u_theta(z, t, cond)
            # s = self.s_theta(z, t, cond)
            v = self.u_theta(z, t, cond).to(device) if cond is not None else self.u_theta(z, t).to(device)
            s = self.s_theta(z, t, cond).to(device) if cond is not None else self.s_theta(z, t).to(device)

            drift = v + sigma * s
            diffusion = torch.sqrt(2 * sigma * dt)

            z = z + drift * dt + diffusion * noise
        
        return z

    @torch.no_grad()
    def plot_state(self, data_fn, history, step, show_samples=4096, suptitle=True, n_step=1, noise_level=1.0, cond_val=None):
        """Plots the current state of training (loss curve and sample comparison)."""
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss curve
        axes[0].plot(history, lw=2)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Iteration")
        axes[0].grid(True)

        # Plot real vs. generated samples
        real_data, _ = data_fn(show_samples, cond_val=cond_val)
        real = real_data.cpu()
        fake = self.sample(show_samples, n_step=n_step, noise_level=noise_level, cond_val=cond_val).cpu()

        if real.shape[1] == 1: # 1D Data
            axes[1].hist(real.numpy(), bins=50, alpha=0.5, label="real", density=True)
            axes[1].hist(fake.numpy(), bins=50, alpha=0.5, label="fake", density=True)
            axes[1].set_xlabel("Value")
            axes[1].set_xlim(-5, 5)
            axes[1].set_ylim(0, 1)
        else: # 2D Data
            axes[1].scatter(real[:, 0], real[:, 1], s=5, alpha=0.3, label="real")
            axes[1].scatter(fake[:, 0], fake[:, 1], s=5, alpha=0.3, label="fake")
            axes[1].set_aspect("equal", "box")
            axes[1].set_xlim(-5, 5)
            axes[1].set_ylim(-5, 5)
        
        axes[1].legend()
        axes[1].set_title(f"Samples @ Step {step}")

        if suptitle:
            fig.suptitle("Flow Matcher Training Visualization", fontsize=14)

        plt.show()


def plot_path_trajectories(sample_history, n_frame=4, resample_history=None, divergence_points=None, hard_lim=None, experiment_id="VISUALS", name="gmm_path", deg=-60, num_trajectory_points=20, method_figure=False, interval_d=30):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import torch

    # ===================================================================
    # 1. User-specified parameters & Setup
    # ===================================================================
    n_steps = len(sample_history)
    n_samples = sample_history[0].shape[0]
    
    dp_min_step, dp_max_step = None, None
    if divergence_points is not None:
        dp_min, dp_max = divergence_points[0], divergence_points[-1]
        dp_min_step = int(dp_min * (n_steps - 1))
        dp_max_step = int(dp_max * (n_steps - 1))

    # uniformly space n_frame fractions between 0 and 1, excluding the divergence region
    snapshot_fractions = np.linspace(0, 1, n_frame).tolist()
    if divergence_points is not None:
        snapshot_fractions = [s for s in snapshot_fractions if s < dp_min or s > dp_max]
        snapshot_fractions.extend(divergence_points)
            
    selected_steps = [int(f * (n_steps - 1)) for f in snapshot_fractions]
    
    if resample_history is not None:
        selected_steps.extend(resample_history)
    selected_steps = sorted(list(set(selected_steps)))

    if method_figure:
        selected_steps = resample_history
        before_resample = [t-interval_d for t in selected_steps]
        # after_resample = [t+20 for t in selected_steps]
        selected_steps = before_resample + selected_steps + [0, n_steps-1] #+ after_resample
        selected_steps = sorted(list(set(selected_steps)))

    n_trajectories = num_trajectory_points
    trajectory_indices = np.random.choice(n_samples, size=n_trajectories, replace=False)

    # ===================================================================
    # 2. Plotting Setup
    # ===================================================================
    all_points = np.concatenate([sample_history[s].cpu().numpy() for s in selected_steps])
    ymin, ymax = -5, 5
    zmin, zmax = -5, 5
    ymin = min(ymin, all_points[:, 0].min())
    ymax = max(ymax, all_points[:, 0].max())
    zmin = min(zmin, all_points[:, 1].min())
    zmax = max(zmax, all_points[:, 1].max())

    if hard_lim is not None:
        ymin, ymax, zmin, zmax = -hard_lim, hard_lim, -hard_lim, hard_lim

    fig = plt.figure(figsize=(10,10), dpi=160)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    # ===================================================================
    # **MODIFICATION**: Add transparent planes at divergence points
    # ===================================================================
    if divergence_points is not None:
        # Create a grid in the Y-Z plane
        scale_factor = 2  # Scale factor to extend planes beyond data limits
        y_plane = np.linspace(scale_factor * ymin, scale_factor * ymax, 2)
        z_plane = np.linspace(scale_factor * zmin, scale_factor * zmax, 2)
        Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)

        # Plot the first plane at dp_min_step
        X_plane1 = np.full_like(Y_plane, dp_min_step)
        ax.plot_surface(X_plane1, Y_plane, Z_plane, color='orange', alpha=0.2, rstride=1, cstride=1, linewidth=0)

        # Plot the second plane at dp_max_step
        X_plane2 = np.full_like(Y_plane, dp_max_step)
        ax.plot_surface(X_plane2, Y_plane, Z_plane, color='green', alpha=0.2, rstride=1, cstride=1, linewidth=0)
    # ===================================================================


    # ===================================================================
    # 3. Visualization with Z-Ordering and Gap Handling
    # ===================================================================
    
    zorder_base = 0 

    for i in range(len(selected_steps)):
        current_step = selected_steps[i]

        if i > 0:
            prev_step = selected_steps[i-1]
            if divergence_points and prev_step <= dp_min_step and current_step >= dp_max_step:
                pass 
            else:
                for j in trajectory_indices:
                    time_segment = np.arange(prev_step, current_step + 1)
                    point_segment = np.array([sample_history[t][j].cpu().numpy() for t in time_segment])
                    ax.plot(
                        time_segment, point_segment[:, 0], point_segment[:, 1],
                        color='black', linewidth=0.5, alpha=1, zorder=zorder_base
                    )
        
        pts = sample_history[current_step].cpu().numpy()
        if method_figure and current_step in before_resample:
            ax.scatter(
                np.ones(n_samples) * current_step, pts[:, 0], pts[:, 1],
                s=1, alpha=0.5, color='tab:green', depthshade=False, zorder=zorder_base + 1
            )
        
        else: 
            ax.scatter(
                np.ones(n_samples) * current_step, pts[:, 0], pts[:, 1],
                s=1, alpha=0.5, color='tab:blue', depthshade=False, zorder=zorder_base + 1
            )

        if resample_history and current_step in resample_history:
            ax.scatter(
                np.ones(n_samples) * current_step, pts[:, 0], pts[:, 1],
                s=1, alpha=0.5, color='orange', depthshade=False, zorder=zorder_base + 2
            )
        
        zorder_base += 3

    for j in trajectory_indices:
        traj_points = np.array([sample_history[s][j].cpu().numpy() for s in selected_steps])
        ax.scatter(
            selected_steps, traj_points[:, 0], traj_points[:, 1],
            s=1, color='black', linewidths=1, alpha=0.5, depthshade=False, zorder=zorder_base
        )
    
    # [The rest of your styling code remains unchanged...]
    # ===================================================================
    # 4. Final Plot Styling
    # ===================================================================
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((0.4 * len(selected_steps), 1, 1))
    y_floor = 0
    z_floor = ax.get_zlim()[0]
    y_tick_size = (ymax - ymin) * 0.5
    n_ground_lines = 10
    x_positions = np.linspace(0, n_steps - 1, n_ground_lines)
    for x_pos in x_positions:
        ax.plot([x_pos, x_pos], [y_floor - y_tick_size, y_floor + y_tick_size], [z_floor, z_floor],
                color="black", linewidth=0.7, alpha=0.9)
    label_offset = (ymax - ymin) * 0.3
    ax.text(-label_offset, y_floor - label_offset, z_floor - 0.3 * label_offset, r"$p_0$", fontsize=10, ha='center', va='top')
    ax.text(n_steps - 1 - label_offset, y_floor - label_offset, z_floor - 0.3 * label_offset, r"$p_1$", fontsize=10, ha='center', va='top')
    ax.set_xlabel("time $t$")
    ax.set_xticks([])
    ax.set_ylabel(""); ax.set_yticks([])
    ax.set_zlabel(""); ax.set_zticks([])
    ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
    ax.grid(False)
    # make all axis colors white
    ax.xaxis.line.set_color((1,1,1,0))
    # ax.yaxis.line.set_color((1,1,1,0))
    ax.zaxis.line.set_color((1,1,1,0))
    ax.view_init(elev=9, azim=deg)
    plt.tight_layout()
    plt.savefig(f"{experiment_id}/trajectory_plot_{name}.png", dpi=300)
    plt.show()

# --- Example Usage (Conceptual) ---

# if __name__ == '__main__':
#     # This block is for demonstration and won't run without the full notebook context
#     # (e.g., model definitions, data loaders).
    
#     # 1. Define your models (u_theta, s_theta)
#     # class SimpleNet(nn.Module):
#     #     ...
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # u_theta = SimpleNet(z_dim=2, cond_dim=1).to(device)
#     # s_theta = SimpleNet(z_dim=2, cond_dim=1).to(device)

#     # 2. Define your data sampling function
#     # def sample_data_fn(batch_size, cond_val=None):
#     #     ...
#     #     return x, cond

#     # --- USAGE SCENARIO 1: Default Linear Interpolant ---
#     print("--- Using Default Linear Interpolant ---")
#     # default_interpolant = Interpolant() # Or just pass None
#     # fm_trainer_default = FlowMatcher(u_theta, s_theta, interpolant=default_interpolant)
#     # fm_trainer_default.train(sample_data_fn, n_iters=10) # Using few iterations for demo
    
#     # --- USAGE SCENARIO 2: Custom Non-Linear Interpolant ---
#     print("\n--- Using Custom Non-Linear Interpolant ---")
    
#     # Example from our discussions: alpha_t = (1-t)(1-2t)^2
#     def custom_alpha_t(t):
#         return (1.0 - t) * torch.pow(1.0 - 2.0 * t, 2)
        
#     def custom_d_alpha_t(t):
#         # Derivative of (1-t)(1-4t+4t^2) = 1-5t+8t^2-4t^3 is -5+16t-12t^2
#         return -5.0 + 16.0 * t - 12.0 * t**2

#     # Beta can remain the same
#     def custom_beta_t(t):
#         return t

#     def custom_d_beta_t(t):
#         return torch.ones_like(t)

#     # Create the custom interpolant instance
#     custom_interpolant = Interpolant(
#         alpha_t=custom_alpha_t,
#         beta_t=custom_beta_t,
#         d_alpha_t=custom_d_alpha_t,
#         d_beta_t=custom_d_beta_t
#     )
    
#     # Initialize the trainer with the custom interpolant
#     # fm_trainer_custom = FlowMatcher(u_theta, s_theta, interpolant=custom_interpolant)
    
#     # And run training
#     # fm_trainer_custom.train(sample_data_fn, n_iters=10)
    
#     print("\nCode has been refactored. The example usage demonstrates how to")
#     print("initialize and train with both default and custom schedules.")











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