import torch
import matplotlib.pyplot as plt
import numpy as np
import random, os
from datetime import datetime
import os
import torch
import numpy as np
import pandas as pd
from utils.metrics.export import compute_sample_based_metrics
from hcg.sample_data import plot_diagnostics
from hcg.hcg import simulate_hcg_generalized
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); random.seed(0); np.random.seed(0)
print("device →", device)

### Run this ONCE per experiment
new_experiment = True  # Set to False to reuse an old experiment ID
if new_experiment:  # Run the experiment with the current date and time as the experiment ID
    experiment_id = f"experiment_gmm_{datetime.now().strftime('%Y%m%d')}" #_%H%M%S
    os.makedirs(experiment_id, exist_ok=True)
else:
    experiment_id = "experiment_gmm_20250901"  # Use a fixed ID for reproducibility

bs = 10000


import numpy as np
import torch
from scipy.stats import multivariate_normal, norm

def ground_truth_gmm_2d(bs):
    n_components = 5
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    means = np.array([[0, 0], [3, 3], [-3, -3], [3, -3], [-3, 3]])
    covariances = np.array([[[0.5, 0], [0, 0.5]]] * n_components)
    
    samples = []
    while len(samples) < bs:
        component = np.random.choice(n_components, p=weights)
        sample = np.random.multivariate_normal(means[component], covariances[component])
        samples.append(sample)
    
    return torch.tensor(samples, dtype=torch.float32)


def p_x(x):
    """
    Marginal distribution p(x) from the 2D GMM, 
    i.e. integrate out y: p(x) = ∫ p(x,y) dy
    """
    n_components = 5
    weights = np.ones(n_components) / n_components
    means = np.array([[0, 0], [3, 3], [-3, -3], [3, -3], [-3, 3]])
    covariances = np.array([[[0.5, 0], [0, 0.5]]] * n_components)
    
    px = np.zeros(len(x))
    for w, m, c in zip(weights, means, covariances):
        # marginalize over y: the marginal of a Gaussian is still Gaussian
        mean_x = m[0]
        var_x = c[0,0]
        px += w * norm.pdf(x, loc=mean_x, scale=np.sqrt(var_x))
    return px


def conditional_gmm(bs):
    # draw from joint p(x,y)
    xy = ground_truth_gmm_2d(5*bs).numpy()
    x = xy[:, 0]
    y = xy[:, 1]
    
    
    weights_A = np.array([0.1, 0.9])
    means_A = np.array([-3, 3])
    std_A = np.array([0.5, 0.5])
    
    px_given_A = np.zeros_like(x)
    for w, m, s in zip(weights_A, means_A, std_A):
        px_given_A += w * norm.pdf(x, loc=m, scale=s)
    
    px = p_x(x)
    
    # importance weights
    imp_w = px_given_A / (px + 1e-8)
    imp_w = imp_w / np.sum(imp_w)  # normalize
    
    # resample according to weights
    idx = np.random.choice(len(x), size=5*bs, p=imp_w)
    samples = xy[idx]
    
    return torch.tensor(samples[:bs], dtype=torch.float32)

def conditional_gmm_p1p2(bs):
    # draw from joint p(x,y)
    xy = ground_truth_gmm_2d(5*bs).numpy()
    x = xy[:, 0]
    y = xy[:, 1]
    
    
    weights_A = np.array([0.1, 0.9])
    means_A = np.array([-3, 3])
    std_A = np.array([0.5, 0.5])
    
    px_given_A = np.zeros_like(x)
    for w, m, s in zip(weights_A, means_A, std_A):
        px_given_A += w * norm.pdf(x, loc=m, scale=s)
    
    # importance weights
    imp_w = px_given_A
    imp_w = imp_w / np.sum(imp_w)  # normalize
    
    # resample according to weights
    idx = np.random.choice(len(x), size=5*bs, p=imp_w)
    samples = xy[idx]
    
    return torch.tensor(samples[:bs], dtype=torch.float32)

samples = conditional_gmm(bs).cpu().numpy()
print(samples.shape)
plot_diagnostics(samples, torch.zeros(bs), [torch.zeros(bs)], save_name=f"{experiment_id}/ground_truth", full=False)

plt.hist(samples[:,0], bins=200, density=True, alpha=0.7)
plt.title("Samples (Projected onto x-axis)")
plt.grid(True)
plt.xlabel("x"); plt.xlim(-5,5)
plt.ylabel("Density")
plt.show()

results = []

import torch
import math

_eps = 1e-12
_TWOPI = 2.0 * math.pi

# -------------------------
# Time schedules (batch-aware)
# -------------------------
def alpha_fn(t):
    # t: [B,1]
    return 1.0 - t          # [B,1]

def beta_fn(t):
    # t: [B,1]
    return 2.0 / (1.0 - t + 1e-2)  # [B,1]

def sigma_fn(t):
    return torch.sqrt(beta_fn(t))    # [B,1]

# -------------------------
# Stable 1D log-normal
# -------------------------
def _log_normal_1d(x, mu, var):
    # x: [B,1] or [B], mu: [B,K], var: [B,K]
    x = x.reshape(-1, 1)           # [B,1]
    log_pref = -0.5 * torch.log(_TWOPI * var)    # [B,K]
    exp_term = -0.5 * ((x - mu)**2) / var        # [B,K]
    return log_pref + exp_term                    # [B,K]

# -------------------------
# 1D time-dependent score for p(x|A)
# -------------------------
def s1_fn(x, t):
    """
    x: [B,1] or [B], t: [B,1]
    returns: score of same shape as x
    """
    device = x.device
    dtype = x.dtype
    B = x.shape[0]

    # GMM parameters for p(x|A)
    pis = torch.tensor([0.1, 0.9], device=device, dtype=dtype)    # [K]
    mus0 = torch.tensor([-3.0, 3.0], device=device, dtype=dtype) # [K]
    sig2_0 = torch.tensor([0.5**2, 0.5**2], device=device, dtype=dtype) # [K]

    K = pis.shape[0]

    # schedule
    alpha = alpha_fn(t)                  # [B,1]
    s2 = 1.0 - alpha**2                  # [B,1]

    # batch-aware component params: mu_t [B,K], tau2 [B,K]
    mu_t = alpha * mus0.reshape(1,K)     # [B,K]
    tau2 = alpha**2 * sig2_0.reshape(1,K) + s2  # [B,K]

    # log components [B,K]
    x_in = x.reshape(B,1)                # [B,1]
    log_comp = torch.log(pis.reshape(1,K)) + _log_normal_1d(x_in, mu_t, tau2)  # [B,K]

    # responsibilities
    logp = torch.logsumexp(log_comp, dim=1, keepdim=True)   # [B,1]
    r = torch.exp(log_comp - logp)                           # [B,K]

    # component scores
    comp_scores = (mu_t - x_in) / (tau2 + _eps)  # [B,K]
    score = (r * comp_scores).sum(dim=1, keepdim=True)  # [B,1]

    if x.ndim == 2 and x.shape[1] == 1:
        return score
    else:
        return score.reshape(-1)  # [B]

# -------------------------
# 1D time-dependent score for p_X (1D marginal)
# -------------------------
def s2_fn(x, t):
    """
    x: [B,1] or [B], t: [B,1]
    returns: score [B,1]
    """
    device = x.device
    dtype = x.dtype
    B = x.shape[0]

    pis = torch.tensor([0.4,0.2,0.4], device=device, dtype=dtype)  # [K]
    mus0 = torch.tensor([-3.0,0.0,3.0], device=device, dtype=dtype)
    sig2_0 = torch.tensor([0.5**2]*3, device=device, dtype=dtype)

    K = pis.shape[0]

    alpha = alpha_fn(t)                 # [B,1]
    s2 = 1.0 - alpha**2                 # [B,1]

    mu_t = alpha * mus0.reshape(1,K)          # [B,K]
    tau2 = alpha**2 * sig2_0.reshape(1,K) + s2  # [B,K]

    x_in = x.reshape(B,1)                   # [B,1]
    log_comp = torch.log(pis.reshape(1,K)) + _log_normal_1d(x_in, mu_t, tau2)  # [B,K]

    logp = torch.logsumexp(log_comp, dim=1, keepdim=True)   # [B,1]
    r = torch.exp(log_comp - logp)                           # [B,K]

    comp_scores = (mu_t - x_in) / (tau2 + _eps)  # [B,K]
    score = (r * comp_scores).sum(dim=1, keepdim=True)  # [B,1]

    if x.ndim == 2 and x.shape[1] == 1:
        return score
    else:
        return score.reshape(-1)
    
    
# -------------------------
# Stable isotropic multivariate log-normal
# -------------------------
def _log_normal_isotropic(x, mus, tau2):
    # x: [B,D], mus: [B,K,D], tau2: [B,K]
    diff2 = ((x.unsqueeze(1) - mus)**2).sum(-1)   # [B,K]
    D = x.shape[1]
    log_pref = -0.5 * D * torch.log(_TWOPI * tau2)   # [B,K]
    exp_term = -0.5 * diff2 / tau2                    # [B,K]
    return log_pref + exp_term                        # [B,K]

# -------------------------
# 2D time-dependent score for p_X (2D GMM)
# -------------------------
def s3_fn(x, t):
    """
    x: [B,2], t: [B,1]
    returns: score [B,2]
    """
    device = x.device
    dtype = x.dtype
    B = x.shape[0]
    K = 5

    pis = torch.ones(K, device=device, dtype=dtype) / K     # [K]
    mus0 = torch.tensor([[0.,0.],[3.,3.],[-3.,-3.],[3.,-3.],[-3.,3.]], device=device, dtype=dtype)  # [K,2]
    sig2_0 = torch.tensor([0.5]*K, device=device, dtype=dtype)  # [K]

    # schedule
    alpha = alpha_fn(t)                 # [B,1]
    s2 = 1.0 - alpha**2                 # [B,1]

    # batch-aware component params: mu_t [B,K,2], tau2 [B,K]
    mu_t = alpha.unsqueeze(1) * mus0.unsqueeze(0)   # [B,K,2]
    tau2 = alpha**2 * sig2_0.reshape(1,K) + s2      # [B,K]

    # log component densities [B,K]
    x_exp = x.unsqueeze(1)             # [B,1,2]
    log_comp = torch.log(pis.reshape(1,K)) + _log_normal_isotropic(x, mu_t, tau2)  # [B,K]

    # responsibilities
    logp = torch.logsumexp(log_comp, dim=1, keepdim=True)   # [B,1]
    r = torch.exp(log_comp - logp)                           # [B,K]

    # component scores [B,K,2]
    comp_scores = (mu_t - x_exp) / (tau2.unsqueeze(-1) + _eps)  # [B,K,2]
    score = (r.unsqueeze(-1) * comp_scores).sum(dim=1)          # [B,2]
    return score



# -------------------------
# 1D drift for p(x|A)
# -------------------------
def v1_fn(x, t):
    """
    x: [B,1]
    t: [B,1]
    returns: drift [B,1] (same shape as x)
    """
    beta_t = beta_fn(1 - t)           # [B,1]
    score = s1_fn(x, 1 - t)           # [B,1]
    drift = 0.5 * beta_t * x + 0.5 * beta_t * score   # [B,1], broadcasting works
    return drift

# -------------------------
# 2D drift for p_X (2D GMM)
# -------------------------
def v2_fn(x, t):
    """
    x: [B,1]
    t: [B,1]
    returns: drift [B,1]
    """
    beta_t = beta_fn(1 - t)           # [B,1]
    score = s2_fn(x, 1 - t)           # [B,2]
    drift = 0.5 * beta_t * x + 0.5 * beta_t * score  # [B,2], broadcasting works
    return drift

# -------------------------
# 1D drift for p_X (1D marginal)
# -------------------------
def v3_fn(x, t):
    """
    x: [B,2]
    t: [B,1]
    returns: drift [B,2]
    """
    beta_t = beta_fn(1 - t)           # [B,1]
    score = s3_fn(x, 1 - t)           # [B,1]
    drift = 0.5 * beta_t * x + 0.5 * beta_t * score  # [B,1]
    return drift

print("Models Defined.")
    
for i in tqdm([0,1,2,3,4]):
    torch.manual_seed(i); random.seed(i); np.random.seed(i)
    
    print("Running HCG p1p3/p2")
    x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t), # v1(X|A)
            lambda x, t: v2_fn(x[:, :1], t),                                                 # v2(X)
            lambda x, t: v3_fn(x, t)         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], 1-t), # s1(X|A)
            lambda x, t: s2_fn(x[:, :1], 1-t),                                                 # s2(X)
            lambda x, t: s3_fn(x, 1-t)         # s3(Z|B)
        ],
        gamma_list=[1, -1, 1],     # from log-weight increment in simulate_hcg
        emb_list=[
            lambda z: z[:, :1],    # project to X
            lambda z: z[:, :1],    # project to X
            lambda z: z            # identity for Z
        ],
        trans_emb_list=[
            lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
            lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
            lambda z: z  # identity
        ],
        sigma_fn=sigma_fn,
        #lambda z, t: torch.cat([v1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), torch.zeros(z.size(0), 1, device=z.device)], dim=1), # v_star = v1
        v_star= lambda z, t: v3_fn(z, t), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=True,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/hcg-1")

    samples_gt = conditional_gmm(bs).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["HCG[1,-1,1]", w1, w2, mmd_rbf, total_var])



    print("Running FKC p1p3/p2")
    x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample

    # choose f(y) = N(0, 1)  => score s_f(y) = -y
    def s_fp(y):           # y has shape (bs,1)
        return -y
    def s_fn(y):           # y has shape (bs,1)
        return -y # -0.5*y
    def v_fp(y, t):        # choose zero deterministic drift in y
        return torch.zeros_like(y)
    def v_fn(y, t):        # choose zero deterministic drift in y
        return torch.zeros_like(y)

    # build v_fn_list and s_fn_list that return 2D vectors for (x,y)
    v_fn_list = [
        lambda z, t: torch.cat([ v1_fn(z[:, :1], t), v_fp(z[:, 1:2], t) ], dim=1),
        lambda z, t: torch.cat([ v2_fn(z[:, :1], t), v_fn(z[:, 1:2], t) ], dim=1),
        lambda z, t: v3_fn(z, t)   # already 2D
    ]
    s_fn_list = [
        lambda z, t: torch.cat([ s1_fn(z[:, :1], 1-t), s_fp(z[:, 1:2]) ], dim=1),
        lambda z, t: torch.cat([ s2_fn(z[:, :1], 1-t), s_fn(z[:, 1:2]) ], dim=1),
        lambda z, t: s3_fn(z, 1-t)   # already 2D
    ]
    # pass identity projections since all are now 2D
    emb_list = [lambda z: z, lambda z: z, lambda z: z]
    trans_emb_list = [lambda z: z, lambda z: z, lambda z: z]
    # v_star: extend original v1 in x and zero in y
    v_star = lambda z, t: v3_fn(z, t)
    #lambda z, t: torch.cat([ v1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), torch.zeros(z.size(0), 1, device=z.device) ], dim=1)


    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[2, 2, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=v_fn_list,
        s_fn_list=s_fn_list,
        gamma_list=[1, -1, 1],     # from log-weight increment in simulate_hcg
        emb_list=emb_list,
        trans_emb_list=trans_emb_list,
        sigma_fn=sigma_fn,
        v_star=v_star,
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=True,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/fkc-1")


    samples_gt = conditional_gmm(bs).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["FKC[1,-1,1]", w1, w2, mmd_rbf, total_var])




    # Product of Experts

    print("Running HCG p1p2")
    x0 = (1/torch.sqrt(torch.tensor(2))) * torch.randn(bs, 2).to("cuda")  # (X, Y) sample

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t), # v1(X|A)
            lambda x, t: v3_fn(x, t)         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], 1-t), # s1(X|A)
            lambda x, t: s3_fn(x, 1-t)         # s3(Z|B)
        ],
        gamma_list=[1, 1],     # from log-weight increment in simulate_hcg
        emb_list=[
            lambda z: z[:, :1],    # project to X
            lambda z: z            # identity for Z
        ],
        trans_emb_list=[
            lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
            lambda z: z  # identity
        ],
        sigma_fn=sigma_fn,
        #lambda z, t: torch.cat([v1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), torch.zeros(z.size(0), 1, device=z.device)], dim=1), # v_star = v1
        v_star= lambda z, t: v3_fn(z, t), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=True,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/hcg-2")


    samples_gt = conditional_gmm_p1p2(bs).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["HCG[1,1]", w1, w2, mmd_rbf, total_var])





    print("Running FKC p1p2")
    x0 = (1/torch.sqrt(torch.tensor(2))) * torch.randn(bs, 2).to("cuda")  # (X, Y) sample

    # choose f(y) = N(0, 1)  => score s_f(y) = -y
    def s_f(y):           # y has shape (bs,1)
        return -y
    def v_f(y, t):        # choose zero deterministic drift in y
        return torch.zeros_like(y)

    v_fn_list = [
        lambda z, t: torch.cat([ v1_fn(z[:, :1], t), v_f(z[:, 1:2], t) ], dim=1),
        lambda z, t: v3_fn(z, t)   # already 2D
    ]
    s_fn_list = [
        lambda z, t: torch.cat([ s1_fn(z[:, :1], 1-t), s_f(z[:, 1:2]) ], dim=1),
        lambda z, t: s3_fn(z, 1-t)   # already 2D
    ]
    emb_list = [lambda z: z, lambda z: z]
    trans_emb_list = [lambda z: z, lambda z: z]
    v_star = lambda z, t: v3_fn(z, t)


    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[2, 2],
        v_fn_list=v_fn_list,
        s_fn_list=s_fn_list,
        gamma_list=[1, 1],     # from log-weight increment in simulate_hcg
        emb_list=emb_list,
        trans_emb_list=trans_emb_list,
        sigma_fn=sigma_fn,
        v_star=v_star,
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=True,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/fkc-2")

    samples_gt = conditional_gmm_p1p2(bs).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["FKC[1,1]", w1, w2, mmd_rbf, total_var])



    print("Running Target Score p1p3/p2")
    x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t), # v1(X|A)
            lambda x, t: v2_fn(x[:, :1], t),                                                 # v2(X)
            lambda x, t: v3_fn(x, t)         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], 1-t), # s1(X|A)
            lambda x, t: s2_fn(x[:, :1], 1-t),                                                 # s2(X)
            lambda x, t: s3_fn(x, 1-t)         # s3(Z|B)
        ],
        gamma_list=[1, -1, 1],     # from log-weight increment in simulate_hcg
        emb_list=[
            lambda z: z[:, :1],    # project to X
            lambda z: z[:, :1],    # project to X
            lambda z: z            # identity for Z
        ],
        trans_emb_list=[
            lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
            lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
            lambda z: z  # identity
        ],
        sigma_fn=sigma_fn,
        #lambda z, t: torch.cat([v1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), torch.zeros(z.size(0), 1, device=z.device)], dim=1), # v_star = v1
        v_star= lambda z, t: v3_fn(z, t), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=False,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/heur-1")

    samples_gt = conditional_gmm(bs).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["Heuristic[1,-1,1]", w1, w2, mmd_rbf, total_var])
    
    
    
    print("Running Target Score p1p2")
    x0 = (1/torch.sqrt(torch.tensor(2))) * torch.randn(bs, 2).to("cuda")  # (X, Y) sample

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t), # v1(X|A)
            lambda x, t: v3_fn(x, t)         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], 1-t), # s1(X|A)
            lambda x, t: s3_fn(x, 1-t)         # s3(Z|B)
        ],
        gamma_list=[1, 1],     # from log-weight increment in simulate_hcg
        emb_list=[
            lambda z: z[:, :1],    # project to X
            lambda z: z            # identity for Z
        ],
        trans_emb_list=[
            lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
            lambda z: z  # identity
        ],
        sigma_fn=sigma_fn,
        #lambda z, t: torch.cat([v1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), torch.zeros(z.size(0), 1, device=z.device)], dim=1), # v_star = v1
        v_star= lambda z, t: v3_fn(z, t), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=True,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/heur-2")


    samples_gt = conditional_gmm_p1p2(bs).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["Heuristic[1,1]", w1, w2, mmd_rbf, total_var])
    
    
    
    print(results)
    # save results as pd
    import pandas as pd

    df = pd.DataFrame(results, columns=["Method", "W1", "W2", "MMD", "TV"])
    df.to_csv(f"{experiment_id}/results_gmm.csv", index=False)

import os
from datetime import datetime
import pandas as pd
import numpy as np


exp_type = "gmm"  # Options: "checker", "gmm"

# Load results 
result_csv_path = f"{experiment_id}/results_{exp_type}.csv"
results_csv = pd.read_csv(result_csv_path)
print(f"Results loaded from {result_csv_path}")
stats_by_method = results_csv.groupby('Method').agg({
    'W1': ['mean', 'std'],
    'W2': ['mean', 'std'],
    'MMD': ['mean', 'std'],
    'TV': ['mean', 'std']
}).reset_index()
print(stats_by_method)
# Save stats to a new CSV
stats_csv_path = f"{experiment_id}/stats_summary_{exp_type}.csv"
stats_by_method.to_csv(stats_csv_path, index=False)
print(f"Stats summary saved to {stats_csv_path}")