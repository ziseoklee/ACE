# interpolants.py
import torch
import torch.nn.functional as F

# -------------------------
# Default linear
# -------------------------
def alpha_default(t): return 1 - t
def beta_default(t): return t
def d_alpha_default(t): return -1
def d_beta_default(t): return 1

# -------------------------
# Cos(t) interpolant
# -------------------------
def alpha_cos_t(t): return torch.cos(torch.pi * 0.5 * t)
def beta_cos_t(t): return torch.sin(torch.pi * 0.5 * t)
def d_alpha_cos_t(t): return -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t)
def d_beta_cos_t(t): return 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t)

# -------------------------
# Cosine schedule (small s bump)
# -------------------------
s_cosine = torch.tensor(0.008)
denom_cosine = torch.cos(s_cosine/(1+s_cosine)*torch.pi/2)**2
def abar_cosine(t): return (torch.cos((t+s_cosine)/(1+s_cosine)*torch.pi/2)**2) / denom_cosine
def dabar_cosine(t): return -torch.sin((t+s_cosine)/(1+s_cosine)*torch.pi/2) * torch.cos((t+s_cosine)/(1+s_cosine)*torch.pi/2) * (torch.pi/(1+s_cosine)) / denom_cosine
def alpha_cosine(t): return torch.sqrt(abar_cosine(t))
def beta_cosine(t): return torch.sqrt(1 - abar_cosine(t))
def d_alpha_cosine(t): return 0.5 * dabar_cosine(t) / (torch.sqrt(abar_cosine(t)) + 1e-8)
def d_beta_cosine(t): return -0.5 * dabar_cosine(t) / (torch.sqrt(1 - abar_cosine(t)) + 1e-8)

# -------------------------
# DDPM linear
# -------------------------
beta0, beta1 = torch.tensor(0.1), torch.tensor(20.0)
def I_ddpm(t): return beta0 * t + 0.5 * (beta1 - beta0) * t**2
def dI_ddpm(t): return beta0 + (beta1 - beta0) * t
def alpha_ddpm_linear(t): return torch.sqrt(torch.exp(-0.5 * I_ddpm(t)))
def beta_ddpm_linear(t): return torch.sqrt(1 - torch.exp(-0.5 * I_ddpm(t)))
def d_alpha_ddpm_linear(t): return -0.25 * dI_ddpm(t) * torch.exp(-0.5 * I_ddpm(t)) / (torch.sqrt(torch.exp(-0.5 * I_ddpm(t))) + 1e-8)
def d_beta_ddpm_linear(t): return 0.25 * dI_ddpm(t) * torch.exp(-0.5 * I_ddpm(t)) / (torch.sqrt(1 - torch.exp(-0.5 * I_ddpm(t))) + 1e-8)

# -------------------------
# One minus t squared
# -------------------------
def alpha_one_minus_t_squared(t): return 1 - t**2
def beta_one_minus_t_squared(t): return t
def d_alpha_one_minus_t_squared(t): return -2 * t
def d_beta_one_minus_t_squared(t): return torch.ones_like(t)

# -------------------------
# Denovo
# -------------------------
def alpha_denovo(t): return 1 - t**2
def beta_denovo(t): return torch.sqrt(1 - (1 - t**2)**2)
def d_alpha_denovo(t): return -2 * t
def d_beta_denovo(t): return (2*t*(1 - t**2)) / torch.sqrt(2*t**2 - t**4 + 1e-8)

# -------------------------
# Sigmoid-based
# -------------------------
def sigmoid_schedule(t): return torch.sqrt(1 - torch.exp(-((20/12) * F.softplus((t-0.5)*12) + 0.001*t)))
def dsigmoid_schedule(t):
    exp_term = torch.exp(-((20/12)*F.softplus((t-0.5)*12) + 0.001*t))
    return (exp_term * (20*torch.sigmoid((t-0.5)*12) + 0.001)) / (2 * torch.sqrt(1 - exp_term) + 1e-12)
def alpha_sigmoid(t): return sigmoid_schedule(1 - t)
def beta_sigmoid(t): return t
def d_alpha_sigmoid(t): return -dsigmoid_schedule(1 - t)
def d_beta_sigmoid(t): return torch.ones_like(t)

# -------------------------
# Exponential
# -------------------------
def alpha_exponential(t): return torch.exp(-5.0 * t)
def beta_exponential(t): return torch.sqrt(1 - torch.exp(-2 * 5.0 * t))
def d_alpha_exponential(t): return -5.0 * torch.exp(-5.0 * t)
def d_beta_exponential(t): return 5.0 * torch.exp(-2 * 5.0 * t)/(torch.sqrt(1 - torch.exp(-2 * 5.0 * t)) + 1e-8)

# -------------------------
# Variance exploding
# -------------------------
def alpha_ve(t): return torch.ones_like(t)
def beta_ve(t): return 0.01 * ((50.0/0.01) ** t)
def d_alpha_ve(t): return torch.zeros_like(t)
def d_beta_ve(t): return 0.01 * ((50.0/0.01) ** t) * torch.log(torch.tensor(50.0/0.01))

# -------------------------
# Custom polynomial
# -------------------------
def alpha_custom_poly(t): return (1.0 - t) * torch.pow(t + (1.0 - 2 * t), 2)
def beta_custom_poly(t): return t
def d_alpha_custom_poly(t): return -4.0 + 14.0 * t - 12.0 * t**2
def d_beta_custom_poly(t): return torch.ones_like(t)

# -------------------------
# Dictionary of all interpolants
# -------------------------
INTERPOLANTS = {
    "default_linear": (alpha_default, beta_default, d_alpha_default, d_beta_default),
    "cos_t": (alpha_cos_t, beta_cos_t, d_alpha_cos_t, d_beta_cos_t),
    "cosine": (alpha_cosine, beta_cosine, d_alpha_cosine, d_beta_cosine),
    "ddpm_linear": (alpha_ddpm_linear, beta_ddpm_linear, d_alpha_ddpm_linear, d_beta_ddpm_linear),
    "one_minus_t_squared": (alpha_one_minus_t_squared, beta_one_minus_t_squared, d_alpha_one_minus_t_squared, d_beta_one_minus_t_squared),
    "denovo": (alpha_denovo, beta_denovo, d_alpha_denovo, d_beta_denovo),
    "sigmoid": (alpha_sigmoid, beta_sigmoid, d_alpha_sigmoid, d_beta_sigmoid),
    "exponential": (alpha_exponential, beta_exponential, d_alpha_exponential, d_beta_exponential),
    "variance_exploding": (alpha_ve, beta_ve, d_alpha_ve, d_beta_ve),
    "custom_poly": (alpha_custom_poly, beta_custom_poly, d_alpha_custom_poly, d_beta_custom_poly),
}