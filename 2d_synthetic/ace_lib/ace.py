import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def simulate_ace(
    x0: torch.tensor,          # [bs, d]
    v_fn_list: list,            # List of velocity functions v_i(x, t)
    s_fn_list: list,            # List of score functions s_i(x, t)
    gamma_list: list,           # List of gammas (callables gamma_i(t) or constants)
    proj_list: list,            # List of projection functions pi_i(x)
    emb_list: list,             # List of embedding functions iota_i(x)
    sigma_fn: callable,
    v_star: callable,           # v_star(x, t)
    d_gamma_list: list = None,  # Optional: list of gamma derivatives d_gamma_i(t)
    t0=0.0, t1=1.0,
    n_steps=1000,
    device="cuda",
    resample=True,
    ess_threshold=0.5,
    print_resample_history=False,
    t_max = 0.85,
    t_min = 0.00,
    t_list = None
):
    """
    Simulates a heterogeneous particle system using the Adaptive path Correction with Exponents (ACE) method.
    Allows time-dependent exponents gamma_i(t).
    """
    # Use Hutchinson trace estimator for divergence terms
    def divergence(f, x, t):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            e = torch.randn_like(x)
            out = torch.sum(f(x, t) * e)
            grad = torch.autograd.grad(out, x)[0]
            return (grad * e).sum(dim=-1, keepdim=True)

    x = x0.clone().to(device)
    bs = x.size(0)
    logw = torch.zeros(bs, 1, device=device)
    times = torch.linspace(t0, t1, n_steps + 1, device=device)
    dt = torch.tensor((t1 - t0) / n_steps, device=device)

    if not callable(gamma_list[0]):
        gamma_fns = [lambda t, val=g: torch.full_like(t, val) for g in gamma_list]
        d_gamma_fns = [lambda t: torch.zeros_like(t) for _ in gamma_list]
    else:
        gamma_fns = gamma_list
        if d_gamma_list is None:
            raise ValueError("d_gamma_list must be provided for time-dependent gammas")
        d_gamma_fns = d_gamma_list

    # Initialize log q_i based on standard Gaussian density at t=0
    log_q_i_list = []
    for i, proj in enumerate(proj_list):
        x_proj = proj(x)
        d_i = x_proj.shape[-1]
        log_norm_const = -0.5 * d_i * np.log(2 * np.pi)
        log_exp = -0.5 * torch.sum(x_proj**2, dim=-1, keepdim=True)
        log_q_i_list.append(log_norm_const + log_exp)

    logw_history, sample_history, resample_history = [], [], []

    for it in tqdm(range(n_steps)):
        t = times[it].expand(bs, 1)
        sigma_t = sigma_fn(t)
        v_star_t = v_star(x, t)

        gamma_t_vals = [g(t) for g in gamma_fns]
        d_gamma_t_vals = [dg(t) for dg in d_gamma_fns]

        s_star_t = sum(
            gamma_t_vals[i] * emb_list[i](s_fn_list[i](proj_list[i](x), t))
            for i in range(len(gamma_fns))
        )

        drift_t = v_star_t + 0.5 * sigma_t**2 * s_star_t
        noise = torch.randn_like(x) * (sigma_t * torch.sqrt(dt))
        x = x + drift_t * dt + noise

        div_v_star_t = divergence(v_star, x, t)
        
        corrector_terms = []
        for i in range(len(gamma_fns)):
            v_tilde_i = emb_list[i](v_fn_list[i](proj_list[i](x), t))
            s_tilde_i = emb_list[i](s_fn_list[i](proj_list[i](x), t))
            div_v_tilde_i = divergence(lambda _x, _t: emb_list[i](v_fn_list[i](proj_list[i](_x), _t)), x, t)
            dot_product = torch.sum((v_star_t - v_tilde_i) * s_tilde_i, dim=1, keepdim=True)
            corrector_terms.append(-div_v_tilde_i + dot_product)

        # Update each log q_i according to its ODE
        for i in range(len(gamma_fns)):
            log_q_i_list[i] += corrector_terms[i] * dt
            
        # Sum terms for the logw increment
        d_gamma_log_q_sum = sum(d_gamma_t_vals[i] * log_q_i_list[i] for i in range(len(gamma_fns)))
        gamma_corrector_sum = sum(gamma_t_vals[i] * corrector_terms[i] for i in range(len(gamma_fns)))
        
        increment = div_v_star_t + d_gamma_log_q_sum + gamma_corrector_sum
        logw += increment * dt
        
        logw_history.append(logw.clone().cpu())

        if resample:
            if (t_list is None and it < n_steps * t_max and it > n_steps * t_min):
                weights = F.softmax(logw.squeeze(-1), dim=0)
                ess = 1.0 / torch.sum(weights**2)
                if ess < ess_threshold * bs or it == n_steps - 5:
                    resample_history.append(it)
                    idx = torch.multinomial(weights, bs, replacement=True)
                    x = x[idx]
                    logw = torch.zeros_like(logw)
                    log_q_i_list = [log_q[idx] for log_q in log_q_i_list]
            elif (t_list is not None and (it*1.0)/n_steps in t_list):
                weights = F.softmax(logw.squeeze(-1), dim=0)
                resample_history.append(it)
                idx = torch.multinomial(weights, bs, replacement=True)
                x = x[idx]
                logw = torch.zeros_like(logw)
                log_q_i_list = [log_q[idx] for log_q in log_q_i_list]
                
        sample_history.append(x.clone().cpu())

    if print_resample_history:
        return x, logw, logw_history, sample_history, resample_history
    else:
        return x, logw, logw_history, sample_history