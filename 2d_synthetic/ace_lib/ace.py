import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def simulate_ace(
    x0: torch.tensor,          # [bs, d]
    v_fn_list: list,           # List of velocity functions v_i(x, t)
    s_fn_list: list,           # List of score functions s_i(x, t)
    gamma_list: list,          # List of gammas: callables gamma_i(t) or constants
    proj_list: list,           # List of projection functions pi_i(x)
    emb_list: list,            # List of embedding functions iota_i(x)
    sigma_fn: callable,
    v_star: callable,          # v_star(x, t)
    d_gamma_list: list = None, # Optional: list of gamma derivatives d_gamma_i(t)
    t0=0.0,
    t1=1.0,
    n_steps=1000,
    device="cuda",
    resample=True,
    ess_threshold=0.5,
    print_resample_history=False,
    t_max=0.85,
    t_min=0.00,
    t_list=None,
):
    """
    Simulates ACE with time-dependent exponents.

    Correct SDE version.

    Particle dynamics:
        dX_t = [v_star(X_t,t) + 0.5 sigma_t^2 s_star(X_t,t)] dt
               + sigma_t dW_t

    Tracked component log-density dynamics:
        d log q_i(t, X_t)
        =
        [
            - div v_i
            + (v_star - v_i) dot s_i
            + 0.5 sigma_t^2 (s_star dot s_i + div s_i)
        ] dt
        + sigma_t s_i dot dW_t

    Weight increment:
        d log w_t
        =
        [
            div v_star
            + sum_i dot_gamma_i(t) log q_i(t, X_t)
            + sum_i gamma_i(t) {
                - div v_i + (v_star - v_i) dot s_i
              }
        ] dt

    Returns:
        If print_resample_history:
            x, logw, logw_history, sample_history, resample_history
        Else:
            x, logw, logw_history, sample_history
    """

    def divergence(f, x, t):
        """Hutchinson trace estimator for div f(x,t)."""
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            e = torch.randn_like(x_req)
            out = torch.sum(f(x_req, t) * e)
            grad = torch.autograd.grad(out, x_req, create_graph=False)[0]
            return (grad * e).sum(dim=-1, keepdim=True).detach()

    def make_lifted_v(i):
        def lifted_v(_x, _t):
            return emb_list[i](v_fn_list[i](proj_list[i](_x), _t))
        return lifted_v

    def make_lifted_s(i):
        def lifted_s(_x, _t):
            return emb_list[i](s_fn_list[i](proj_list[i](_x), _t))
        return lifted_s

    def eval_time_fn(fn, t, bs, device):
        val = fn(t)
        if not torch.is_tensor(val):
            val = torch.tensor(val, device=device, dtype=t.dtype)
        val = val.to(device=device, dtype=t.dtype)

        if val.ndim == 0:
            val = val.view(1, 1).expand(bs, 1)
        elif val.shape == t.shape:
            pass
        elif val.numel() == 1:
            val = val.reshape(1, 1).expand(bs, 1)

        return val

    x = x0.clone().to(device)
    bs = x.size(0)
    logw = torch.zeros(bs, 1, device=device)

    times = torch.linspace(t0, t1, n_steps + 1, device=device)
    dt = torch.tensor((t1 - t0) / n_steps, device=device)
    sqrt_dt = torch.sqrt(dt)

    if len(gamma_list) == 0:
        raise ValueError("gamma_list must contain at least one exponent.")

    if callable(gamma_list[0]):
        gamma_fns = gamma_list
        if d_gamma_list is None:
            raise ValueError("d_gamma_list must be provided for time-dependent gammas.")
        d_gamma_fns = d_gamma_list
    else:
        gamma_fns = [
            lambda t, val=float(g): torch.full_like(t, val)
            for g in gamma_list
        ]
        d_gamma_fns = [
            lambda t: torch.zeros_like(t)
            for _ in gamma_list
        ]

    # Initialize log q_i at t=0 as standard Gaussian density on each projected space.
    log_q_i_list = []
    for proj in proj_list:
        x_proj = proj(x)
        d_i = x_proj.shape[-1]
        log_norm_const = -0.5 * d_i * np.log(2.0 * np.pi)
        log_exp = -0.5 * torch.sum(x_proj ** 2, dim=-1, keepdim=True)
        log_q_i_list.append((log_norm_const + log_exp).detach())

    logw_history = []
    sample_history = []
    resample_history = []

    if t_list is not None:
        t_set = set(float(v) for v in t_list)
    else:
        t_set = None

    for it in tqdm(range(n_steps)):
        # Everything in this block is evaluated at pre-propagation X_t.
        t = times[it].expand(bs, 1)

        gamma_t_vals = [
            eval_time_fn(g, t, bs, device)
            for g in gamma_fns
        ]
        d_gamma_t_vals = [
            eval_time_fn(dg, t, bs, device)
            for dg in d_gamma_fns
        ]

        sigma_t = sigma_fn(t)
        if not torch.is_tensor(sigma_t):
            sigma_t = torch.tensor(sigma_t, device=device, dtype=x.dtype)
        sigma_t = sigma_t.to(device=device, dtype=x.dtype)
        if sigma_t.ndim == 0:
            sigma_t = sigma_t.view(1, 1).expand(bs, 1)
        elif sigma_t.numel() == 1:
            sigma_t = sigma_t.reshape(1, 1).expand(bs, 1)

        # Component fields at X_t.
        v_tilde_list = []
        s_tilde_list = []
        div_v_tilde_list = []
        div_s_tilde_list = []

        for i in range(len(gamma_fns)):
            v_tilde_i = emb_list[i](v_fn_list[i](proj_list[i](x), t)).detach()
            s_tilde_i = emb_list[i](s_fn_list[i](proj_list[i](x), t)).detach()

            div_v_tilde_i = divergence(make_lifted_v(i), x, t)
            div_s_tilde_i = divergence(make_lifted_s(i), x, t)

            v_tilde_list.append(v_tilde_i)
            s_tilde_list.append(s_tilde_i)
            div_v_tilde_list.append(div_v_tilde_i)
            div_s_tilde_list.append(div_s_tilde_i)

        s_star_t = sum(
            gamma_t_vals[i] * s_tilde_list[i]
            for i in range(len(gamma_fns))
        )

        v_star_t = v_star(x, t).detach()
        div_v_star_t = divergence(v_star, x, t)

        # ACE component corrector D_i evaluated at X_t.
        corrector_terms = []
        for i in range(len(gamma_fns)):
            dot_product = torch.sum(
                (v_star_t - v_tilde_list[i]) * s_tilde_list[i],
                dim=1,
                keepdim=True,
            )
            D_i = -div_v_tilde_list[i] + dot_product
            corrector_terms.append(D_i)

        # Weight update uses current log q_i(t, X_t), not the post-update value.
        d_gamma_log_q_sum = sum(
            d_gamma_t_vals[i] * log_q_i_list[i]
            for i in range(len(gamma_fns))
        )

        gamma_corrector_sum = sum(
            gamma_t_vals[i] * corrector_terms[i]
            for i in range(len(gamma_fns))
        )

        increment = div_v_star_t + d_gamma_log_q_sum + gamma_corrector_sum
        logw = (logw + increment * dt).detach()

        # Use one Brownian increment for both X update and Ito log q update.
        xi = torch.randn_like(x)
        dW = sqrt_dt * xi

        drift_t = v_star_t + 0.5 * sigma_t ** 2 * s_star_t
        x_next = (x + drift_t * dt + sigma_t * dW).detach()

        # SDE Ito update for log q_i(t, X_t) -> log q_i(t+dt, X_{t+dt}).
        for i in range(len(gamma_fns)):
            ito_drift = 0.5 * sigma_t ** 2 * (
                torch.sum(s_star_t * s_tilde_list[i], dim=1, keepdim=True)
                + div_s_tilde_list[i]
            )

            ito_noise = sigma_t * torch.sum(
                s_tilde_list[i] * dW,
                dim=1,
                keepdim=True,
            )

            log_q_i_list[i] = (
                log_q_i_list[i]
                + (corrector_terms[i] + ito_drift) * dt
                + ito_noise
            ).detach()

        # Resampling after propagation and weight update.
        if resample:
            if t_set is None:
                do_resample = (
                    it < n_steps * t_max
                    and it > n_steps * t_min
                )
            else:
                do_resample = (float(it) / float(n_steps)) in t_set

            if do_resample:
                weights = F.softmax(logw.squeeze(-1), dim=0)
                ess = 1.0 / torch.sum(weights ** 2)

                if t_set is not None or ess < ess_threshold * bs or it == n_steps - 5:
                    resample_history.append(it)

                    idx = torch.multinomial(weights, bs, replacement=True)

                    x = x_next[idx].detach()
                    logw = torch.zeros_like(logw)
                    log_q_i_list = [log_q[idx].detach() for log_q in log_q_i_list]
                else:
                    x = x_next
            else:
                x = x_next
        else:
            x = x_next

        logw_history.append(logw.clone().cpu())
        sample_history.append(x.clone().cpu())

    if print_resample_history:
        return x, logw, logw_history, sample_history, resample_history

    return x, logw, logw_history, sample_history