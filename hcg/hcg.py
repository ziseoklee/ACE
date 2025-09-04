import torch
import torch.nn.functional as F

@torch.no_grad()
def simulate_hcg(
    z0,                       # [bs, 2]
    v1_fn, s1_fn,            # q^1(X|A)
    v2_fn, s2_fn,            # q^2(X) (marginal)
    v3_fn, s3_fn,            # q^3(Z|B)
    sigma_fn,                # t → [bs, 1]
    A=1, B=1,                # conditioning values
    t0=0.0, t1=1.0,
    n_steps=1000,
    device="cuda",
    resample=True,
    perturb=True,            # whether to add perturbation after resampling
    ess_threshold=0.5,
    v_star_type = "v3"       # "v1", "v2", "v3", "v1+v2+v3"
):
    logw_history = []  # store log weights each step
    sample_history = []  # store samples each step

    z = z0.clone().to(device)             # [bs, 2] → Z = [X, Y]
    bs = z.size(0)
    dx = 1
    dy = z.size(1) - dx
    logw = torch.zeros(bs, 1, device=device)
    times = torch.linspace(t0, t1, n_steps + 1, device=device)
    dt = torch.tensor((t1 - t0) / n_steps, device=device)


    # Conditioning tensors
    A_tensor = torch.full((bs, 1), A, dtype=torch.float32, device=device)
    B_tensor = torch.full((bs, 1), B, dtype=torch.float32, device=device)

    for i in range(n_steps):
        t = times[i].expand(bs, 1)
        sigma_t = sigma_fn(t)

        x = z[:, :dx]
    
        # Use Hutchinson trace estimator for divergence terms
        def divergence(f, x, t):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                e = torch.randn_like(x)
                e.requires_grad_(False)
                out = torch.sum(f(x, t) * e)
                grad = torch.autograd.grad(out, x, create_graph=False, retain_graph=False)[0]
                return (grad * e).sum(dim=1, keepdim=True)
        
        # --- Full-space model (Z = [X,Y]) conditioned on B ---
        v1 = v1_fn(x, t, A_tensor)
        v2 = v2_fn(x, t)
        s1 = s1_fn(x, t, A_tensor)
        s2 = s2_fn(x, t)
        v3 = v3_fn(z, t, B_tensor)
        s3 = s3_fn(z, t, B_tensor)

        div_v1 = divergence(lambda _x, _t: v1_fn(_x, _t, A_tensor), x, t)
        div_v2 = divergence(v2_fn, x, t)
        div_v3 = divergence(lambda _z, _t: v3_fn(_z, _t, B_tensor), z, t)
        # --- Projected velocity/score with conditioning on A ---

        # Extend to Z-space
        if v_star_type == "v1":
            v_star = torch.cat([v1, torch.zeros(bs, dy, device=device)], dim=1)
            div_v_star = div_v1
        elif v_star_type == "v2":
            v_star = torch.cat([v2, torch.zeros(bs, dy, device=device)], dim=1)
            div_v_star = div_v2
        elif v_star_type == "v3":
            v_star = v3
            div_v_star = div_v3
        elif v_star_type == "v1+v2+v3":
            v_star = torch.cat([v1 + v2, torch.zeros(bs, dy, device=device)], dim=1) + v3
            div_v_star = div_v1 + div_v2 + div_v3
        else:
            raise ValueError("v_star must be one of 'v1', 'v2', 'v3', 'v1+v2+v3'")

        # Extended score
        s_star = torch.cat([s1 - s2, torch.zeros(bs, dy, device=device)], dim=1) + s3
        

        # --- Drift & diffusion update ---
        drift = v_star + 0.5 * sigma_t**2 * (s_star)
        noise = torch.randn_like(z) * (sigma_t * torch.sqrt(dt))
        z = z + drift * dt + noise

        # --- Log weight update ---
        increment = (
            - div_v1 + div_v2 - div_v3 + div_v_star
            - torch.sum(v1 * s1, dim=1, keepdim=True)
            + torch.sum(v2 * s2, dim=1, keepdim=True)
            - torch.sum(v3 * s3, dim=1, keepdim=True)
            + torch.sum(v_star*s_star, dim=1, keepdim=True)
        )
        logw += increment * dt
        logw_history.append(logw.clone().cpu())

        # --- Resampling ---
        if resample:
            weights = F.softmax(logw.squeeze(-1), dim=0)
            ess = 1.0 / torch.sum(weights**2)
            if ess < ess_threshold * bs or i == n_steps - 5:
                idx = torch.multinomial(weights, bs, replacement=True)
                z = z[idx]
                if perturb:
                    # --- Resampling with perturbation ---
                    epsilon = 1e-3  # adjust scale as needed
                    z = z + epsilon * torch.randn_like(z)
                logw = torch.zeros_like(logw)
        sample_history.append(z.clone().cpu())
    return z, logw, logw_history, sample_history

@torch.no_grad()
def simulate_hcg_generalized(
    x0: torch.tensor,        # [bs, 2]
    dim_list: list,          # sorted in ascending order
    v_fn_list: list,         # list of velocity functions
    s_fn_list: list,         # list of score functions
    gamma_list: list,        # list of gammas (exponents of each q^(i)_t)
    proj_list: list,          # list of embedding functions
    emb_list: list,    # list of transpose embedding functions
    sigma_fn: callable,
    v_star: callable,        # v_star(X, t)
    t0=0.0, t1=1.0,
    n_steps=1000,
    device="cuda",
    resample=True,
    ess_threshold=0.5
):
    # Use Hutchinson trace estimator for divergence terms
    def divergence(f, x, t):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            e = torch.randn_like(x)
            e.requires_grad_(False)
            out = torch.sum(f(x, t) * e)
            grad = torch.autograd.grad(out, x, create_graph=False, retain_graph=False)[0]
            return (grad * e).sum(dim=1, keepdim=True)

    x = x0.clone().to(device)             # [bs, 2] → Z = [X, Y]
    bs = x.size(0)
    logw = torch.zeros(bs, 1, device=device)
    times = torch.linspace(t0, t1, n_steps + 1, device=device)
    dt = torch.tensor((t1 - t0) / n_steps, device=device)

    logw_history = []  # store log weights each step
    sample_history = []  # store samples each step
    
    for i in range(n_steps):
        t = times[i].expand(bs, 1)
        sigma_t = sigma_fn(t)

        v_star_t = v_star(x, t)
        s_star_t = sum([gamma_list[i] * emb_list[i](s_fn_list[i](proj_list[i](x), t)) for i in range(len(gamma_list))])
        div_v_star_t = divergence(v_star, x, t)
        other_terms = sum([gamma_list[i]*(-divergence(lambda _x, _t: emb_list[i](v_fn_list[i](proj_list[i](_x), _t)), x, t) + torch.sum((v_star_t - emb_list[i](v_fn_list[i](proj_list[i](x), t))) * emb_list[i](s_fn_list[i](proj_list[i](x), t)), dim=1, keepdim=True)) for i in range(len(gamma_list) )])

        drift_t = v_star_t + 0.5 * sigma_t**2 * (s_star_t)
        noise = torch.randn_like(x) * (sigma_t * torch.sqrt(dt))
        x = x + drift_t*dt + noise

        increment = div_v_star_t + other_terms
        logw += increment * dt
        logw_history.append(logw.clone().cpu())

        # --- Resampling ---
        if resample:
            weights = F.softmax(logw.squeeze(-1), dim=0)
            ess = 1.0 / torch.sum(weights**2)
            if ess < ess_threshold * bs or i == n_steps - 5:
                idx = torch.multinomial(weights, bs, replacement=True)
                x = x[idx]
                logw = torch.zeros_like(logw)
        sample_history.append(x.clone().cpu())
    return x, logw, logw_history, sample_history