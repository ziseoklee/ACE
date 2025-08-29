def sample_checkerboard(bs):
    x1 = torch.rand(bs) * 4 - 2
    x2_ = torch.rand(bs) - torch.randint(2, (bs,)) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    return (torch.cat([x1[:, None], x2[:, None]], 1) * 2)

def make_conditions(XY):
    X, Y = XY[:, 0], XY[:, 1]
    A = (((-2 <= X) & (X <= 0)) | ((2 <= X) & (X <= 4))).float().unsqueeze(1)
    B = ((0 <= Y) & (Y <= 4)).float().unsqueeze(1)
    C = ((0 <= X) & (X <= 4)).float().unsqueeze(1)
    return X.unsqueeze(1), Y.unsqueeze(1), A, B, C

def sample_data_model1(batch_size, cond_val=None):
    """
    Sample x | A = cond_val
    Returns:
        x:   (batch_size, 1)
        a:   (batch_size, 1)
    """
    xy = sample_checkerboard(batch_size * 10)  # oversample to filter
    X, _, A, _, _ = make_conditions(xy)
    if cond_val is None:
        # Sample without condition
        return X[:batch_size], A[:batch_size]
    else:
        # Sample with condition
        mask = (A.squeeze() == cond_val)
        X_cond = X[mask][:batch_size]
        A_cond = A[mask][:batch_size]
        return X_cond, A_cond

def sample_data_model2(batch_size, cond_val=None):
    """
    Sample (x, y) | B = cond_val
    Returns:
        xy: (batch_size, 2)
        b:  (batch_size, 1)
    """
    xy = sample_checkerboard(batch_size * 10)
    _, _, _, B, _ = make_conditions(xy)
    if cond_val is None:
        # Sample without condition
        return xy[:batch_size], B[:batch_size]
    else:
        # Sample with condition
        mask = (B.squeeze() == cond_val)
        if mask.sum() >= batch_size:
            xy_cond = xy[mask][:batch_size]
            B_cond = B[mask][:batch_size]
            return xy_cond, B_cond
        
def sample_data_model3(batch_size, cond_val=None):
    """
    Sample x ~ p(x)
    Returns:
        x: (batch_size, 1)
        None: for compatibility
    """
    xy = sample_checkerboard(batch_size)
    X, _, _, _, _ = make_conditions(xy)
    return X, None

def ground_truth_hcg(batch_size, cond_A=None, cond_B=None):
    """
    Sample from the ground truth HCG distribution p(x, y | A=cond_A, B=cond_B)
    Returns:
        xy: (batch_size, 2)
    """
    samples = []
    while len(samples) < batch_size:
        xy = sample_checkerboard(batch_size * 10)
        X, Y, A, B, _ = make_conditions(xy)
        if cond_A is not None and cond_B is not None:
            mask = (A.squeeze() == cond_A) & (B.squeeze() == cond_B)
        elif cond_A is not None:
            mask = (A.squeeze() == cond_A)
        elif cond_B is not None:
            mask = (B.squeeze() == cond_B)
        else:
            mask = torch.ones(len(X), dtype=bool)
        filtered_xy = xy[mask]
        samples.append(filtered_xy)
        samples = torch.cat(samples, dim=0)
    return samples[:batch_size]

import matplotlib.pyplot as plt
import torch

def plot_diagnostics(samples, logw_final, logw_history, save_name="diagnostics"):
    """
    Generate diagnostic plots:
    1. Scatter plot of samples
    2. Histogram of final weights
    3. ESS and variance over time
    
    Args:
        samples (torch.Tensor or np.ndarray): Shape (N,2), 2D samples.
        logw_final (torch.Tensor): Final log-weights.
        logw_history (list[torch.Tensor]): Sequence of log-weight tensors over time.
        save_name (str): Prefix for saved files.
    """
    # --- Plot samples ---
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3)
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal")
    plt.title(r"Samples from $p^*_1$")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"{save_name}_samples.png", dpi=300)
    plt.show()
    plt.close()

    # --- Plot final weight distribution ---
    final_weights = torch.exp(logw_final - torch.max(logw_final))  # stabilize
    final_weights = final_weights.cpu().numpy().squeeze()

    plt.figure(figsize=(6,4))
    plt.hist(final_weights, bins=50, density=True, alpha=0.7)
    plt.title("Final Weight Distribution")
    plt.xlabel("Weight")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(f"{save_name}_weights.png", dpi=300)
    plt.show()
    plt.close()

    # --- Time evolution diagnostics ---
    ess_list, var_list = [], []
    for lw in logw_history:
        w = torch.exp(lw - torch.max(lw))
        w = w / torch.sum(w)
        ess = 1.0 / torch.sum(w**2)
        var = torch.var(w)
        ess_list.append(ess.item())
        var_list.append(var.item())

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(ess_list)
    plt.title("ESS over time")
    plt.xlabel("Step")
    plt.ylabel("ESS")

    plt.subplot(1,2,2)
    plt.plot(var_list)
    plt.title("Weight Variance over time")
    plt.xlabel("Step")
    plt.ylabel("Variance")
        
    plt.tight_layout()
    plt.savefig(f"{save_name}_diagnostics.png", dpi=300)
    plt.show()
    plt.close()

