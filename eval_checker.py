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
from hcg.interpolant import MLPInstFlexible, run_training_v_s
from hcg.sample_data import sample_checkerboard, make_conditions, sample_data_model1, sample_data_model2, sample_data_model3, ground_truth_hcg, plot_diagnostics
from hcg.hcg import simulate_hcg, simulate_hcg_generalized
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); random.seed(0); np.random.seed(0)
print("device →", device)

results = []

### Run this ONCE per experiment
new_experiment = True  # Set to False to reuse an old experiment ID
if new_experiment:  # Run the experiment with the current date and time as the experiment ID
    experiment_id = f"experiment_checker_{datetime.now().strftime('%Y%m%d')}" #_%H%M%S
    os.makedirs(experiment_id, exist_ok=True)
else:
    experiment_id = "experiment_checker_20250830"  # Use a fixed ID for reproducibility


bs = 10000
A, B = 1, 1  # Conditions to use for training
# plot ground truth
samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).cpu().numpy()
plot_diagnostics(samples_gt, torch.zeros(bs), [torch.zeros(bs)], save_name=f"{experiment_id}/ground_truth")

# Load all the models
u_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device); u_model1.load_state_dict(torch.load(f"models/u_model1_X_given_A.pth")); u_model1.eval()
s_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device); s_model1.load_state_dict(torch.load(f"models/s_model1_X_given_A.pth")); s_model1.eval()
u_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device); u_model2.load_state_dict(torch.load(f"models/u_model2_XY_given_B.pth")); u_model2.eval()
s_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device); s_model2.load_state_dict(torch.load(f"models/s_model2_XY_given_B.pth")); s_model2.eval()
u_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device); u_model3.load_state_dict(torch.load(f"models/u_model3_X.pth")); u_model3.eval()
s_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device); s_model3.load_state_dict(torch.load(f"models/s_model3_X.pth")); s_model3.eval()

def v1_fn(x, t, A): return u_model1(x, t, A)
def s1_fn(x, t, A): return s_model1(x, t, A)
def v2_fn(x, t): return u_model3(x, t)
def s2_fn(x, t): return s_model3(x, t)
def v3_fn(z, t, B): return u_model2(z, t, B)
def s3_fn(z, t, B): return s_model2(z, t, B)
def sigma_fn(t): return 1.0 * torch.ones_like(t)
print("Models loaded.")
    
    
for i in tqdm([0,1,2,3,4]):
    torch.manual_seed(i); random.seed(i); np.random.seed(i)
    
    print("Running HCG p1p3/p2")
    x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
    A, B = 1, 1  # Conditioning values

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # v1(X|A)
            lambda x, t: v2_fn(x[:, :1], t),                                                 # v2(X)
            lambda x, t: v3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # s1(X|A)
            lambda x, t: s2_fn(x[:, :1], t),                                                 # s2(X)
            lambda x, t: s3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # s3(Z|B)
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
        v_star= lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=True,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/hcg-1")

    samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["HCG[1,-1,1]", w1, w2, mmd_rbf, total_var])



    print("Running FKC p1p3/p2")
    x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
    A, B = 1, 1  # Conditioning values
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
        lambda z, t: torch.cat([ v1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), v_fp(z[:, 1:2], t) ], dim=1),
        lambda z, t: torch.cat([ v2_fn(z[:, :1], t),            v_fn(z[:, 1:2], t) ], dim=1),
        lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device))   # already 2D
    ]
    s_fn_list = [
        lambda z, t: torch.cat([ s1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), s_fp(z[:, 1:2]) ], dim=1),
        lambda z, t: torch.cat([ s2_fn(z[:, :1], t),            s_fn(z[:, 1:2]) ], dim=1),
        lambda z, t: s3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device))   # already 2D
    ]
    # pass identity projections since all are now 2D
    emb_list = [lambda z: z, lambda z: z, lambda z: z]
    trans_emb_list = [lambda z: z, lambda z: z, lambda z: z]
    # v_star: extend original v1 in x and zero in y
    v_star = lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device))
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


    samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["FKC[1,-1,1]", w1, w2, mmd_rbf, total_var])




    # Product of Experts

    print("Running HCG p1p2")
    x0 = (1/torch.sqrt(torch.tensor(2))) * torch.randn(bs, 2).to("cuda")  # (X, Y) sample
    A, B = 1, 1  # Conditioning values

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # v1(X|A)
            lambda x, t: v3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # s1(X|A)
            lambda x, t: s3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # s3(Z|B)
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
        v_star= lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=True,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/hcg-2")


    samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["HCG[1,1]", w1, w2, mmd_rbf, total_var])





    print("Running FKC p1p2")
    x0 = (1/torch.sqrt(torch.tensor(2))) * torch.randn(bs, 2).to("cuda")  # (X, Y) sample
    A, B = 1, 1  # Conditioning values

    # choose f(y) = N(0, 1)  => score s_f(y) = -y
    def s_f(y):           # y has shape (bs,1)
        return -y
    def v_f(y, t):        # choose zero deterministic drift in y
        return torch.zeros_like(y)

    v_fn_list = [
        lambda z, t: torch.cat([ v1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), v_f(z[:, 1:2], t) ], dim=1),
        lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device))   # already 2D
    ]
    s_fn_list = [
        lambda z, t: torch.cat([ s1_fn(z[:, :1], t, torch.full((z.size(0), 1), A, device=z.device)), s_f(z[:, 1:2]) ], dim=1),
        lambda z, t: s3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device))   # already 2D
    ]
    emb_list = [lambda z: z, lambda z: z]
    trans_emb_list = [lambda z: z, lambda z: z]
    v_star = lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device))


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

    samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["FKC[1,1]", w1, w2, mmd_rbf, total_var])



    print("Running Target Score p1p3/p2")
    x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
    A, B = 1, 1  # Conditioning values

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # v1(X|A)
            lambda x, t: v2_fn(x[:, :1], t),                                                 # v2(X)
            lambda x, t: v3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # s1(X|A)
            lambda x, t: s2_fn(x[:, :1], t),                                                 # s2(X)
            lambda x, t: s3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # s3(Z|B)
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
        v_star= lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=False,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/heur-1")

    samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["Heuristic[1,-1,1]", w1, w2, mmd_rbf, total_var])
    
    
    
    print("Running Target Score p1p2")
    x0 = (1/torch.sqrt(torch.tensor(2))) * torch.randn(bs, 2).to("cuda")  # (X, Y) sample
    A, B = 1, 1  # Conditioning values

    samples, logw_final, logw_history, sample_history = simulate_hcg_generalized(
        x0=x0, 
        dim_list=[1, 2],  # dims for q^1(X|A), q^2(X), q^3(Z|B)
        v_fn_list=[
            lambda x, t: v1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # v1(X|A)
            lambda x, t: v3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # v3(Z|B)
        ],
        s_fn_list=[
            lambda x, t: s1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # s1(X|A)
            lambda x, t: s3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # s3(Z|B)
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
        v_star= lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
        t0=0.0, t1=1.0, n_steps=1000,
        device="cuda",
        resample=False,
        ess_threshold=0.5
    )
    samples = samples.cpu().numpy()
    plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/heur-2")


    samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
    w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
        torch.tensor(samples_gt), torch.tensor(samples)
    )
    results.append(["Heuristic[1,1]", w1, w2, mmd_rbf, total_var])
    
    
    
    print(results)
    # save results as pd
    import pandas as pd

    df = pd.DataFrame(results, columns=["Method", "W1", "W2", "MMD", "TV"])
    df.to_csv(f"{experiment_id}/results_checker.csv", index=False)


import os
from datetime import datetime
import pandas as pd
import numpy as np


exp_type = "checker"  # Options: "checker", "gmm"

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