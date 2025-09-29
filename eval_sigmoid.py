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
from hcg.interpolant import Interpolant as Interpolant
from hcg.interpolant import FlowMatcher as FlowMatcher
from hcg.interpolant import plot_path_trajectories as plot_path_trajectories
import json
from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); random.seed(0); np.random.seed(0)
print("device →", device)

top_experiment_id = "#Final_Evaluation_Results#"  # Use a fixed ID for reproducibility
if not os.path.exists(top_experiment_id):
    os.makedirs(top_experiment_id)

model_path = "PretrainedToyModels"

def load_interpolants_from_json(path):
    """Load interpolants.json and return a dict of Interpolant objects."""
    with open(path, "r") as f:
        interpolants_raw = json.load(f)

    interpolants = {}
    for name, funcs in interpolants_raw.items():
        alpha_t = eval(funcs["alpha_t"], {"torch": torch})
        beta_t = eval(funcs["beta_t"], {"torch": torch})
        d_alpha_t = eval(funcs["d_alpha_t"], {"torch": torch})
        d_beta_t = eval(funcs["d_beta_t"], {"torch": torch})
        interpolants[name] = Interpolant(alpha_t, beta_t, d_alpha_t, d_beta_t, name=name)
    return interpolants

interpolant_schedules = load_interpolants_from_json("interpolant_schedules.json")

Bound = 10
bs = 10000; n_steps=1000; ess_threshold=0.7
A, B = 1, 1  # Conditions to use for training
# plot ground truth
samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).cpu().numpy()
plot_diagnostics(samples_gt, torch.zeros(bs), [torch.zeros(bs)], save_name=f"{top_experiment_id}/ground_truth")
clear_output()

names = ["sigmoid", "default_linear", "default_linear"]
subexperiment_id = f"{names}"
experiment_id = f"{top_experiment_id}/{subexperiment_id}_sigmoid_bs{bs}_n_steps{n_steps}_ess{ess_threshold}"
if not os.path.exists(experiment_id):
    os.makedirs(experiment_id)

u_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device); u_model1.load_state_dict(torch.load(f"{model_path}/u_model1_X_given_A_alpha={names[0]}.pth")); u_model1.eval()
s_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device); s_model1.load_state_dict(torch.load(f"{model_path}/s_model1_X_given_A_alpha={names[0]}.pth")); s_model1.eval()
u_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device); u_model2.load_state_dict(torch.load(f"{model_path}/u_model2_XY_given_B_alpha={names[2]}.pth")); u_model2.eval()
s_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device); s_model2.load_state_dict(torch.load(f"{model_path}/s_model2_XY_given_B_alpha={names[2]}.pth")); s_model2.eval()
u_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device); u_model3.load_state_dict(torch.load(f"{model_path}/u_model3_X_alpha={names[1]}.pth")); u_model3.eval()
s_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device); s_model3.load_state_dict(torch.load(f"{model_path}/s_model3_X_alpha={names[1]}.pth")); s_model3.eval()

def v1_fn(x, t, A): return u_model1(x, t, A)
def s1_fn(x, t, A): return s_model1(x, t, A)
def v2_fn(x, t): return u_model3(x, t)
def s2_fn(x, t): return s_model3(x, t)
def v3_fn(z, t, B): return u_model2(z, t, B)
def s3_fn(z, t, B): return s_model2(z, t, B)
def sigma_fn(t): return 0.5 * torch.ones_like(t)

v_fn_list=[
        lambda x, t: v1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # v1(X|A)
        lambda x, t: v2_fn(x[:, :1], t),                                                 # v2(X)
        lambda x, t: v3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # v3(Z|B)
    ]
s_fn_list=[
        lambda x, t: s1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # s1(X|A)
        lambda x, t: s2_fn(x[:, :1], t),                                                 # s2(X)
        lambda x, t: s3_fn(x, t, torch.full((x.size(0), 1), B, device=x.device))         # s3(Z|B)
    ]
proj_list=[
        lambda z: z[:, :1],    # project to X 
        lambda z: z[:, :1],    # project to X
        lambda z: z            # identity for Z
    ]
emb_list=[
        lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
        lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
        lambda z: z  # identity
    ]

print("Models loaded.")
seeds=[0,1,2,3,4]
gs = [-1.0, -1.2, -1.4, -1.6, -1.8, -2, -4, -6]
for g in gs:
    g = torch.tensor(g)
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
        print(f"Seed set to {seed}")

        print(f"Annealed Temperature = {g}")
        print("Simulating NR")
        gamma_list = [
            lambda t : torch.tensor(1),
            lambda t : torch.tensor(g),
            lambda t : torch.tensor(1)
        ]
        d_gamma_list = [
            lambda t: torch.zeros_like(t),
            lambda t: torch.zeros_like(t),
            lambda t: torch.zeros_like(t)
        ]
        
        x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
        A, B = 1, 1  # Conditioning values
        samples, logw_final, logw_history, sample_history, resample_history = simulate_hcg_generalized(
            x0=x0, v_fn_list=v_fn_list, s_fn_list=s_fn_list, proj_list=proj_list, emb_list=emb_list, sigma_fn=sigma_fn,
            v_star= lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
            t0=0.0, t1=1.0, n_steps=n_steps, device="cuda", ess_threshold=ess_threshold, print_resample_history=True,
            gamma_list=gamma_list,
            d_gamma_list=d_gamma_list,
            resample=False
        )
        samples = samples.cpu().numpy()
        plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/alpha={names}_NR_seed{seed}_g{g}")
        plot_path_trajectories(sample_history, n_frame=6, resample_history=None, experiment_id=experiment_id, name=f"alpha={names}_NR_seed{seed}_g{g}", deg=-50)
        plt.close(); clear_output()
        
        print("Evaluating NR")
        w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
            torch.tensor(samples_gt), torch.tensor(samples)
        )
        results.append([seed, "NR", w1, w2, mmd_rbf, total_var])

        print("Simulating FKC-H (Constant Gammas)")
        gamma_list = [
            lambda t : torch.tensor(1),
            lambda t : torch.tensor(g),
            lambda t : torch.tensor(1)
        ]
        d_gamma_list = [
            lambda t: torch.zeros_like(t),
            lambda t: torch.zeros_like(t),
            lambda t: torch.zeros_like(t)
        ]

        x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
        A, B = 1, 1  # Conditioning values
        samples, logw_final, logw_history, sample_history, resample_history = simulate_hcg_generalized(
            x0=x0, v_fn_list=v_fn_list, s_fn_list=s_fn_list, proj_list=proj_list, emb_list=emb_list, sigma_fn=sigma_fn,
            v_star= lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
            t0=0.0, t1=1.0, n_steps=n_steps, device="cuda", ess_threshold=ess_threshold, print_resample_history=True,
            gamma_list=gamma_list,
            d_gamma_list=d_gamma_list,
            resample=True
        )
        samples = samples.cpu().numpy()
        plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/alpha={names}_FKC-H_seed{seed}_g{g}")
        plot_path_trajectories(sample_history, n_frame=6, resample_history=None, experiment_id=experiment_id, name=f"alpha={names}_FKC-H_seed{seed}_g{g}", deg=-50)
        plt.close(); clear_output()

        print("Evaluating FKC-H")
        w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
            torch.tensor(samples_gt), torch.tensor(samples)
        )
        results.append([seed, "FKC-H", w1, w2, mmd_rbf, total_var])

        print("Simulating ACE (Adaptive Gammas)")
        gamma_list = [
            lambda t : torch.tensor(1),
            lambda t : torch.tensor(g),
            lambda t : torch.tensor(1) + torch.tensor(Bound * t * (1-t))
        ]
        d_gamma_list = [
            lambda t: torch.zeros_like(t),
            lambda t: torch.zeros_like(t),
            lambda t: torch.zeros_like(t) + torch.tensor(Bound * (1 - 2*t))
        ]

        x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
        A, B = 1, 1  # Conditioning values
        samples, logw_final, logw_history, sample_history, resample_history = simulate_hcg_generalized(
            x0=x0, v_fn_list=v_fn_list, s_fn_list=s_fn_list, proj_list=proj_list, emb_list=emb_list, sigma_fn=sigma_fn,
            v_star= lambda z, t: v3_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
            t0=0.0, t1=1.0, n_steps=n_steps, device="cuda", ess_threshold=ess_threshold, print_resample_history=True,
            gamma_list=gamma_list,
            d_gamma_list=d_gamma_list,
            resample=True
        )
        samples = samples.cpu().numpy()
        plot_diagnostics(samples, logw_final, logw_history, save_name=f"{experiment_id}/alpha={names}_ACE_seed{seed}_g{g}")
        plot_path_trajectories(sample_history, n_frame=6, resample_history=None, experiment_id=experiment_id, name=f"alpha={names}_ACE_seed{seed}_g{g}", deg=-50)
        plt.close(); clear_output()

        print("Evaluating ACE")
        w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
            torch.tensor(samples_gt), torch.tensor(samples)
        )
        results.append([seed, "ACE", w1, w2, mmd_rbf, total_var])

    df = pd.DataFrame(results, columns=["seed", "method", "W1", "W2", "MMD_RBF", "Total Var."])
    df.to_csv(f"{experiment_id}/experiment_results_numseeds{len(seeds)}_bs{bs}_n_steps{n_steps}_g{g}.csv", index=False)
    print(df)