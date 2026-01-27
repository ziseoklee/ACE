import torch
import matplotlib.pyplot as plt
import numpy as np
import random, os
from datetime import datetime
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from ace_lib.metrics.export import compute_sample_based_metrics
from ace_lib.interpolant import MLPInstFlexible, run_training_v_s
from ace_lib.sample_data import sample_checkerboard, make_conditions, sample_data_model1, sample_data_model2, sample_data_model3, ground_truth_hcg, plot_diagnostics
from ace_lib.ace import simulate_ace
from ace_lib.utils import get_experiment_dir, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)
print("device: ", device)

exp_dir = get_experiment_dir(f"ace_eval_runs_{datetime.now().strftime('%Y%m%d')}", "checker_onesided")
print(f"Experiment directory: {exp_dir}")

schedule_combination = ["cos_t", "ddpm_linear", "default_linear"]

results = []

bs = 10000
A, B = 1, 1  # Conditions to use for training
# plot ground truth
samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).cpu().numpy()
plot_diagnostics(samples_gt, torch.zeros(bs), [torch.zeros(bs)], save_name=f"{exp_dir}/ground_truth")

# Load the pretrained models
model_path = "PretrainedToyModels"
u_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device); u_model1.load_state_dict(torch.load(f"{model_path}/u_model1_X_given_A_alpha={schedule_combination[0]}.pth")); u_model1.eval()
s_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device); s_model1.load_state_dict(torch.load(f"{model_path}/s_model1_X_given_A_alpha={schedule_combination[0]}.pth")); s_model1.eval()
u_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device); u_model2.load_state_dict(torch.load(f"{model_path}/u_model2_XY_given_B_alpha={schedule_combination[1]}.pth")); u_model2.eval()
s_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device); s_model2.load_state_dict(torch.load(f"{model_path}/s_model2_XY_given_B_alpha={schedule_combination[1]}.pth")); s_model2.eval()
u_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device); u_model3.load_state_dict(torch.load(f"{model_path}/u_model3_X_alpha={schedule_combination[2]}.pth")); u_model3.eval()
s_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device); s_model3.load_state_dict(torch.load(f"{model_path}/s_model3_X_alpha={schedule_combination[2]}.pth")); s_model3.eval()

def v1_fn(x, t, A): return u_model1(x, t, A)
def s1_fn(x, t, A): return s_model1(x, t, A)
def v2_fn(z, t, B): return u_model2(z, t, B)
def s2_fn(z, t, B): return s_model2(z, t, B)
def v3_fn(x, t): return u_model3(x, t)
def s3_fn(x, t): return s_model3(x, t)
def sigma_fn(t): return 0.5 * torch.ones_like(t)
print("Models loaded.")

v_fn_list=[
        lambda x, t: v1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # v1(X|A)
        lambda x, t: v2_fn(x, t, torch.full((x.size(0), 1), B, device=x.device)), # v2(X|B)
        lambda x, t: v3_fn(x[:, :1], t)                                                         # v3(Z)
    ]
s_fn_list=[
        lambda x, t: s1_fn(x[:, :1], t, torch.full((x.size(0), 1), A, device=x.device)), # s1(X|A)
        lambda x, t: s2_fn(x, t, torch.full((x.size(0), 1), B, device=x.device)), # s2(X|B)
        lambda x, t: s3_fn(x[:, :1], t)                                                         # s3(Z)
    ]
proj_list=[
        lambda z: z[:, :1],    # project to X 
        lambda z: z,           # identity for Z
        lambda z: z[:, :1]     # project to X
    ]
emb_list=[
        lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1),  # embed X→Z
        lambda z: z,                                                                  # identity
        lambda x: torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1)   # embed X→Z
    ]
print("Velocity, score, projection, and embedding functions defined.")

    
for seed in tqdm([0,1,2,3,4]):
    set_seed(seed)

    Bump_Values = [0.0, 10.0, 20.0, 30.0, 50.0, 100.0]
    Ramp_Values = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    Ramp = 0.0
    for Bump in Bump_Values:
        method = "ACE"
        print(f"Running {method} p1p2/p3 with Bump={Bump}, Ramp={Ramp}")
        weight = 1.0

        x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
        samples, logw_final, logw_history, sample_history, resample_history = simulate_ace(
            x0=x0, 
            v_fn_list=v_fn_list, 
            s_fn_list=s_fn_list, 
            proj_list=proj_list, 
            emb_list=emb_list, 
            sigma_fn=sigma_fn,
            v_star= lambda z, t: v2_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
            t0=0.0, t1=1.0, n_steps=1000, device="cuda", ess_threshold=0.7, print_resample_history=True,
            gamma_list=[
                lambda t : torch.tensor(1) + Bump * t * (1 - t) + Ramp * t,
                lambda t : torch.tensor(1 * weight),
                lambda t : torch.tensor(-1 * weight)
            ],
            d_gamma_list=[
                lambda t: torch.zeros_like(t) + Bump * (1 - 2 * t) + Ramp,
                lambda t: torch.zeros_like(t),
                lambda t: torch.zeros_like(t)
            ],
            resample=True
        )
        samples = samples.cpu().numpy()
        plot_diagnostics(samples, logw_final, logw_history, save_name=f"{exp_dir}/{method}_seed{seed}_Bump{Bump}_Ramp{Ramp}_weight{weight}")

        samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
        w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
            torch.tensor(samples_gt), torch.tensor(samples)
        )
        results.append([method, seed, Bump, Ramp, weight, w1, w2, mmd_rbf, total_var])
        print(results)
    ##################################################################################

    Bump = 0.0
    for Ramp in Ramp_Values:
        method = "ACE"
        print(f"Running {method} p1p2/p3 with Bump={Bump}, Ramp={Ramp}")
        weight = 1.0

        x0 = torch.randn(bs, 2).to("cuda")  # (X, Y) sample
        samples, logw_final, logw_history, sample_history, resample_history = simulate_ace(
            x0=x0, 
            v_fn_list=v_fn_list, 
            s_fn_list=s_fn_list, 
            proj_list=proj_list, 
            emb_list=emb_list, 
            sigma_fn=sigma_fn,
            v_star= lambda z, t: v2_fn(z, t, torch.full((z.size(0), 1), B, device=z.device)), 
            t0=0.0, t1=1.0, n_steps=1000, device="cuda", ess_threshold=0.7, print_resample_history=True,
            gamma_list=[
                lambda t : torch.tensor(1) + Bump * t * (1 - t) + Ramp * t,
                lambda t : torch.tensor(1 * weight),
                lambda t : torch.tensor(-1 * weight)
            ],
            d_gamma_list=[
                lambda t: torch.zeros_like(t) + Bump * (1 - 2 * t) + Ramp,
                lambda t: torch.zeros_like(t),
                lambda t: torch.zeros_like(t)
            ],
            resample=True
        )
        samples = samples.cpu().numpy()
        plot_diagnostics(samples, logw_final, logw_history, save_name=f"{exp_dir}/{method}_seed{seed}_Bump{Bump}_Ramp{Ramp}_weight{weight}")

        samples_gt = ground_truth_hcg(bs, cond_A=A, cond_B=B).numpy()
        w1, w2, mmd_rbf, total_var = compute_sample_based_metrics(
            torch.tensor(samples_gt), torch.tensor(samples)
        )
        results.append([method, seed, Bump, Ramp, weight, w1, w2, mmd_rbf, total_var])
        print(results)
    ##################################################################################
    
    df = pd.DataFrame(results, columns=["Method", "seed", "Bump", "Ramp", "weight", "W1", "W2", "MMD", "TV"])
    df.to_csv(f"{exp_dir}/results_checker_onesided.csv", index=False)


result_csv_path = f"{exp_dir}/results_checker_onesided.csv"
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
stats_csv_path = f"{exp_dir}/stats_summary_onesided.csv"
stats_by_method.to_csv(stats_csv_path, index=False)
print(f"Stats summary saved to {stats_csv_path}")