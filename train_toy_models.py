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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); random.seed(0); np.random.seed(0)
print("device →", device)

experiment_id = "PretrainedToyModels"  # Use a fixed ID for reproducibility
if not os.path.exists(experiment_id):
    os.makedirs(experiment_id)

def train_all_three_models(interpolant, name):
    print(f"Training X|A of {name}")
    u_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device) # X | A
    s_model1 = MLPInstFlexible(z_dim=1, cond_dim=1, width=256, depth=4, output_dim=1).to(device) # X | A
    fm_trainer_custom = FlowMatcher(u_model1, s_model1, interpolant=interpolant)
    fm_trainer_custom.train(sample_data_model1, n_iters=2000, plot_cond_val=1.0) # Using few iterations for demo
    torch.save(u_model1.state_dict(), f"{experiment_id}/u_model1_X_given_A_alpha={name}.pth")
    torch.save(s_model1.state_dict(), f"{experiment_id}/s_model1_X_given_A_alpha={name}.pth")

    print(f"Training X,Y|B of {name}")
    u_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device) # X, Y | B
    s_model2 = MLPInstFlexible(z_dim=2, cond_dim=1, width=256, depth=4, output_dim=2).to(device) # X, Y | B
    fm_trainer_default = FlowMatcher(u_model2, s_model2, interpolant=interpolant)
    fm_trainer_default.train(sample_data_model2, n_iters=10000, plot_cond_val=1.0) # Using few iterations for demo
    torch.save(u_model2.state_dict(), f"{experiment_id}/u_model2_XY_given_B_alpha={name}.pth")
    torch.save(s_model2.state_dict(), f"{experiment_id}/s_model2_XY_given_B_alpha={name}.pth")

    print(f"Training X of {name}")
    u_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device) # X
    s_model3 = MLPInstFlexible(z_dim=1, cond_dim=0, width=256, depth=4, output_dim=1).to(device) # X
    fm_trainer_default = FlowMatcher(u_model3, s_model3, interpolant=interpolant)
    fm_trainer_default.train(sample_data_model3, n_iters=2000) # Using few iterations for demo
    torch.save(u_model3.state_dict(), f"{experiment_id}/u_model3_X_alpha={name}.pth")
    torch.save(s_model3.state_dict(), f"{experiment_id}/s_model3_X_alpha={name}.pth")


def load_interpolants_from_json(path):
    """Load interpolants.json and return a dict of Interpolant objects."""
    with open(path, "r") as f:
        interpolants_raw = json.load(f)

    interpolants = {}
    for name, funcs in interpolants_raw.items():
        alpha_t = eval(funcs["alpha_t"])
        beta_t = eval(funcs["beta_t"])
        d_alpha_t = eval(funcs["d_alpha_t"])
        d_beta_t = eval(funcs["d_beta_t"])
        interpolants[name] = Interpolant(alpha_t, beta_t, d_alpha_t, d_beta_t, name=name)
    return interpolants


interpolant_schedules = load_interpolants_from_json("interpolant_schedules.json")

for name in interpolant_schedules.keys():
    train_all_three_models(interpolant=interpolant_schedules[name], name=name)