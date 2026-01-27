import json
import torch
from ace_lib.interpolant import Interpolant
import random
import numpy as np
import os
import re
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")

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


def get_experiment_dir(root: str, *sub_paths) -> Path:
    """
    Creates and returns a Path object for an experiment directory.
    
    Args:
        root (str): The top-level directory under which to create the experiment folder.
        *sub_paths: Any number of sub-folders or configuration lists.
                    Lists/tuples will be joined by underscores.
    
    Returns:
        Path: The absolute path to the created directory.
    """
    processed_subs = []
    for item in sub_paths:
        if isinstance(item, (list, tuple)):
            # Convert ["a", "b"] -> "a_b" to avoid brackets in filenames
            clean_name = "_".join(str(x) for x in item)
        else:
            clean_name = str(item)
        clean_name = re.sub(r'[\\/*?:"<>|]', "", clean_name)
        processed_subs.append(clean_name)

    full_path = Path(root).joinpath(*processed_subs)
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path