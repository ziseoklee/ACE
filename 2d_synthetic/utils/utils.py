import torch
import matplotlib.pyplot as plt
import sys
import contextlib, importlib, os, sys
import pandas as pd
from pathlib import Path
import subprocess

@contextlib.contextmanager
def sandbox_import_dir(dir_path: str):
    """
    Temporarily prefer `dir_path` for imports and avoid reusing
    same-named modules from sys.modules. Restores everything on exit.
    """
    dir_path = os.fspath(dir_path)
    # Detect module base names present in the directory (to guard)
    guard = {Path(p).stem for p in os.listdir(dir_path)}
    # print(guard)
    # Stash any conflicting modules already loaded
    # print(f'utils in sys.modules: {sys.modules["utils"]}')
    stash = {name: sys.modules.pop(name)
             for name in list(sys.modules)
             if name in guard or any(name.split(".")[0] == g for g in guard)}
    # print(f'stash: {stash}')
    sys.path.insert(0, dir_path)
    cwd = os.getcwd()
    os.chdir(dir_path)
    sys.path = list(filter(lambda x: x != cwd, sys.path))
    # sys.path.remove(cwd)
    # import utils
    # print(f'utils in sys.modules: {sys.modules["utils"]}')
    importlib.invalidate_caches()
    before = set(sys.modules)

    try:
        yield
    finally:
        # Remove anything imported from this dir during the sandbox
        for name in list(sys.modules):
            if name not in before:
                sys.modules.pop(name, None)
        # assert 'utils' not in sys.modules
        # Restore stashed modules and sys.path
        os.chdir(cwd)
        sys.path.remove(dir_path)
        sys.path.insert(0, cwd)
        sys.modules.update(stash)
        # assert 'utils' in sys.modules
        importlib.invalidate_caches()


### padding tensor ###
def pad_tensor(x, pad_size, dim=0):
    padding_shape = list(x.shape)
    padding_shape[dim] = pad_size - x.shape[dim]
    return torch.cat([x, torch.zeros(padding_shape, device=x.device)], dim=dim)

def pad_tensor_list(x_list, pad_size, dim=0):
    return [pad_tensor(x, pad_size, dim) for x in x_list]


### plot any number of samples ###
def plot_samples(samples_dict):
    for key, samples in samples_dict.items():
        plt.scatter(samples[:, 0], samples[:, 1], label=key)
    plt.legend()
    plt.show()





REPOS = {
    "DiffSBDD": "https://github.com/arneschneuing/DiffSBDD.git",
    "GeoDiff": "https://github.com/MinkaiXu/GeoDiff.git",
    "e3_diffusion_for_molecules": "https://github.com/ehoogeboom/e3_diffusion_for_molecules.git",
}


def ensure_repo(name: str, url: str) -> Path:
    """Clone repo into baseline/ if it does not already exist."""
    root = Path(f'{__file__}').parent / 'src' / 'pretrained_models'
    path = root / name
    if not path.exists():
        subprocess.run(["git", "clone", url, str(path)], check=True)
    return path


def load_diffsbsd() -> str:
    """Instantiate the DiffSBDD model and load checkpoint weights."""
    path = ensure_repo("DiffSBDD", REPOS["DiffSBDD"])
    ckpt = path / "checkpoints" / "crossdocked_ca_cond.ckpt"
    with sandbox_import_dir(path):
        from export import export_diffsbsd
        model = export_diffsbsd(ckpt)
    return model


def load_geodiff() -> str:
    """Instantiate the GeoDiff model and load checkpoint weights."""
    path = ensure_repo("GeoDiff", REPOS["GeoDiff"])
    ckpt = path / "log" / "model" / "checkpoints" / "qm9_default.pt"
    with sandbox_import_dir(path):
        from export import export_geodiff
        config, model = export_geodiff(ckpt)
    return config, model

def load_e3_diffusion() -> str:
    """Instantiate the EDM model and load checkpoint weights."""
    path = ensure_repo("e3_diffusion_for_molecules", REPOS["e3_diffusion_for_molecules"])
    ckpt = path / "outputs" / "edm_geom_qm9" / "generative_model_ema.npy"
    args = path / "outputs" / "edm_geom_qm9" / "args.pickle"
    with sandbox_import_dir(path):
        from export import export_edm
        model = export_edm(ckpt, args)
    return model


# DataFrame Utilities
def df_to_latex(df, save_path):
    latex = df.to_latex(
        multicolumn=True,     # combine top-level column headers
        multirow=True,        # combine repeated row index labels
        index=True,           # include row MultiIndex
        bold_rows=False,      # set True if you want bold index labels
        na_rep="",
        escape=True,          # set False if your labels include LaTeX
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex)

def df_to_csv(df, save_path):
    df.to_csv(save_path, index=True)    

def df_to_excel(df, save_path, sheet_name="Sheet1"):
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=True,          # include row MultiIndex as left columns
            merge_cells=True,    # merge repeated headers (default True)
            na_rep=""
        )