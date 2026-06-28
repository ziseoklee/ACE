import argparse
import contextlib
import importlib
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from rdkit.Chem import Mol

if TYPE_CHECKING:
    from e3_diffusion_for_molecules.equivariant_diffusion.en_diffusion import EnVariationalDiffusion


_E3_ROOT = Path(__file__).resolve().parent / "e3_diffusion_for_molecules"
_E3_LEGACY_IMPORTS = (
    "configs",
    "egnn",
    "equivariant_diffusion",
    "qm9",
    "utils",
)


@contextlib.contextmanager
def _e3_import_context():
    old_path = sys.path[:]
    root = str(_E3_ROOT)
    sys.path = [root] + [path for path in sys.path if path != root]

    saved_modules = {
        name: module
        for legacy_name in _E3_LEGACY_IMPORTS
        for name, module in sys.modules.items()
        if name == legacy_name or name.startswith(f"{legacy_name}.")
    }
    for name in saved_modules:
        del sys.modules[name]

    try:
        yield
    finally:
        for legacy_name in _E3_LEGACY_IMPORTS:
            for name in list(sys.modules):
                if name == legacy_name or name.startswith(f"{legacy_name}."):
                    del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path = old_path


def _load_e3_helpers():
    with _e3_import_context():
        datasets_config = importlib.import_module("e3_diffusion_for_molecules.configs.datasets_config")
        diffusion_utils = importlib.import_module("e3_diffusion_for_molecules.equivariant_diffusion.utils")
        qm9_models = importlib.import_module("e3_diffusion_for_molecules.qm9.models")
        qm9_visualizer = importlib.import_module("e3_diffusion_for_molecules.qm9.visualizer")
    return (
        datasets_config.get_dataset_info,
        diffusion_utils.assert_correctly_masked,
        diffusion_utils.assert_mean_zero_with_mask,
        diffusion_utils.remove_mean_with_mask,
        qm9_models.get_model,
        qm9_visualizer.save_xyz_file,
    )


(
    get_dataset_info,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    get_model,
    save_xyz_file,
) = _load_e3_helpers()


def export_edm(ckpt_path: Path, args_path: Path, device="cpu"):
    """Instantiate the EDM model and load checkpoint weights."""
    with open(args_path, "rb") as f:
        args: argparse.Namespace = pickle.load(f)

    device = torch.device(device)
    if not hasattr(args, "normalization_factor"):
        args.normalization_factor = 1
    if not hasattr(args, "aggregation_method"):
        args.aggregation_method = "sum"
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    model: EnVariationalDiffusion = get_model(args, device, dataset_info, dataloader_train=None)[0]
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"e3 Diffusion model loaded with {n_params} parameters")
    model.to(device)
    return args, model


def prepare_data(
    model: nn.Module,
    num_samples: int = 10,
    num_nodes: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    batch_size = num_samples
    max_n_nodes = 29

    nodesxsample = (torch.ones(batch_size, device=device) * num_nodes).long()

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, num_nodes)
    for i in range(batch_size):
        node_mask[i, 0 : nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * num_nodes * num_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    context = None

    if isinstance(model, torch.nn.DataParallel):
        z = model.module.sample_combined_position_feature_noise(batch_size, num_nodes, node_mask)
    else:
        z = model.sample_combined_position_feature_noise(batch_size, num_nodes, node_mask)
    # print(f'z.device: {z.device}')
    if isinstance(model, torch.nn.DataParallel):
        n_dims = model.module.n_dims
    else:
        n_dims = model.n_dims
    assert_mean_zero_with_mask(z[:, :, :n_dims], node_mask)
    data_shape = z.shape
    z = z.reshape(num_samples, -1)

    prepared_data = {
        "model": model,
        "batch_size": batch_size,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "context": context,
        "max_n_nodes": max_n_nodes,
        "device": device,
        "z": z,
        "data_shape": data_shape,
    }

    return prepared_data


def score_function(
    t: torch.Tensor,
    z: torch.Tensor,
    prepared_data: dict[str, Any],
    score_scale: float = 1.0,
) -> torch.Tensor:
    model: EnVariationalDiffusion = prepared_data["model"]
    batch_size = prepared_data["batch_size"]
    node_mask = prepared_data["node_mask"]
    edge_mask = prepared_data["edge_mask"]
    context = prepared_data["context"]
    max_n_nodes = prepared_data["max_n_nodes"]
    device = prepared_data["device"]
    data_shape = prepared_data["data_shape"]
    curr_shape = z.shape
    # print(f'data_shape: {data_shape}')
    # print(f'curr_shape: {curr_shape}')
    z = z.reshape(data_shape)
    z[:, :, : model.n_dims] = remove_mean_with_mask(z[:, :, : model.n_dims], node_mask)

    i = (model.T * t).int().clamp(1, model.T - 1)[0].item()

    t = torch.full(size=(1,), fill_value=i, dtype=torch.long, device=device)

    s_array = torch.full((batch_size, 1), fill_value=i - 1, device=device)
    t_array = torch.full((batch_size, 1), fill_value=i, device=device)
    s_array = s_array / model.T
    t_prev_array = (t_array - 1) / model.T
    t_array = t_array / model.T

    # z = model.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, None)

    gamma_s = model.gamma(s_array)
    gamma_t = model.gamma(t_array)
    gamma_t_prev = model.gamma(t_prev_array)
    sigma_t = model.sigma(gamma_t, target_tensor=z)
    alpha_t = model.alpha(gamma_t, z)
    alpha_t_prev = model.alpha(gamma_t_prev, z)
    beta_t = -2 * (torch.log(alpha_t / alpha_t_prev) * model.T)

    # print(f'beta_t : {beta_t[0][0]}')
    # print(f'alpha_t : {alpha_t[0][0]}')
    # print(f'alpha_t_prev : {alpha_t_prev[0][0]}')
    # print(f'gamma_t : {gamma_t[0][0]}')
    # print(f'gamma_t_prev : {gamma_t_prev[0][0]}')
    # print(f'sigma_t : {sigma_t[0][0]}')
    # print(f'alpha_s : {alpha_s[0][0]}')
    # print(f'gamma_s : {gamma_s[0][0]}')

    # Neural net prediction.
    eps_t = model.phi(z, t_array, node_mask, edge_mask, context)
    score = -eps_t / sigma_t * score_scale
    score = score.reshape(batch_size, -1, 9)
    # print(f'score shape: {score.shape}')
    # score[:, :, :model.n_dims] = score[:, :, :model.n_dims] - score[:, :, :model.n_dims].mean(dim=1).unsqueeze(1)
    score = torch.cat(
        [
            score[:, :, : model.n_dims] - score[:, :, : model.n_dims].mean(dim=1).unsqueeze(1),
            score[:, :, model.n_dims :],
        ],
        dim=2,
    )
    # print(f'centeralized score_edm2')
    return score.reshape(curr_shape)


def encode_xh(
    args: argparse.Namespace,
    model: nn.Module,
    mol: Mol,
    device="cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    assert args.dataset == "qm9", "only qm9 is supported currently"
    num_atoms = mol.GetNumAtoms()
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    h_cat = torch.zeros(num_atoms, len(dataset_info["atom_decoder"]), device=device)
    h_int = torch.zeros(num_atoms, 1, device=device)
    for i in range(num_atoms):
        h_cat[i, dataset_info["atom_encoder"][mol.GetAtomWithIdx(i).GetSymbol()]] = 1
        if args.include_charges:
            h_int[i][0] = mol.GetAtomWithIdx(i).GetAtomicNum()
    h = {"categorical": h_cat[None], "integer": h_int[None]}

    x = torch.tensor(mol.GetConformer().GetPositions(), device=device)
    node_mask = torch.ones(num_atoms, device=device)[:, None]

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # we assume isinstance(model, EnVariationalDiffusion) is true
    x, h, _ = model.normalize(x[None], h, node_mask[None])
    h = torch.cat([h["categorical"][0], h["integer"][0]], dim=1)
    return x[0], h
