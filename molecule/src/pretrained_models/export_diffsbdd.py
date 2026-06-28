import contextlib
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from Bio.PDB.PDBParser import PDBParser
from rdkit.Chem import Mol
from torch_scatter import scatter_add, scatter_mean

if TYPE_CHECKING:
    from diffsbdd.lightning_modules import LigandPocketDDPM


_DIFFSBDD_ROOT = Path(__file__).resolve().parent / "DiffSBDD"
_DIFFSBDD_LEGACY_IMPORTS = (
    "analysis",
    "constants",
    "dataset",
    "equivariant_diffusion",
    "geometry_utils",
    "utils",
)


@contextlib.contextmanager
def _diffsbdd_import_context():
    old_path = sys.path[:]
    root = str(_DIFFSBDD_ROOT)
    sys.path = [root] + [path for path in sys.path if path != root]

    saved_modules = {
        name: module
        for legacy_name in _DIFFSBDD_LEGACY_IMPORTS
        for name, module in sys.modules.items()
        if name == legacy_name or name.startswith(f"{legacy_name}.")
    }
    for name in saved_modules:
        del sys.modules[name]

    try:
        yield
    finally:
        for legacy_name in _DIFFSBDD_LEGACY_IMPORTS:
            for name in list(sys.modules):
                if name == legacy_name or name.startswith(f"{legacy_name}."):
                    del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path = old_path


def _load_diffsbdd_helpers():
    with _diffsbdd_import_context():
        molecule_builder = importlib.import_module("diffsbdd.analysis.molecule_builder")
        diff_utils = importlib.import_module("diffsbdd.utils")
    return (
        molecule_builder.build_molecule,
        molecule_builder.process_molecule,
        diff_utils.get_pocket_from_ligand,
        diff_utils.num_nodes_to_batch_mask,
    )


(
    build_molecule,
    process_molecule,
    get_pocket_from_ligand,
    num_nodes_to_batch_mask,
) = _load_diffsbdd_helpers()


def export_diffsbdd(ckpt, device="cpu"):
    # fix_biopython_imports()
    with _diffsbdd_import_context():
        lightning_modules = importlib.import_module("diffsbdd.lightning_modules")
    _LigandPocketDDPM = lightning_modules.LigandPocketDDPM

    model: LigandPocketDDPM = _LigandPocketDDPM.load_from_checkpoint(str(ckpt), map_location=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiffSBDD model loaded with {n_params} parameters")
    return model


def prepare_data(
    model: nn.Module,
    pdb_file: Path,
    ref_ligand: Mol,
    num_samples: int = 10,
    num_nodes: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    pdb_struct = PDBParser(QUIET=True).get_structure("", pdb_file)[0]
    residues = get_pocket_from_ligand(pdb_struct, ref_ligand)
    if isinstance(model, torch.nn.DataParallel):
        pocket = model.module.prepare_pocket(residues, repeats=num_samples)
    else:
        pocket = model.prepare_pocket(residues, repeats=num_samples)
    lig_mask = num_nodes_to_batch_mask(num_samples, num_nodes, device)
    if isinstance(model, torch.nn.DataParallel):
        _, pocket = model.module.ddpm.normalize(pocket=pocket)
    else:
        _, pocket = model.ddpm.normalize(pocket=pocket)
    pocket_com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)
    xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
    mu_lig_x = scatter_mean(pocket["x"], pocket["mask"], dim=0)
    if isinstance(model, torch.nn.DataParallel):
        mu_lig_h = torch.zeros((num_samples, model.module.ddpm.atom_nf), device=device)
    else:
        mu_lig_h = torch.zeros((num_samples, model.ddpm.atom_nf), device=device)
    mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
    sigma = torch.ones_like(pocket["size"]).unsqueeze(1)
    if isinstance(model, torch.nn.DataParallel):
        z_lig, xh_pocket = model.module.ddpm.sample_normal_zero_com(mu_lig, xh0_pocket, sigma, lig_mask, pocket["mask"])
    else:
        z_lig, xh_pocket = model.ddpm.sample_normal_zero_com(mu_lig, xh0_pocket, sigma, lig_mask, pocket["mask"])
    data_shape = z_lig.shape
    z_lig = z_lig.reshape(num_samples, -1)

    prepared_data = {
        "model": model,
        "num_samples": num_samples,
        "lig_mask": lig_mask,
        "pocket": pocket,
        "pocket_com_before": pocket_com_before,
        "device": device,
        "z": z_lig,
        "xh_pocket": xh_pocket,
        "data_shape": data_shape,
    }

    return prepared_data


def score_function(
    t: torch.Tensor,
    z: torch.Tensor,
    prepared_data: dict[str, Any],
    score_scale: float = 1.0,
) -> torch.Tensor:
    model: LigandPocketDDPM = prepared_data["model"]
    num_samples = prepared_data["num_samples"]
    lig_mask = prepared_data["lig_mask"]
    pocket = prepared_data["pocket"]
    device = prepared_data["device"]
    xh_pocket = prepared_data["xh_pocket"]
    data_shape = prepared_data["data_shape"]
    curr_shape = z.shape

    z = z.reshape(data_shape)

    if isinstance(model, torch.nn.DataParallel):
        T = model.module.T
    else:
        T = model.T
    i = (T * t).int().clamp(1, T - 1)[0].item()

    s_array = torch.full((num_samples, 1), fill_value=i - 1, device=device)
    t_array = s_array + 1
    s_array = s_array / T
    t_prev_array = (t_array - 1) / T
    t_array = t_array / T

    # print(f's_array: {s_array.shape}')
    # print(f't_array: {t_array.shape}')
    # print(f'z_lig: {z_lig.shape}')
    # print(f'xh_pocket: {xh_pocket.shape}')
    # print(f'lig_mask: {lig_mask.shape}')
    # print(f'pocket["mask"]: {pocket["mask"].shape}')
    # print(f'num_samples: {num_samples}')
    # print(f'model.ddpm.n_dims: {model.ddpm.n_dims}')

    if isinstance(model, torch.nn.DataParallel):
        gamma_s = model.module.ddpm.gamma(s_array)
        gamma_t = model.module.ddpm.gamma(t_array)
        gamma_t_prev = model.module.ddpm.gamma(t_prev_array)
        sigma_t = model.module.ddpm.sigma(gamma_t, target_tensor=z)
        alpha_t = model.module.ddpm.alpha(gamma_t, target_tensor=z)
        alpha_t_prev = model.module.ddpm.alpha(gamma_t_prev, target_tensor=z)
        beta_t = (-2 * torch.log(alpha_t / alpha_t_prev)) * T
    else:
        gamma_s = model.ddpm.gamma(s_array)
        gamma_t = model.ddpm.gamma(t_array)
        gamma_t_prev = model.ddpm.gamma(t_prev_array)
        sigma_t = model.ddpm.sigma(gamma_t, target_tensor=z)
        alpha_t = model.ddpm.alpha(gamma_t, target_tensor=z)
        alpha_t_prev = model.ddpm.alpha(gamma_t_prev, target_tensor=z)
        beta_t = (-2 * torch.log(alpha_t / alpha_t_prev)) * T
    # score
    # if type(model) == torch.nn.DataParallel:
    #     eps_t_lig, _ = model.module.ddpm.dynamics(
    #         z, xh_pocket, t_array, lig_mask, pocket['mask'])
    # else:
    eps_t_lig, _ = model.ddpm.dynamics(z, xh_pocket, t_array, lig_mask, pocket["mask"])
    score = -eps_t_lig / sigma_t[lig_mask] * score_scale

    return score.reshape(curr_shape)


def interleave_fn(
    z_lig: torch.Tensor,
    choice: np.ndarray,
    prepared_data: dict[str, Any],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    model: LigandPocketDDPM = prepared_data["model"]
    lig_mask = prepared_data["lig_mask"]
    pocket = prepared_data["pocket"]
    # print(f'pocket mask: {pocket["mask"]}')
    prepared_data["xh_pocket"] = (
        prepared_data["xh_pocket"]
        .view(len(choice), -1, prepared_data["xh_pocket"].shape[-1])[choice]
        .reshape(prepared_data["xh_pocket"].shape)
    )
    xh_pocket = prepared_data["xh_pocket"]

    data_shape = prepared_data["data_shape"]
    _z_lig = z_lig.clone().detach()
    if mask is not None:
        z_lig = z_lig[:, mask]
    else:
        z_lig = _z_lig
    curr_shape = z_lig.shape
    z_lig = z_lig.reshape(data_shape)
    # print(f'pocket mask shape: {pocket["mask"].shape}')
    # print(f'lig mask shape: {lig_mask.shape}')
    # print(f'z_lig shape: {z_lig.shape}')
    # print(f'xh_pocket shape: {xh_pocket.shape}')
    # print(f'data_shape shape: {data_shape}')
    # print(f'curr_shape shape: {curr_shape}')
    # print(f'_z_lig shape: {_z_lig.shape}')
    # mean_pocket = scatter_mean(xh_pocket[:, :model.ddpm.n_dims], pocket['mask'], dim=0)[0]
    # mean_lig = scatter_mean(z_lig[:, :model.ddpm.n_dims], lig_mask, dim=0)[0]
    # print(f'mean_pocket: {mean_pocket}')
    # print(f'mean_lig: {mean_lig}')
    # delta_1 = xh_pocket[0, :model.ddpm.n_dims] - scatter_mean(z_lig[:, :model.ddpm.n_dims], pocket['mask'], dim=0)[0]
    # print(f'pos 1: {xh_pocket[0, :model.ddpm.n_dims]}')
    # print(f'delta_1: {delta_1}')
    if isinstance(model, torch.nn.DataParallel):
        (
            z_lig[:, : model.module.ddpm.n_dims],
            xh_pocket[:, : model.module.ddpm.n_dims],
        ) = model.module.ddpm.remove_mean_batch(
            z_lig[:, : model.module.ddpm.n_dims],
            xh_pocket[:, : model.module.ddpm.n_dims],
            lig_mask,
            pocket["mask"],
        )
    else:
        z_lig[:, : model.ddpm.n_dims], xh_pocket[:, : model.ddpm.n_dims] = model.ddpm.remove_mean_batch(
            z_lig[:, : model.ddpm.n_dims],
            xh_pocket[:, : model.ddpm.n_dims],
            lig_mask,  # type: ignore
            pocket["mask"],
        )
    # mean_pocket = scatter_mean(xh_pocket[:, :model.ddpm.n_dims], pocket['mask'], dim=0)[0]
    # mean_lig = scatter_mean(z_lig[:, :model.ddpm.n_dims], lig_mask, dim=0)[0]
    # print(f'mean_pocket: {mean_pocket}')
    # print(f'mean_lig: {mean_lig}')
    # print(f'pos 2: {xh_pocket[0, :model.ddpm.n_dims]}')

    if isinstance(model, torch.nn.DataParallel):
        model.module.ddpm.assert_mean_zero_with_mask(z_lig[:, : model.module.ddpm.n_dims], lig_mask)
    else:
        model.ddpm.assert_mean_zero_with_mask(z_lig[:, : model.ddpm.n_dims], lig_mask)
    # print(f'z_lig: {z_lig.shape}')
    # print(f'mask: {mask.shape}')
    # print(f'curr_shape: {curr_shape}')
    # print(f'_z_lig: {_z_lig.shape}')
    if mask is not None:
        _z_lig[:, mask] = z_lig.reshape(curr_shape)
    else:
        _z_lig = z_lig.reshape(curr_shape)
    return _z_lig


def postprocess_fn(
    z_lig: torch.Tensor,
    prepared_data: dict[str, Any],
    mask: torch.Tensor | None = None,
    frag_atom_type: torch.Tensor | None = None,
) -> list[Mol | None]:
    model: LigandPocketDDPM = prepared_data["model"]
    lig_mask = prepared_data["lig_mask"]
    pocket = prepared_data["pocket"]
    xh_pocket = prepared_data["xh_pocket"]
    num_samples = prepared_data["num_samples"]
    pocket_com_before = prepared_data["pocket_com_before"]
    data_shape = prepared_data["data_shape"]
    _z_lig = z_lig.clone().detach()
    # breakpoint()
    if mask is not None:
        z_lig = z_lig[:, mask]
    else:
        z_lig = _z_lig
    z_lig = z_lig.reshape(data_shape)

    if isinstance(model, torch.nn.DataParallel):
        x_lig, h_lig, x_pocket, h_pocket = model.module.ddpm.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket["mask"], num_samples
        )
    else:
        x_lig, h_lig, x_pocket, h_pocket = model.ddpm.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket["mask"], num_samples
        )

    if isinstance(model, torch.nn.DataParallel):
        model.module.ddpm.assert_mean_zero_with_mask(x_lig, lig_mask)
    else:
        model.ddpm.assert_mean_zero_with_mask(x_lig, lig_mask)

    # Correct CoM drift for examples without intermediate states
    max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
    if max_cog > 5e-2:
        print(f"Warning CoG drift with error {max_cog:.3f}. Projecting the positions down.")
        if isinstance(model, torch.nn.DataParallel):
            x_lig, x_pocket = model.module.ddpm.remove_mean_batch(x_lig, x_pocket, lig_mask, pocket["mask"])
        else:
            x_lig, x_pocket = model.ddpm.remove_mean_batch(
                x_lig,
                x_pocket,
                lig_mask,  # type: ignore
                pocket["mask"],
            )

    # Overwrite last frame with the resulting x and h.
    xh_lig = torch.cat([x_lig, h_lig], dim=1)
    xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)

    # Postprocess denoised result to output object
    pocket_com_after = scatter_mean(xh_pocket[:, : model.x_dims], pocket["mask"], dim=0)

    xh_pocket[:, : model.x_dims] += (pocket_com_before - pocket_com_after)[pocket["mask"]]
    xh_lig[:, : model.x_dims] += (pocket_com_before - pocket_com_after)[lig_mask]

    # Build mol objects
    x = xh_lig[:, : model.x_dims].detach().cpu()
    atom_type = xh_lig[:, model.x_dims :].argmax(1).detach().cpu()

    # print(f'atom_type shape: {atom_type.shape}')
    # if frag_atom_type is not None:
    #     atom_type[:len(frag_atom_type)] = frag_atom_type

    lig_mask = lig_mask.cpu()

    molecules = []
    x_by_sample = _batch_to_list_preserve_atom_order(x, lig_mask)
    atom_type_by_sample = _batch_to_list_preserve_atom_order(atom_type, lig_mask)
    for mol_pc in zip(x_by_sample, atom_type_by_sample):
        if frag_atom_type is not None:
            mol_pc[1][: len(frag_atom_type)] = frag_atom_type
        mol = build_molecule(*mol_pc, model.dataset_info, add_coords=True)
        if mol is None:
            molecules.append(None)
            continue
        mol = process_molecule(mol, add_hydrogens=False, sanitize=False, relax_iter=0, largest_frag=False)
        if mol is not None:
            molecules.append(mol)

    return molecules


def _batch_to_list_preserve_atom_order(data: torch.Tensor, batch_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split a flattened ligand batch without reordering atoms inside each sample."""
    if batch_mask.device != data.device:
        batch_mask = batch_mask.to(data.device)
    sample_ids = torch.unique(batch_mask, sorted=True)
    return tuple(data[batch_mask == sample_id] for sample_id in sample_ids)
