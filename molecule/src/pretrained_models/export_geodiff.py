import contextlib
import copy
import importlib
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from rdkit.Chem import Mol
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose

if TYPE_CHECKING:
    from pretrained_models.GeoDiff.models.epsnet import get_model
    from pretrained_models.GeoDiff.models.epsnet.dualenc import DualEncoderEpsNetwork, clip_norm
    from pretrained_models.GeoDiff.models.geometry import eq_transform
    from pretrained_models.GeoDiff.utils.datasets import rdmol_to_data
    from pretrained_models.GeoDiff.utils.misc import repeat_data
    from pretrained_models.GeoDiff.utils.transforms import AddHigherOrderEdges, CountNodesPerGraph


_GEODIFF_ROOT = Path(__file__).resolve().parent / "GeoDiff"
_GEODIFF_LEGACY_IMPORTS = ("models", "utils")


@contextlib.contextmanager
def _geodiff_import_context():
    old_path = sys.path[:]
    root = str(_GEODIFF_ROOT)
    sys.path = [root] + [path for path in sys.path if path != root]

    saved_modules = {
        name: module
        for legacy_name in _GEODIFF_LEGACY_IMPORTS
        for name, module in sys.modules.items()
        if name == legacy_name or name.startswith(f"{legacy_name}.")
    }
    for name in saved_modules:
        del sys.modules[name]

    try:
        yield
    finally:
        for legacy_name in _GEODIFF_LEGACY_IMPORTS:
            for name in list(sys.modules):
                if name == legacy_name or name.startswith(f"{legacy_name}."):
                    del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path = old_path


def _public_symbols(module):
    return {name: getattr(module, name) for name in dir(module) if not name.startswith("_")}


def _load_geodiff_helpers():
    with _geodiff_import_context():
        epsnet = importlib.import_module("GeoDiff.models.epsnet")
        dualenc = importlib.import_module("GeoDiff.models.epsnet.dualenc")
        geometry = importlib.import_module("GeoDiff.models.geometry")
        datasets = importlib.import_module("GeoDiff.utils.datasets")
        misc = importlib.import_module("GeoDiff.utils.misc")
        transforms = importlib.import_module("GeoDiff.utils.transforms")

    symbols = {}
    for module in (epsnet, datasets, misc, transforms):
        symbols.update(_public_symbols(module))
    symbols["clip_norm"] = dualenc.clip_norm
    symbols["eq_transform"] = geometry.eq_transform
    symbols["get_model"] = epsnet.get_model
    return symbols


globals().update(_load_geodiff_helpers())


class ConformationDatasetFromDataList(Dataset):
    def __init__(self, data_list, transform=None):
        super().__init__()
        self.data = data_list
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):
        data = Data(**self.data[idx].__dict__)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)


class PackedConformationDatasetFromDataList(ConformationDatasetFromDataList):
    def __init__(self, data_list, transform=None):
        super().__init__(data_list, transform)
        # k:v = idx: data_obj
        self._pack_data_by_mol()

    def _pack_data_by_mol(self):
        """
        pack confs with same mol into a single data object
        """
        self._packed_data = defaultdict(list)
        if hasattr(self.data, "idx"):
            for i in range(len(self.data)):
                self._packed_data[self.data[i].idx.item()].append(self.data[i])
        else:
            for i in range(len(self.data)):
                self._packed_data[self.data[i].smiles].append(self.data[i])
        print(f"[Packed] {len(self._packed_data)} Molecules, {len(self.data)} Conformations.")

        new_data = []
        # logic
        # save graph structure for each mol once, but store all confs
        for k, v in self._packed_data.items():
            # breakpoint()
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                all_pos.append(v[i].pos)
            data.pos_ref = torch.cat(all_pos, 0)  # (num_conf*num_node, 3)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            # del data.pos

            if hasattr(data, "totalenergy"):
                del data.totalenergy
            if hasattr(data, "boltzmannweight"):
                del data.boltzmannweight
            new_data.append(data)
        self.new_data = new_data

    def __getitem__(self, idx):
        num_nodes = self.data[idx].num_nodes
        data = Data(**self.data[idx].to_dict())
        # data.num_nodes = num_nodes
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.new_data)


def disable_inplace_relu(module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def export_geodiff(ckpt, device="cpu"):
    cwd = os.getcwd()
    os.chdir(_GEODIFF_ROOT)
    try:
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        model = get_model(ckpt_data["config"].model)
        model.load_state_dict(ckpt_data["model"])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"GeoDiff model loaded with {n_params} parameters")
    finally:
        os.chdir(cwd)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(ckpt)), "*.yml"))[0]
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))
    disable_inplace_relu(model)

    return config, model


def make_inferernce_dataset(rdmol_list: List[Mol], transforms):
    data_list = []
    for rdmol in rdmol_list:
        data_list.append(rdmol_to_data(rdmol))
    # print(f'num_nodes: {data_list[0].num_nodes}')
    dataset = PackedConformationDatasetFromDataList(data_list, transform=transforms)
    return dataset


def prepare_data(
    config: EasyDict,
    model: nn.Module,
    mol: Mol,
    num_samples: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    global_start_sigma = 0.5
    w_global = 1.0
    clip = 1000.0
    # Datasets and loaders
    transforms = Compose(
        [
            CountNodesPerGraph(),
            AddHigherOrderEdges(order=config.model.edge_order),  # Offline edge augmentation
        ]
    )
    test_set = make_inferernce_dataset([mol], transforms)
    data = test_set[0]
    # print(f'num_nodes: {data.num_nodes}')

    data_input = data.clone()
    batch = repeat_data(data_input, num_samples).to(device)
    clip_local = None  # Or 20 if it makes floating point error

    z = torch.randn(batch.num_nodes, 3).to(device)
    data_shape = z.shape
    z = z.reshape(num_samples, -1)

    prepared_data = {
        "model": model,
        "batch": batch,
        "global_start_sigma": global_start_sigma,
        "w_global": w_global,
        "clip_local": clip_local,
        "clip": clip,
        "z": z,
        "num_samples": num_samples,
        "data_shape": data_shape,
    }

    return prepared_data


def score_function(
    t: torch.Tensor,
    z: torch.Tensor,
    prepared_data: Dict[str, Any],
    score_scale: float = 1.0,
) -> torch.Tensor:
    model: DualEncoderEpsNetwork = prepared_data["model"]
    batch = prepared_data["batch"]
    global_start_sigma = prepared_data["global_start_sigma"]
    w_global = prepared_data["w_global"]
    clip_local = prepared_data["clip_local"]
    clip = prepared_data["clip"]
    data_shape = prepared_data["data_shape"]
    num_samples = prepared_data["num_samples"]
    curr_shape = z.shape
    z = z.reshape(data_shape)
    if isinstance(model, torch.nn.DataParallel):
        alphas = model.module.alphas
        betas = model.module.betas * 10000
    else:
        alphas = model.alphas
        betas = model.betas * 10000
    sigmas = (1.0 - alphas).sqrt() / alphas.sqrt()

    if isinstance(model, torch.nn.DataParallel):
        num_timesteps = model.module.num_timesteps
    else:
        num_timesteps = model.num_timesteps
    i = (num_timesteps * t).int().clamp(0, num_timesteps - 1)[0].item()

    # print(f'i: {i} alpha: {alphas[i]} beta: {betas[i]} sigma: {sigmas[i]}')

    t = torch.full(size=(1,), fill_value=i, dtype=torch.long, device=z.device)

    # print(f'pos requires_grad: {pos.requires_grad}')
    (
        edge_inv_global,
        edge_inv_local,
        edge_index,
        edge_type,
        edge_length,
        local_edge_mask,
    ) = model(
        atom_type=batch.atom_type,
        pos=z / alphas[i].sqrt(),
        bond_index=batch.edge_index,
        bond_type=batch.edge_type,
        batch=batch.batch,
        time_step=t,
        return_edges=True,
        extend_order=False,
        extend_radius=True,
        is_sidechain=None,
    )  # (E_global, 1), (E_local, 1)

    # Local
    node_eq_local = eq_transform(
        edge_inv_local,
        z / alphas[i].sqrt(),
        edge_index[:, local_edge_mask],
        edge_length[local_edge_mask],
    )
    if clip_local is not None:
        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
    # Global
    if sigmas[i] < global_start_sigma:
        edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1).float())
        node_eq_global = eq_transform(edge_inv_global, z / alphas[i].sqrt(), edge_index, edge_length)
        node_eq_global = clip_norm(node_eq_global, limit=clip)
    else:
        node_eq_global = 0
    # Sum
    eps_pos = node_eq_local + node_eq_global * w_global  # + eps_pos_reg * w_reg

    score = eps_pos / (1 - torch.ones_like(eps_pos) * alphas[i]).sqrt() * score_scale
    score = score.reshape(num_samples, -1, 3)
    score = score - score.mean(dim=1).unsqueeze(1)
    # print(f'centeralized score_geo')
    return score.reshape(curr_shape)
