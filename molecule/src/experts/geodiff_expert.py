import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

import torch
import yaml
from easydict import EasyDict
from geodiff.models.epsnet import get_model
from geodiff.models.epsnet.dualenc import DualEncoderEpsNetwork, clip_norm
from geodiff.models.geometry import eq_transform
from geodiff.utils.datasets import rdmol_to_data
from geodiff.utils.misc import repeat_data
from geodiff.utils.transforms import AddHigherOrderEdges, CountNodesPerGraph
from jaxtyping import Float
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose

from experts import PRETRAINED_MODEL_DIR
from experts.base_expert import MoEExpertABC

GEODIFF_SOURCE_DIR = PRETRAINED_MODEL_DIR / "GeoDiff"
GEODIFF_CKPT_PATH = GEODIFF_SOURCE_DIR / "log" / "model" / "checkpoints" / "qm9_default.pt"
GEODIFF_CONFIG_PATH = GEODIFF_SOURCE_DIR / "log" / "model" / "qm9_default.yml"


logger = logging.getLogger(__name__)


@dataclass
class GeoDiffInferenceContext:
    batch: Data
    global_start_sigma: float
    w_global: float
    clip_local: float | None
    clip: float
    data_shape: tuple[int, ...]


def disable_inplace_relu(module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def make_inference_dataset(rdmol_list: list[Chem.Mol], transforms):
    data_list = []
    for rdmol in rdmol_list:
        data_list.append(rdmol_to_data(rdmol))
    dataset = PackedConformationDatasetFromDataList(data_list, transform=transforms)
    return dataset


class GeoDiffExpert(MoEExpertABC):
    """
    Expert class for GeoDiff (GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation, ICLR2022).

    Reference code: https://github.com/MinkaiXu/GeoDiff
    """

    device: str
    model: DualEncoderEpsNetwork
    model_config: EasyDict
    _inference_context: GeoDiffInferenceContext

    def __init__(self, device: str, model: DualEncoderEpsNetwork, model_config: EasyDict):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, device: str):
        logger.info(
            f"Loading GeoDiff expert from pretrained model at {GEODIFF_CKPT_PATH} and args at {GEODIFF_CONFIG_PATH}"
        )

        ckpt = torch.load(GEODIFF_CKPT_PATH, map_location=device, weights_only=False)
        model = get_model(ckpt["config"].model)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("GeoDiff model loaded with %d parameters", n_params)

        with open(GEODIFF_CONFIG_PATH) as f:
            geodiff_config = EasyDict(yaml.safe_load(f))
        disable_inplace_relu(model)

        instance = cls(device=device, model=model, model_config=geodiff_config)
        return instance

    def prepare_data(self, batch_size: int, fragment_mol: Chem.Mol):
        # Implementation for preparing data specific to GeoDiff
        transforms = Compose(
            [
                CountNodesPerGraph(),
                AddHigherOrderEdges(
                    order=self.model_config.model.edge_order  # pyright: ignore[reportAttributeAccessIssue]
                ),  # Offline edge augmentation;
            ]
        )
        data = make_inference_dataset([fragment_mol], transforms)[0]

        data_input = data.clone()
        batch = repeat_data(data_input, batch_size).to(self.device)  # pyright: ignore[reportAttributeAccessIssue]
        data_shape = (batch.num_nodes, 3)

        prepared_data = {
            "batch": batch,
            "global_start_sigma": 0.5,
            "w_global": 1.0,
            "clip_local": None,  # Or 20 if it makes floating point error,
            "clip": 1000.0,
            "data_shape": data_shape,
        }
        # Store the prepared data for inference
        self._inference_context = GeoDiffInferenceContext(**prepared_data)  # pyright: ignore[reportArgumentType]
        return prepared_data

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        # Implementation for scoring using the GeoDiff expert
        model = self.model
        batch = self._inference_context.batch
        global_start_sigma = self._inference_context.global_start_sigma
        w_global = self._inference_context.w_global
        clip_local = self._inference_context.clip_local
        clip = self._inference_context.clip
        data_shape = self._inference_context.data_shape
        curr_shape = x.shape
        batch_size = x.shape[0]

        x = x.reshape(data_shape)
        alphas: Float[torch.Tensor, "T"] = cast(torch.Tensor, model.alphas)  # noqa: UP037,F821
        sigmas = (1.0 - alphas).sqrt() / alphas.sqrt()

        num_timesteps = int(model.num_timesteps)
        timestep = int((num_timesteps * t).long().clamp(0, num_timesteps - 1)[0, 0].item())
        alpha_t = alphas[timestep]
        sigma_t = sigmas[timestep]

        time_step = torch.full(size=(1,), fill_value=timestep, dtype=torch.long, device=x.device)
        scaled_pos = x / alpha_t.sqrt()

        (
            edge_inv_global,
            edge_inv_local,
            edge_index,
            edge_type,
            edge_length,
            local_edge_mask,
        ) = model(
            atom_type=batch.atom_type,
            pos=scaled_pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            time_step=time_step,
            return_edges=True,
            extend_order=False,
            extend_radius=True,
            is_sidechain=None,
        )  # (E_global, 1), (E_local, 1)

        # Local
        node_eq_local = eq_transform(
            edge_inv_local,
            scaled_pos,
            edge_index[:, local_edge_mask],
            edge_length[local_edge_mask],
        )
        if clip_local is not None:
            node_eq_local = clip_norm(node_eq_local, limit=clip_local)
        # Global
        if float(sigma_t.item()) < global_start_sigma:
            edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1).float())
            node_eq_global = eq_transform(edge_inv_global, scaled_pos, edge_index, edge_length)
            node_eq_global = clip_norm(node_eq_global, limit=clip)
        else:
            node_eq_global = 0
        # Sum
        eps_pos = node_eq_local + node_eq_global * w_global  # + eps_pos_reg * w_reg

        score = eps_pos / (1 - torch.ones_like(eps_pos) * alpha_t).sqrt()
        score = score.reshape(batch_size, -1, 3)
        score = score - score.mean(dim=1).unsqueeze(1)
        return score.reshape(curr_shape)

    def interleave(self, x: Float[torch.Tensor, "B data"], *args, **kwargs) -> Float[torch.Tensor, "B data"]:
        # Implementation for interleaving data specific to GeoDiff; no-op
        return x

    def postprocess(self, x: Float[torch.Tensor, "B data"], *args, **kwargs) -> Float[torch.Tensor, "B data"]:
        # Implementation for postprocessing results from the GeoDiff expert; no-op
        return x


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
        logger.info(f"[Packed] {len(self._packed_data)} Molecules, {len(self.data)} Conformations.")

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
        data = Data(**self.data[idx].to_dict())
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.new_data)
