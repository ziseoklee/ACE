import copy
import logging
from collections import defaultdict

import torch
import yaml
from easydict import EasyDict
from geodiff.models.epsnet import get_model
from geodiff.models.epsnet.dualenc import DualEncoderEpsNetwork
from geodiff.utils.datasets import rdmol_to_data
from geodiff.utils.misc import repeat_data
from geodiff.utils.transforms import AddHigherOrderEdges, CountNodesPerGraph
from jaxtyping import Float
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose

from src.const import PRETRAINED_MODEL_DIR
from src.experts.base_expert import MoEExpertABC

GEODIFF_SOURCE_DIR = PRETRAINED_MODEL_DIR / "GeoDiff"
GEODIFF_CKPT_PATH = GEODIFF_SOURCE_DIR / "log" / "model" / "checkpoints" / "qm9_default.pt"
GEODIFF_CONFIG_PATH = GEODIFF_SOURCE_DIR / "log" / "model" / "qm9_default.yml"


logger = logging.getLogger(__name__)


def disable_inplace_relu(module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def make_inference_dataset(rdmol_list: list[Chem.Mol], transforms):
    data_list = []
    for rdmol in rdmol_list:
        data_list.append(rdmol_to_data(rdmol))
    # print(f'num_nodes: {data_list[0].num_nodes}')
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
                    order=self.model_config.model.edge_order  # type: ignore
                ),  # Offline edge augmentation;
            ]
        )
        data = make_inference_dataset([fragment_mol], transforms)[0]

        data_input = data.clone()
        batch = repeat_data(data_input, batch_size).to(self.device)
        clip_local = None  # Or 20 if it makes floating point error

        z = torch.randn(batch.num_nodes, 3).to(self.device)
        data_shape = z.shape
        z = z.reshape(batch_size, -1)

        prepared_data = {
            "model": self.model,  # todo; remove this later
            "batch": batch,
            "global_start_sigma": 0.5,
            "w_global": 1.0,
            "clip_local": clip_local,
            "clip": 1000.0,
            "z": z,
            "num_samples": batch_size,
            "data_shape": data_shape,
        }
        return prepared_data

    def score(
        self,
        t: Float[torch.Tensor, " B"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, " B"]:
        # Implementation for scoring using the GeoDiff expert
        ...

    def interleave(self, *args, **kwargs):
        # Implementation for interleaving data specific to GeoDiff
        ...

    def postprocess(self, *args, **kwargs):
        # Implementation for postprocessing results from the GeoDiff expert
        ...


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


if __name__ == "__main__":
    # Example usage of the GeoDiffExpert class
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geodiff_expert = GeoDiffExpert.from_pretrained(device=device)
    logger.info("GeoDiff Expert loaded successfully.")

    processed_data = geodiff_expert.prepare_data(
        batch_size=10,
        fragment_mol=Chem.SDMolSupplier("examples/4m7t_fragment.sdf")[0],
    )
    print("Prepared data keys:", processed_data.keys())
