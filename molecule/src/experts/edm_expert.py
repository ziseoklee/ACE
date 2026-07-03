import argparse
import logging
import pickle
from dataclasses import dataclass

import torch
from e3_diffusion_for_molecules.configs.datasets_config import get_dataset_info
from e3_diffusion_for_molecules.equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from e3_diffusion_for_molecules.equivariant_diffusion.utils import remove_mean_with_mask
from e3_diffusion_for_molecules.qm9.models import get_model
from jaxtyping import Bool, Float

from experts import PRETRAINED_MODEL_DIR
from experts.base_expert import MoEExpertABC
from utils.logging_utils import redirect_output_to_logger

EDM_SOURCE_DIR = PRETRAINED_MODEL_DIR / "e3_diffusion_for_molecules"
EDM_CKPT_PATH = EDM_SOURCE_DIR / "outputs" / "edm_qm9" / "generative_model_ema.npy"
EDM_MODEL_CONFIG_PATH = EDM_SOURCE_DIR / "outputs" / "edm_qm9" / "args.pickle"


logger = logging.getLogger(__name__)

DataMask = Bool[torch.Tensor, "data"]  # noqa: F821


@dataclass
class EDMInferenceContext:
    node_mask: Float[torch.Tensor, "B L 1"]
    edge_mask: Float[torch.Tensor, "B_L_L 1"]  # flattened from (B, L, L)
    data_shape: tuple[int, ...]


class EDMExpert(MoEExpertABC):
    """
    Expert class for EDM (Equivariant Diffusion for Molecule Generation in 3D, ICML2022).

    Reference code: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """

    _EDM_MAX_NODES = 29  # Maximum number of nodes in the QM9 dataset

    device: str
    model: EnVariationalDiffusion
    model_config: argparse.Namespace
    _inference_context: EDMInferenceContext

    def __init__(self, device: str, model: EnVariationalDiffusion, model_config: argparse.Namespace):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, device: str):
        logger.info(f"Loading EDM expert from pretrained model at {EDM_CKPT_PATH} and args at {EDM_MODEL_CONFIG_PATH}")

        with open(EDM_MODEL_CONFIG_PATH, "rb") as f:
            edm_config: argparse.Namespace = pickle.load(f)
        if not hasattr(edm_config, "normalization_factor"):
            edm_config.normalization_factor = 1
        if not hasattr(edm_config, "aggregation_method"):
            edm_config.aggregation_method = "sum"

        with redirect_output_to_logger(logger):
            dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)
            model: EnVariationalDiffusion = get_model(edm_config, device, dataset_info, dataloader_train=None)[0]
        ckpt = torch.load(EDM_CKPT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        model.to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("EDM model loaded with %d parameters", n_params)

        instance = cls(device=device, model=model, model_config=edm_config)
        return instance

    def prepare_data(self, batch_size: int, num_nodes: int):
        # Implementation for preparing data specific to EDM
        assert num_nodes <= self._EDM_MAX_NODES, (
            f"num_nodes must be <= {self._EDM_MAX_NODES} as per the EDM model's training configuration."
        )

        node_mask: Float[torch.Tensor, "B L"] = torch.ones(batch_size, num_nodes)

        # Compute edge_mask; only consider edges between existing nodes (excluding self-loops)
        edge_mask: Float[torch.Tensor, "B L L"] = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask_flat: Float[torch.Tensor, "B_L_L 1"] = edge_mask.view(batch_size * num_nodes * num_nodes, 1).to(
            self.device
        )

        node_mask_3d: Float[torch.Tensor, "B L 1"] = node_mask.unsqueeze(2).to(self.device)
        data_shape = (batch_size, num_nodes, self.model.n_dims + self.model.in_node_nf)

        prepared_data = {
            "node_mask": node_mask_3d,
            "edge_mask": edge_mask_flat,
            "data_shape": data_shape,
        }
        # Store the prepared data for inference
        self._inference_context = EDMInferenceContext(**prepared_data)  # pyright: ignore[reportArgumentType]
        return prepared_data

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        # Implementation for scoring using the EDM expert
        model = self.model
        node_mask = self._inference_context.node_mask
        edge_mask = self._inference_context.edge_mask
        data_shape = self._inference_context.data_shape
        curr_shape = x.shape
        batch_size = x.shape[0]

        x = x.reshape(data_shape)
        x[:, :, : model.n_dims] = remove_mean_with_mask(x[:, :, : model.n_dims], node_mask)

        i = (model.T * t).int().clamp(1, model.T - 1)[0].item()

        t = torch.full(size=(1,), fill_value=i, dtype=torch.long, device=self.device)

        t_array = torch.full((batch_size, 1), fill_value=i, device=self.device)
        t_array = t_array / model.T

        gamma_t = model.gamma(t_array)
        sigma_t = model.sigma(gamma_t, target_tensor=x)

        # Neural net prediction.
        eps_t = model.phi(x, t_array, node_mask, edge_mask, context=None)
        score = -eps_t / sigma_t
        score = torch.cat(
            [
                score[:, :, : model.n_dims] - score[:, :, : model.n_dims].mean(dim=1).unsqueeze(1),
                score[:, :, model.n_dims :],
            ],
            dim=2,
        )

        return score.reshape(curr_shape)

    def interleave(
        self,
        x: Float[torch.Tensor, "B data"],
        *args,
        coord_mask: DataMask | None = None,
        num_nodes: int | None = None,
        **kwargs,
    ) -> Float[torch.Tensor, "B data"]:
        """Project EDM-owned coordinates back to the mean-zero subspace."""
        if coord_mask is None:
            return x
        if num_nodes is None:
            raise ValueError("num_nodes must be provided when coord_mask is used for EDM interleave.")

        coord_mask = coord_mask.to(device=x.device)
        expected_num_coord_values = num_nodes * self.model.n_dims
        actual_num_coord_values = int(coord_mask.sum().item())
        if actual_num_coord_values != expected_num_coord_values:
            raise ValueError(
                "EDM coordinate mask size does not match the expected node/coordinate shape: "
                f"{actual_num_coord_values} != {expected_num_coord_values}."
            )

        node_mask = self._inference_context.node_mask.to(device=x.device, dtype=x.dtype)
        if node_mask.shape[1] != num_nodes:
            raise ValueError(f"EDM node_mask has {node_mask.shape[1]} nodes, but coord_mask expects {num_nodes}.")

        x_out = x.detach()
        coords = x_out[..., coord_mask].reshape(x.shape[0], num_nodes, self.model.n_dims)
        coords = remove_mean_with_mask(coords, node_mask)
        x_out[..., coord_mask] = coords.reshape(x.shape[0], -1)
        return x_out

    def postprocess(
        self,
        x: Float[torch.Tensor, "B data"],
        h_mask: DataMask | None = None,
    ) -> Float[torch.Tensor, "B data"]:
        """
        Decode EDM-owned H channels for the current DiffSBDD-EDM MoE layout.

        C/N/O/F channels are decoded by DiffSBDD to avoid double unnormalization.
        EDM's integer feature is nuclear charge, not formal charge, and remains
        ignored by the current molecule construction path.
        """
        if h_mask is None:
            return x

        h_mask = h_mask.to(device=x.device)
        x_out = x.clone()
        x_out[..., h_mask] = x_out[..., h_mask] * self.model.norm_values[1] + self.model.norm_biases[1]
        return x_out
