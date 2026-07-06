import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
EDM_PRETRAINED_QM9: Final = "qm9"
EDM_PRETRAINED_GEOM_DRUG: Final = "geom_drug"


logger = logging.getLogger(__name__)

DataMask = Bool[torch.Tensor, "data"]  # noqa: F821


@dataclass(frozen=True)
class EDMPretrainedSpec:
    name: str
    output_dir: Path

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "generative_model_ema.npy"

    @property
    def model_config_path(self) -> Path:
        return self.output_dir / "args.pickle"


EDM_PRETRAINED_SPECS: Final[dict[str, EDMPretrainedSpec]] = {
    EDM_PRETRAINED_QM9: EDMPretrainedSpec(
        name="EDM_QM9",
        output_dir=EDM_SOURCE_DIR / "outputs" / "edm_qm9",
    ),
    EDM_PRETRAINED_GEOM_DRUG: EDMPretrainedSpec(
        name="EDM_GEOM_DRUG",
        output_dir=EDM_SOURCE_DIR / "outputs" / "edm_geom_drugs",
    ),
}


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

    device: str
    model: EnVariationalDiffusion
    model_config: argparse.Namespace
    dataset_info: dict
    max_nodes: int
    _inference_context: EDMInferenceContext

    def __init__(
        self,
        device: str,
        model: EnVariationalDiffusion,
        model_config: argparse.Namespace,
        dataset_info: dict,
    ):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.model_config = model_config
        self.dataset_info = dataset_info
        self.max_nodes = int(dataset_info["max_n_nodes"])

    @classmethod
    def from_pretrained(cls, device: str, pretrained_model: str = EDM_PRETRAINED_QM9):
        try:
            pretrained_spec = EDM_PRETRAINED_SPECS[pretrained_model]
        except KeyError as exc:
            supported = ", ".join(EDM_PRETRAINED_SPECS)
            raise ValueError(f"Unsupported EDM pretrained model {pretrained_model!r}. Supported: {supported}.") from exc

        logger.info(
            "Loading %s expert from pretrained model at %s and args at %s",
            pretrained_spec.name,
            pretrained_spec.checkpoint_path,
            pretrained_spec.model_config_path,
        )

        with open(pretrained_spec.model_config_path, "rb") as f:
            edm_config: argparse.Namespace = pickle.load(f)
        if not hasattr(edm_config, "normalization_factor"):
            edm_config.normalization_factor = 1
        if not hasattr(edm_config, "aggregation_method"):
            edm_config.aggregation_method = "sum"

        with redirect_output_to_logger(logger):
            dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)
            model: EnVariationalDiffusion = get_model(edm_config, device, dataset_info, dataloader_train=None)[0]
        ckpt = torch.load(pretrained_spec.checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        model.to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "%s model loaded with %d parameters; max_nodes=%d, include_charges=%s",
            pretrained_spec.name,
            n_params,
            int(dataset_info["max_n_nodes"]),
            edm_config.include_charges,
        )

        instance = cls(device=device, model=model, model_config=edm_config, dataset_info=dataset_info)
        return instance

    def prepare_data(self, batch_size: int, num_nodes: int):
        # Implementation for preparing data specific to EDM
        assert num_nodes <= self.max_nodes, (
            f"num_nodes must be <= {self.max_nodes} as per the EDM model's training configuration."
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
        categorical_mask: DataMask | None = None,
    ) -> Float[torch.Tensor, "B data"]:
        """
        Decode selected EDM-owned categorical channels from latent scale.

        EDM's integer feature is nuclear charge, not formal charge, and remains ignored
        by the current molecule construction path.
        """
        if categorical_mask is None:
            return x

        categorical_mask = categorical_mask.to(device=x.device)
        x_out = x.clone()
        x_out[..., categorical_mask] = (
            x_out[..., categorical_mask] * self.model.norm_values[1] + self.model.norm_biases[1]
        )
        return x_out
