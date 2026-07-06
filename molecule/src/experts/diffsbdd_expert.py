import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from Bio.PDB.PDBParser import PDBParser
from diffsbdd.lightning_modules import LigandPocketDDPM
from diffsbdd.utils import get_pocket_from_ligand, num_nodes_to_batch_mask
from jaxtyping import Float, Int
from rdkit import Chem
from torch_scatter import scatter_add, scatter_mean

from experts import PRETRAINED_MODEL_DIR
from experts.base_expert import MoEExpertABC, SBDDPocketType
from utils.logging_utils import redirect_output_to_logger

DIFFSBDD_SOURCE_DIR = PRETRAINED_MODEL_DIR / "DiffSBDD"
DIFFSBDD_CKPT_PATH = DIFFSBDD_SOURCE_DIR / "checkpoints" / "crossdocked_fullatom_cond.ckpt"

PathLike = os.PathLike[str] | str


logger = logging.getLogger(__name__)


@dataclass
class DiffSBDDInferenceContext:
    """DiffSBDD-specific masks and shapes around the current batched pocket state."""

    lig_mask: Float[torch.Tensor, "B*ligand_nodes"]  # noqa: F821
    pocket_mask: Float[torch.Tensor, "B*pocket_nodes"]  # noqa: F821
    xh_pocket: SBDDPocketType
    initial_pocket_com: Float[torch.Tensor, "B coords"]
    data_shape: tuple[
        int, ...
    ]  # DiffSBDD's internal data shape (B*L, D) where L is the number of nodes in the ligand and D is the feature dimension
    batch_size: int


class DiffSBDDExpert(MoEExpertABC):
    """
    Expert class for DiffSBDD (DiffSBDD: Structure-based Drug Design with Equivariant Diffusion Models, Nature Computational Science 2024).

    Reference code: https://github.com/arneschneuing/DiffSBDD
    """

    device: str
    model: LigandPocketDDPM
    model_config: None
    _inference_context: DiffSBDDInferenceContext

    def __init__(self, device: str, model: LigandPocketDDPM, model_config: None = None):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, device: str):
        logger.info(f"Loading DiffSBDD expert from pretrained model at {DIFFSBDD_CKPT_PATH}")

        with redirect_output_to_logger(logger):
            model = LigandPocketDDPM.load_from_checkpoint(DIFFSBDD_CKPT_PATH, map_location=device, weights_only=False)
        model.to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("DiffSBDD model loaded with %d parameters", n_params)

        instance = cls(device=device, model=model)
        return instance

    def prepare_data(
        self,
        batch_size: int,
        num_nodes: int,
        protein_pocket_pdb_path: PathLike,
        reference_ligand_mol: Chem.Mol,
    ):
        # Implementation for preparing data specific to DiffSBDD
        protein_pocket_pdb_path = Path(protein_pocket_pdb_path)
        parser = PDBParser(QUIET=True)
        protein_pocket_structure = parser.get_structure("", str(protein_pocket_pdb_path))[0]  # pyright: ignore[reportOptionalSubscript]
        residues = get_pocket_from_ligand(protein_pocket_structure, reference_ligand_mol)

        pocket = self.model.prepare_pocket(residues, repeats=batch_size)
        pocket: dict[str, torch.Tensor] = self.model.ddpm.normalize(pocket=pocket)[1]  # pyright: ignore[reportAssignmentType]
        pocket_mask = pocket["mask"]
        initial_pocket_com = scatter_mean(pocket["x"], pocket_mask, dim=0)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        lig_mask = num_nodes_to_batch_mask(batch_size, num_nodes, device=self.device)
        data_shape = (len(lig_mask), self.model.ddpm.n_dims + self.model.ddpm.atom_nf)
        xh_pocket = xh0_pocket.view(batch_size, -1, xh0_pocket.shape[-1])

        prepared_data = {
            "lig_mask": lig_mask,
            "pocket_mask": pocket_mask,
            "initial_pocket_com": initial_pocket_com,
            "xh_pocket": xh_pocket,
            "data_shape": data_shape,
            "batch_size": batch_size,
        }
        # Store the prepared data for inference
        self._inference_context = DiffSBDDInferenceContext(**prepared_data)  # pyright: ignore[reportArgumentType]
        return prepared_data

    def get_current_pocket_xh(self) -> SBDDPocketType:
        return self._inference_context.xh_pocket

    def set_current_pocket_xh(self, xh_pocket: SBDDPocketType) -> None:
        if xh_pocket.ndim != 3:
            raise ValueError(f"Expected pocket xh with shape (B, pocket_nodes, feature), got {tuple(xh_pocket.shape)}.")
        if xh_pocket.shape[0] != self._inference_context.batch_size:
            raise ValueError(
                "Pocket xh batch size does not match DiffSBDD context: "
                f"{xh_pocket.shape[0]} != {self._inference_context.batch_size}."
            )
        if xh_pocket.shape[1:] != self._inference_context.xh_pocket.shape[1:]:
            raise ValueError(
                "Pocket xh node/feature shape does not match DiffSBDD context: "
                f"{tuple(xh_pocket.shape[1:])} != {tuple(self._inference_context.xh_pocket.shape[1:])}."
            )
        self._inference_context.xh_pocket = xh_pocket

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        # Implementation for scoring using the DiffSBDD expert
        model = self.model
        lig_mask = self._inference_context.lig_mask
        pocket_mask = self._inference_context.pocket_mask
        xh_pocket = self._flat_current_pocket_xh()
        data_shape = self._inference_context.data_shape
        batch_size = x.shape[0]
        curr_shape = x.shape

        x = x.reshape(data_shape)
        num_timesteps = model.T

        if not torch.allclose(t, t[:1].expand_as(t)):
            raise ValueError("DiffSBDD score expects all samples in the batch to share the same timestep.")

        i = torch.round(num_timesteps * t[0, 0]).long().clamp(1, num_timesteps).item()

        t_array = torch.full((batch_size, 1), fill_value=i, device=self.device, dtype=t.dtype)
        t_array = t_array / num_timesteps

        gamma_t = model.ddpm.gamma(t_array)
        sigma_t = model.ddpm.sigma(gamma_t, target_tensor=x)

        eps_t_lig, _ = model.ddpm.dynamics(x, xh_pocket, t_array, lig_mask, pocket_mask)
        score = -eps_t_lig / sigma_t[lig_mask]

        return score.reshape(curr_shape)

    def interleave(
        self,
        x: Float[torch.Tensor, "B data"],
        choices: Int[np.ndarray, "B"],  # noqa: F821
        mask: torch.Tensor,
    ) -> Float[torch.Tensor, "B data"]:
        """
        Keep DiffSBDD conditioning in the ligand-centered frame after resampling.

        This updates the pocket context to follow the resampled ligand particles
        and re-centers the DiffSBDD ligand coordinates so each ligand COM stays
        at zero, with the pocket shifted by the same translation.
        """
        lig_mask = self._inference_context.lig_mask
        pocket_mask = self._inference_context.pocket_mask
        data_shape = self._inference_context.data_shape

        current_xh_pocket = self.get_current_pocket_xh()
        choice_tensor = torch.as_tensor(choices, device=current_xh_pocket.device, dtype=torch.long)
        xh_pocket = current_xh_pocket.index_select(0, choice_tensor)
        self.set_current_pocket_xh(xh_pocket)
        xh_pocket = self._flat_current_pocket_xh()

        x_out = x.detach()
        x_sbdd = x_out[..., mask]
        curr_shape = x_sbdd.shape
        x_sbdd = x_sbdd.reshape(data_shape)

        x_coords, pocket_coords = self.model.ddpm.remove_mean_batch(
            x_sbdd[..., : self.model.ddpm.n_dims],
            xh_pocket[..., : self.model.ddpm.n_dims],
            lig_mask,  # pyright: ignore[reportCallIssue]
            pocket_mask,
        )
        x_sbdd[..., : self.model.ddpm.n_dims] = x_coords
        xh_pocket = xh_pocket.clone()
        xh_pocket[..., : self.model.ddpm.n_dims] = pocket_coords
        self._set_flat_current_pocket_xh(xh_pocket)
        self.model.ddpm.assert_mean_zero_with_mask(x_sbdd[:, : self.model.ddpm.n_dims], lig_mask)

        x_out[..., mask] = x_sbdd.reshape(curr_shape)
        return x_out

    def decode_mu_xh_given_z0(
        self,
        z0_lig: Float[torch.Tensor, "B data"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode z0 with the DiffSBDD posterior mean, without final Gaussian sampling."""
        model = self.model
        lig_mask = self._inference_context.lig_mask
        pocket_mask = self._inference_context.pocket_mask
        xh_pocket = self._flat_current_pocket_xh()
        batch_size = z0_lig.shape[0]

        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = model.ddpm.gamma(t_zeros)
        net_out_lig, _ = model.ddpm.dynamics(z0_lig, xh_pocket, t_zeros, lig_mask, pocket_mask)
        mu_xh_lig = model.ddpm.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)

        x_lig, h_lig = model.ddpm.unnormalize(mu_xh_lig[..., : model.ddpm.n_dims], z0_lig[..., model.ddpm.n_dims :])
        x_pocket, h_pocket = model.ddpm.unnormalize(
            xh_pocket[..., : model.ddpm.n_dims],
            xh_pocket[..., model.ddpm.n_dims :],
        )

        return x_lig, h_lig, x_pocket, h_pocket

    def postprocess(
        self,
        x: Float[torch.Tensor, "B data"],
        mask: torch.Tensor,
    ) -> Float[torch.Tensor, "B data"]:
        """Decode and restore the DiffSBDD portion while preserving the full MoE tensor shape."""
        model = self.model
        lig_mask = self._inference_context.lig_mask
        pocket_mask = self._inference_context.pocket_mask
        initial_pocket_com = self._inference_context.initial_pocket_com
        data_shape = self._inference_context.data_shape

        x_clone = x.clone().detach()
        curr_shape = x_clone[..., mask].shape
        z0_lig = x_clone[..., mask].reshape(data_shape)
        x_lig, h_lig, x_pocket, h_pocket = self.decode_mu_xh_given_z0(z0_lig)

        max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
        if max_cog > 5e-2:
            logger.warning("CoG drift with error %.3f. Projecting the positions down.", max_cog)
            x_lig, x_pocket = model.ddpm.remove_mean_batch(
                x_lig,
                x_pocket,
                lig_mask,  # pyright: ignore[reportCallIssue]
                pocket_mask,
            )
        model.ddpm.assert_mean_zero_with_mask(x_lig, lig_mask)

        xh_lig = torch.cat([x_lig, h_lig], dim=1)
        xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)

        # Re-center the pocket context to follow the resampled ligand particles
        final_pocket_com = scatter_mean(xh_pocket[:, : model.x_dims], pocket_mask, dim=0)
        xh_pocket[:, : model.x_dims] += (initial_pocket_com - final_pocket_com)[pocket_mask]
        xh_lig[:, : model.x_dims] += (initial_pocket_com - final_pocket_com)[lig_mask]

        x_clone[..., mask] = xh_lig.detach().reshape(curr_shape).to(x_clone.device)
        return x_clone

    def _flat_current_pocket_xh(self) -> Float[torch.Tensor, "B*pocket_nodes feature"]:
        xh_pocket = self.get_current_pocket_xh()
        return xh_pocket.reshape(-1, xh_pocket.shape[-1])

    def _set_flat_current_pocket_xh(self, xh_pocket: Float[torch.Tensor, "B*pocket_nodes feature"]) -> None:
        batch_size = self._inference_context.batch_size
        self.set_current_pocket_xh(xh_pocket.reshape(batch_size, -1, xh_pocket.shape[-1]))
