import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from Bio.PDB.PDBParser import PDBParser
from diffsbdd.lightning_modules import LigandPocketDDPM
from diffsbdd.utils import get_pocket_from_ligand, num_nodes_to_batch_mask
from jaxtyping import Float
from rdkit import Chem
from torch_scatter import scatter_mean

from src.const import PRETRAINED_MODEL_DIR
from src.experts.base_expert import MoEExpertABC
from src.utils.logging_utils import redirect_output_to_logger

DIFFSBDD_SOURCE_DIR = PRETRAINED_MODEL_DIR / "DiffSBDD"
DIFFSBDD_CKPT_PATH = DIFFSBDD_SOURCE_DIR / "checkpoints" / "crossdocked_fullatom_cond.ckpt"

PathLike = Path | str


logger = logging.getLogger(__name__)


@dataclass
class DiffSBDDInferenceContext:
    model: LigandPocketDDPM
    num_samples: int
    lig_mask: torch.Tensor
    pocket: dict[str, torch.Tensor]
    pocket_com_before: torch.Tensor
    device: str
    z: Float[torch.Tensor, "B D"]
    xh_pocket: torch.Tensor
    data_shape: tuple[int, ...]


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
        protein_pocket_structure = parser.get_structure("", str(protein_pocket_pdb_path))[0]  # type: ignore
        residues = get_pocket_from_ligand(protein_pocket_structure, reference_ligand_mol)

        pocket = self.model.prepare_pocket(residues, repeats=batch_size)
        pocket: dict[str, torch.Tensor] = self.model.ddpm.normalize(pocket=pocket)[1]  # type: ignore
        pocket_com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        mu_lig_x = scatter_mean(pocket["x"], pocket["mask"], dim=0)
        mu_lig_h = torch.zeros((batch_size, self.model.ddpm.atom_nf), device=self.device)

        lig_mask = num_nodes_to_batch_mask(batch_size, num_nodes, device=self.device)
        mu_lig = torch.cat([mu_lig_x, mu_lig_h], dim=1)[lig_mask]
        sigma = torch.ones_like(pocket["size"]).unsqueeze(1)

        z_lig, xh_pocket = self.model.ddpm.sample_normal_zero_com(mu_lig, xh0_pocket, sigma, lig_mask, pocket["mask"])
        data_shape = z_lig.shape
        z_lig = z_lig.reshape(batch_size, -1)

        prepared_data = {
            "model": self.model,  # todo; remove this later
            "num_samples": batch_size,  # todo; remove this later
            "lig_mask": lig_mask,
            "pocket": pocket,
            "pocket_com_before": pocket_com_before,
            "device": self.device,  # todo; remove this later
            "z": z_lig,
            "xh_pocket": xh_pocket,
            "data_shape": data_shape,
        }
        # Store the prepared data for inference
        self._inference_context = DiffSBDDInferenceContext(**prepared_data)  # type: ignore
        return prepared_data

    def score(
        self,
        t: Float[torch.Tensor, " B"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, " B"]:
        # Implementation for scoring using the DiffSBDD expert
        model = self.model
        lig_mask = self._inference_context.lig_mask
        pocket = self._inference_context.pocket
        xh_pocket = self._inference_context.xh_pocket
        data_shape = self._inference_context.data_shape
        batch_size = x.shape[0]
        curr_shape = x.shape

        x = x.reshape(data_shape)
        T = model.T

        i = (T * t).int().clamp(1, T - 1)[0].item()

        s_array = torch.full((batch_size, 1), fill_value=i - 1, device=self.device)
        t_array = s_array + 1
        s_array = s_array / T
        t_prev_array = (t_array - 1) / T
        t_array = t_array / T

        gamma_s = model.ddpm.gamma(s_array)
        gamma_t = model.ddpm.gamma(t_array)
        gamma_t_prev = model.ddpm.gamma(t_prev_array)
        sigma_t = model.ddpm.sigma(gamma_t, target_tensor=x)
        alpha_t = model.ddpm.alpha(gamma_t, target_tensor=x)
        alpha_t_prev = model.ddpm.alpha(gamma_t_prev, target_tensor=x)
        beta_t = (-2 * torch.log(alpha_t / alpha_t_prev)) * T

        eps_t_lig, _ = model.ddpm.dynamics(x, xh_pocket, t_array, lig_mask, pocket["mask"])
        score = -eps_t_lig / sigma_t[lig_mask]

        return score.reshape(curr_shape)

    def interleave(self, *args, **kwargs):
        # Implementation for interleaving data specific to DiffSBDD
        ...

    def postprocess(self, *args, **kwargs):
        # Implementation for postprocessing results from the DiffSBDD expert
        ...


if __name__ == "__main__":
    # Example usage of the DiffSBDDExpert class
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffsbdd_expert = DiffSBDDExpert.from_pretrained(device=device)
    logger.info("DiffSBDD Expert loaded successfully.")

    reference_ligand_mol = Chem.SDMolSupplier("examples/4m7t_fragment.sdf")[0]
    prepared_data = diffsbdd_expert.prepare_data(
        batch_size=10,
        num_nodes=10,
        protein_pocket_pdb_path="examples/4m7t_pocket.pdb",
        reference_ligand_mol=reference_ligand_mol,
    )
    print("Prepared data keys:", prepared_data.keys())
