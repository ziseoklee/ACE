import argparse
import logging
import pickle

import torch
from e3_diffusion_for_molecules.configs.datasets_config import get_dataset_info
from e3_diffusion_for_molecules.equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from e3_diffusion_for_molecules.qm9.models import get_model
from jaxtyping import Float

from src.const import PRETRAINED_MODEL_DIR
from src.experts.base_expert import MoEExpertABC
from src.utils.logging_utils import redirect_output_to_logger

EDM_SOURCE_DIR = PRETRAINED_MODEL_DIR / "e3_diffusion_for_molecules"
EDM_CKPT_PATH = EDM_SOURCE_DIR / "outputs" / "edm_qm9" / "generative_model_ema.npy"
EDM_MODEL_CONFIG_PATH = EDM_SOURCE_DIR / "outputs" / "edm_qm9" / "args.pickle"


logger = logging.getLogger(__name__)


class EDMExpert(MoEExpertABC):
    """
    Expert class for EDM (Equivariant Diffusion for Molecule Generation in 3D, ICML2022).

    Reference code: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """

    device: str
    model: EnVariationalDiffusion
    model_config: argparse.Namespace

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

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("EDM model loaded with %d parameters", n_params)

        instance = cls(device=device, model=model, model_config=edm_config)
        return instance

    def prepare_data(self, batch_size: int, num_nodes: int):
        # Implementation for preparing data specific to EDM
        ...

    def score(
        self,
        t: Float[torch.Tensor, " B"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, " B"]:
        # Implementation for scoring using the EDM expert
        ...

    def interleave(self, *args, **kwargs):
        # Implementation for interleaving data specific to EDM
        ...

    def postprocess(self, *args, **kwargs):
        # Implementation for postprocessing results from the EDM expert
        ...


if __name__ == "__main__":
    # Example usage of the EDMExpert class
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    edm_expert = EDMExpert.from_pretrained(device=device)
    logger.info("EDM Expert loaded successfully.")
