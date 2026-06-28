import logging

import torch
from diffsbdd.lightning_modules import LigandPocketDDPM
from jaxtyping import Float

from src.const import PRETRAINED_MODEL_DIR
from src.experts.base_expert import MoEExpertABC
from src.utils.logging_utils import redirect_output_to_logger

DIFFSBDD_SOURCE_DIR = PRETRAINED_MODEL_DIR / "DiffSBDD"
DIFFSBDD_CKPT_PATH = DIFFSBDD_SOURCE_DIR / "checkpoints" / "crossdocked_fullatom_cond.ckpt"


logger = logging.getLogger(__name__)


class DiffSBDDExpert(MoEExpertABC):
    """
    Expert class for DiffSBDD (DiffSBDD: Structure-based Drug Design with Equivariant Diffusion Models, Nature Computational Science 2024).

    Reference code: https://github.com/arneschneuing/DiffSBDD
    """

    device: str
    model: LigandPocketDDPM
    model_config: None

    def __init__(self, device: str, model: LigandPocketDDPM, model_config: None = None):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, device: str):
        logger.info(f"Loading DiffSBDD expert from pretrained model at {DIFFSBDD_CKPT_PATH}")

        with redirect_output_to_logger(logger):
            model = LigandPocketDDPM.load_from_checkpoint(DIFFSBDD_CKPT_PATH, map_location=device)
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("DiffSBDD model loaded with %d parameters", n_params)

        instance = cls(device=device, model=model)
        return instance

    def prepare_data(self, batch_size: int, num_nodes: int):
        # Implementation for preparing data specific to DiffSBDD
        ...

    def score(
        self,
        t: Float[torch.Tensor, " B"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, " B"]:
        # Implementation for scoring using the DiffSBDD expert
        ...

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
