import logging

import torch
import yaml
from easydict import EasyDict
from geodiff.models.epsnet import get_model
from geodiff.models.epsnet.dualenc import DualEncoderEpsNetwork
from jaxtyping import Float
from torch import nn

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

    def prepare_data(self, batch_size: int, num_nodes: int):
        # Implementation for preparing data specific to GeoDiff
        ...

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


if __name__ == "__main__":
    # Example usage of the GeoDiffExpert class
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geodiff_expert = GeoDiffExpert.from_pretrained(device=device)
    logger.info("GeoDiff Expert loaded successfully.")
