from abc import ABC, abstractmethod
from typing import Any

import torch
from jaxtyping import Float


class MoEExpertABC(ABC):
    """
    Abstract base class for MoE experts.
    """

    device: str
    model: Any
    model_config: Any

    @classmethod
    @abstractmethod
    def from_pretrained(cls, device: str) -> "MoEExpertABC":
        """
        Load a pretrained MoE expert model.

        Args:
            device: The device on which to load the model.

        Returns:
            An instance of the MoE expert model.
        """
        ...

    @abstractmethod
    def prepare_data(self, batch_size: int, *args, **kwargs) -> dict[str, Any]:
        """Prepare the data for inference."""
        ...

    @abstractmethod
    def score(
        self,
        t: Float[torch.Tensor, " B"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, " B"]:
        """Compute the score function for the given input."""
        ...

    @abstractmethod
    def interleave(self, x: Float[torch.Tensor, "B D"], *args, **kwargs) -> Float[torch.Tensor, "B D"]:
        """
        Interleave function that being applied at each denoising step to ensure the
        correct correspondence between ligand and pocket atoms.
        """
        ...

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        """
        Postprocess function that is applied after the denoising process to ensure the
        correct correspondence between ligand and pocket atoms.
        """
        ...
