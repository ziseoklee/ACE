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
    def prepare_data(self, batch_size: int, *args, **kwargs) -> dict[str, Any]: ...

    @abstractmethod
    def score(
        self,
        t: Float[torch.Tensor, " B"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, " B"]: ...

    @abstractmethod
    def interleave(self, *args, **kwargs): ...

    @abstractmethod
    def postprocess(self, *args, **kwargs): ...
