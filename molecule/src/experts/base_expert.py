from abc import ABC, abstractmethod

import torch
from jaxtyping import Float


class MoEExpertABC(ABC):
    """
    Abstract base class for MoE experts.
    """

    @classmethod
    @abstractmethod
    def from_pretrained(cls, device: str, *args, **kwargs) -> "MoEExpertABC":
        """
        Load a pretrained MoE expert model.

        Args:
            device: The device on which to load the model.
            *args: Additional positional arguments for the model initialization.
            **kwargs: Additional keyword arguments for the model initialization.

        Returns:
            An instance of the MoE expert model.
        """
        ...

    @abstractmethod
    def prepare_data(self, *args, **kwargs): ...

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
