import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol

import torch
from jaxtyping import Bool, Float

logger = logging.getLogger(__name__)


ScoreFunctionType = Callable[
    [Float[torch.Tensor, "B 1"], Float[torch.Tensor, "B data"]],
    Float[torch.Tensor, "B data"],
]
ExponentFunctionType = Callable[[Float[torch.Tensor, "B 1"]], Float[torch.Tensor, "B 1"]]
DataMask = Bool[torch.Tensor, "data"]  # noqa: F821
SCORE_CLAMP_MAGNITUDE = 20.0  # ! IMPORTANT: clamp score function to avoid numerical instability near t=0 or t=1
CacheKeyType = tuple[int, int, int, int, tuple[int, ...], tuple[int, ...], torch.device, torch.device]


class SDEScheduler(Protocol):
    def drift_coeff(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]: ...

    def diffusion_coeff(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]: ...


class ProbabilityPathABC(ABC):
    @abstractmethod
    def drift_coeff(
        self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B data"]
    ) -> Float[torch.Tensor, "B data"]:
        pass

    @abstractmethod
    def v(self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B data"]) -> Float[torch.Tensor, "B data"]:
        pass

    @abstractmethod
    def score(self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B data"]) -> Float[torch.Tensor, "B data"]:
        pass

    @abstractmethod
    def sigma(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        pass


class ProbabilityPath(ProbabilityPathABC):
    """
    Unlike conventional path definition, we  assume t = 0 -> t = 1 for both forward and reverse processes, and the scheduler is defined on [0, 1].
    """

    def __init__(
        self,
        scheduler: SDEScheduler,
        score_model: ScoreFunctionType,
        reverse: bool = True,
    ):
        self.scheduler = scheduler
        self.score_model = score_model
        self.reverse = reverse

    def drift_coeff(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        """Drift coefficient of SDE"""
        if not self.reverse:
            return self.scheduler.drift_coeff(t, x)
        else:
            f = self.scheduler.drift_coeff(1 - t, x)
            return -f + self.sigma(t) ** 2 * self.score(t, x).clamp(-SCORE_CLAMP_MAGNITUDE, SCORE_CLAMP_MAGNITUDE)

    def v(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        """Velocity of PF-ODE"""
        if not self.reverse:
            f = self.scheduler.drift_coeff(t, x)
            return f - 0.5 * self.scheduler.diffusion_coeff(t) ** 2 * self.score(t, x).clamp(
                -SCORE_CLAMP_MAGNITUDE, SCORE_CLAMP_MAGNITUDE
            )
        else:
            f = self.scheduler.drift_coeff(1 - t, x)
            return -f + 0.5 * self.sigma(t) ** 2 * self.score(t, x).clamp(-SCORE_CLAMP_MAGNITUDE, SCORE_CLAMP_MAGNITUDE)

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        """Score function of SDE"""
        if not self.reverse:
            return self.score_model(t, x)
        else:
            return self.score_model(1 - t, x)

    def sigma(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """Diffusion coefficient of SDE"""
        if not self.reverse:
            return self.scheduler.diffusion_coeff(t)
        else:
            return self.scheduler.diffusion_coeff(1 - t)


class PaddedProbabilityPath(ProbabilityPathABC):
    """
    A wrapper around ProbabilityPath that pads the input and output to a common dimension. This is useful for MoE where different paths may have different dimensions but we want to combine them in a single tensor.

    len(paths) = len(mask_list) = 2 should hold. The first path will be responsible for original data dimensions, and the second path will be responsible for the extra dimensions (e.g., for control or steering). The masks indicate which dimensions each path is responsible for.
    """

    def __init__(self, paths: list["ProbabilityPath"], mask_list: list[DataMask]):
        assert len(paths) == len(mask_list), "Number of paths and masks must be the same"
        assert len(paths) == 2, "Paths must be of length 2 for original and extra dimensions"
        self.paths = paths
        self.mask_list = mask_list
        self.sample_size = mask_list[0].shape[0]
        self.reverse = self.check_reverse()
        self.scheduler = self.paths[0].scheduler

    def check_reverse(self) -> bool:
        for path in self.paths:
            if path.reverse != self.paths[0].reverse:
                raise ValueError("All paths must have the same reverse value")
        return self.paths[0].reverse

    def drift_coeff(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        out = torch.zeros(x.shape[0], self.sample_size, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].drift_coeff(t, x[:, self.mask_list[i]])
        return out

    def v(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        out = torch.zeros(x.shape[0], self.sample_size, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].v(t, x[:, self.mask_list[i]])

        return out

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        out = torch.zeros(x.shape[0], self.sample_size, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].score(t, x[:, self.mask_list[i]])
        return out

    def sigma(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        return self.paths[0].sigma(t)


class MoEProbabilityPath(ProbabilityPathABC):
    def __init__(
        self,
        scheduler: SDEScheduler,  # global noise schedule for MoE path, can be different from individual paths' schedulers
        q_list: list["PaddedProbabilityPath"],
        mask_list: list[DataMask],
        exponent_list: list[ExponentFunctionType],
        sample_size: int,
        node_feature_dim: int,
        diffusion_scale: float = 2.0,  # FIXME: this value works well in DiffSBDD and 4-expert MoE inference but not consistently in EDM-only or GeoDiff-only inference. The reason for the default scale 2.0 is not yet fully understood.
    ):
        super().__init__()
        if not isinstance(node_feature_dim, int) or node_feature_dim <= 0:
            raise ValueError(f"node_feature_dim must be a positive int, got {node_feature_dim!r}.")
        if not math.isfinite(float(diffusion_scale)) or diffusion_scale <= 0:
            raise ValueError(f"diffusion_scale must be positive and finite, got {diffusion_scale!r}.")

        self.q_list = q_list
        self.mask_list = mask_list
        self.reverse = self.check_reverse()
        self.exponent_list = exponent_list
        self.scheduler = scheduler
        self.diffusion_scale = float(diffusion_scale)

        self.sample_size = sample_size
        self.node_feature_dim = node_feature_dim

        # for caching the mixture weights, to avoid recomputing them every time
        self.clear_cache()

    def check_reverse(self) -> bool:
        for path in self.q_list:
            if path.reverse != self.q_list[0].reverse:
                raise ValueError("All paths must have the same reverse value")
        return self.q_list[0].reverse

    def clear_cache(self):
        self._cache_key: CacheKeyType | None = None
        self._cache = {}

    def _make_cache_key(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> CacheKeyType:
        return (
            id(t),
            id(x),
            t._version,
            x._version,
            tuple(t.shape),
            tuple(x.shape),
            t.device,
            x.device,
        )

    def _ensure_cache_context(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> None:
        cache_key = self._make_cache_key(t, x)
        if self._cache_key != cache_key:
            self._cache_key = cache_key
            self._cache = {}

    def _gamma(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B E 1"]:
        self._ensure_cache_context(t, x)
        if "gamma" not in self._cache:
            self._cache["gamma"] = torch.stack([exponent_fn(t) for exponent_fn in self.exponent_list], dim=1)
        return self._cache["gamma"]

    def _expert_v(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B E data"]:
        self._ensure_cache_context(t, x)
        if "expert_v" not in self._cache:
            self._cache["expert_v"] = torch.stack(
                [
                    pad_tensor(q.v(t, x[:, mask]), self.sample_size, dim=1)
                    for q, mask in zip(self.q_list, self.mask_list)
                ],
                dim=1,
            )
        return self._cache["expert_v"]

    def _expert_score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B E data"]:
        self._ensure_cache_context(t, x)
        if "expert_score" not in self._cache:
            self._cache["expert_score"] = torch.stack(
                [
                    pad_tensor(q.score(t, x[:, mask]), self.sample_size, dim=1).clamp(
                        -SCORE_CLAMP_MAGNITUDE,
                        SCORE_CLAMP_MAGNITUDE,
                    )
                    for q, mask in zip(self.q_list, self.mask_list)
                ],
                dim=1,
            )
        return self._cache["expert_score"]

    def drift_coeff(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        """Drift coefficient of MoE SDE, calculated as weighted mixture of component-wise velocities and scores."""
        self._ensure_cache_context(t, x)
        if "moe_drift" not in self._cache:
            moe_v = self.v(t, x)
            moe_score = self.score(t, x)
            self._cache["moe_drift"] = moe_v + 0.5 * self.sigma(t) ** 2 * moe_score
        return self._cache["moe_drift"]

    def v(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        """Velocity of MoE PF-ODE, calculated as weighted mixture of component-wise velocities."""
        self._ensure_cache_context(t, x)
        if "moe_v" not in self._cache:
            self._cache["moe_v"] = (self._gamma(t, x) * self._expert_v(t, x)).sum(dim=1)
        return self._cache["moe_v"]

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
    ) -> Float[torch.Tensor, "B data"]:
        """Score function of MoE SDE, calculated as weighted mixture of component-wise scores."""
        self._ensure_cache_context(t, x)
        if "moe_score" not in self._cache:
            self._cache["moe_score"] = (self._gamma(t, x) * self._expert_score(t, x)).sum(dim=1)
        return self._cache["moe_score"]

    def sigma(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """Diffusion coefficient of MoE SDE, calculated as diffusion of global scheduler (not a mixture)."""
        if not self.reverse:
            return self.diffusion_scale * self.scheduler.diffusion_coeff(t)
        else:
            return self.diffusion_scale * self.scheduler.diffusion_coeff(1 - t)

    def get_dlogq(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
        generator: torch.Generator | None = None,
    ) -> tuple[Float[torch.Tensor, "B E 1"], Float[torch.Tensor, "B E data"]]:
        """
        Calculate the logq correction term for each expert, which is used for logq correction in drift and resampling.
        """
        # line 8's logq correction term
        v: Float[torch.Tensor, "B E data"] = self._expert_v(t, x)
        s: Float[torch.Tensor, "B E data"] = self._expert_score(t, x)

        x_subset_list = [x[:, mask] for mask in self.mask_list]
        div_s = torch.stack(
            [
                divergence_hutchinson(q.score, t, x_subset, generator=generator)
                for q, x_subset in zip(self.q_list, x_subset_list)
            ],
            dim=1,
        )
        div_v = torch.stack(
            [
                divergence_hutchinson(q.v, t, x_subset, generator=generator)
                for q, x_subset in zip(self.q_list, x_subset_list)
            ],
            dim=1,
        )
        sigma = self.sigma(t).unsqueeze(1)  # (B, 1)

        # dlogq_A: (B, num_experts, 1), dlogq_B: (B, num_experts, data_dim)
        dlogq_tensor_drift_term = (
            -div_v + ((self.drift_coeff(t, x).unsqueeze(1) - v) * s).sum(dim=2, keepdim=True) + 0.5 * sigma**2 * div_s
        )
        dlogq_tensor_diffusion_term = sigma * s  # dW related term

        return dlogq_tensor_drift_term, dlogq_tensor_diffusion_term

    def get_dlog_weight(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B data"],
        use_logq: bool,
        logq_tensor: Float[torch.Tensor, "B E 1"] | None,
    ) -> Float[torch.Tensor, "B"]:  # noqa: F821
        """
        Calculate the time derivative of log weight for each expert, which is used for resampling and logq correction.
        """
        gamma: Float[torch.Tensor, "B E 1"] = self._gamma(t, x)
        v: Float[torch.Tensor, "B E data"] = self._expert_v(t, x)
        s: Float[torch.Tensor, "B E data"] = self._expert_score(t, x)
        # dlog_weight except for logq correction
        dlog_weight_base: Float[torch.Tensor, "B 1"] = (
            gamma * ((self.v(t, x).unsqueeze(1) - v) * s).sum(dim=2, keepdim=True)
        ).sum(dim=1)

        # update log weight with logq correction if needed
        if use_logq:
            assert logq_tensor is not None, "logq_tensor must be provided when use_logq is True"
            d_gamma = torch.stack(  # time derivatives of exponents
                [
                    torch.func.jacfwd(exponent_fn, argnums=0)(t).squeeze().diag().unsqueeze(1)  # pyright: ignore[reportAttributeAccessIssue]
                    for exponent_fn in self.exponent_list
                ],
                dim=1,
            )
            dlog_weight = dlog_weight_base + (d_gamma * logq_tensor).sum(dim=1)
        else:
            dlog_weight = dlog_weight_base

        return dlog_weight.squeeze(-1)  # (B,)


def pad_tensor(x: Float[torch.Tensor, "B data"], pad_size: int, dim: int) -> Float[torch.Tensor, "B data"]:
    padding_shape = list(x.shape)
    padding_shape[dim] = pad_size - x.shape[dim]
    return torch.cat([x, torch.zeros(padding_shape, device=x.device)], dim=dim)


def divergence_hutchinson(
    score_fn: ScoreFunctionType,
    t: Float[torch.Tensor, "B 1"],
    x: Float[torch.Tensor, "B data"],
    generator: torch.Generator | None = None,
    n_probe: int = 1,
    rademacher: bool = True,
    create_graph: bool = False,  # sampling usually doesn't need higher-order grads
    t_eps: float = 1e-4,
    jitter: float = 1e-6,
    grad_clip: float = 1e6,
) -> Float[torch.Tensor, "B 1"]:
    """
    Safe Hutchinson trace estimator: E_eps[ eps^T (∂ score / ∂x) eps ].

    - clamps t to (t_eps, 1 - t_eps)
    - jitters x -> x + jitter * eps to avoid 1/0 singularities inside score_fn
    - nan_to_num on score and grads
    - clips extreme grads
    - retries once with 10x jitter if backward still fails
    """
    # Ensure t is batch-shaped and safe
    if t.dim() == 0:
        t = t * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    else:
        # broadcast to (..., 1) if needed
        if t.shape[-1] != 1:
            t = t.view(-1, 1)
    t = t.clamp(t_eps, 1.0 - t_eps)

    # We need grads w.r.t x
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    div = x.new_zeros((x.shape[0], 1))
    for k in range(n_probe):
        if rademacher:
            eps = torch.randint(
                low=0,
                high=2,
                size=x.shape,
                device=x.device,
                generator=generator,
            ).to(dtype=x.dtype)
            eps = eps.mul_(2).sub_(1)
        else:
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

        def one_pass(curr_jitter: float):
            with torch.enable_grad():
                xj = x + curr_jitter * eps  # break exact coincidences / zeros
                s = score_fn(t, xj)
                # sanitize forward values (don’t crash on inf/nan)
                s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                phi = (s * eps).sum(dim=-1)
                (g,) = torch.autograd.grad(
                    phi.sum(),
                    x,
                    retain_graph=(k + 1 < n_probe),
                    create_graph=create_graph,
                    allow_unused=True,
                )
                if g is None:
                    g = torch.zeros_like(x)
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                if grad_clip is not None:
                    g = torch.clamp(g, -grad_clip, grad_clip)
                return (g * eps).sum(dim=-1, keepdim=True)

        try:
            contrib = one_pass(jitter)
        except RuntimeError as e:
            logger.error(f"RuntimeError: {e}")
            # Often "ReciprocalBackward0 returned nan"; try with a bigger jitter once.
            contrib = one_pass(jitter * 10)

        div = div + contrib
    return div / n_probe
