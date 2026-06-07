import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import torch
from jaxtyping import Float

from src.scheduler import SchedulerABC

logger = logging.getLogger(__name__)

### Probability Path ###
# The following class implements the probability path attained from the Fokker-Planck equation scheduling given data distribution and prior (Gaussian) distribution.

ScoreFunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ProbabilityPathABC(ABC):
    @abstractmethod
    def drift_coeff(self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B D"]) -> Float[torch.Tensor, "B D"]:
        pass

    @abstractmethod
    def v(self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B D"]) -> Float[torch.Tensor, "B D"]:
        pass

    @abstractmethod
    def score(self, t: Float[torch.Tensor, "B 1"], x: Float[torch.Tensor, "B D"]) -> Float[torch.Tensor, "B D"]:
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
        scheduler: SchedulerABC,
        score_model: ScoreFunctionType,
        reverse: bool = True,
    ):
        self.scheduler = scheduler
        self.score_model = score_model
        self.reverse = reverse

    def drift_coeff(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        """Drift coefficient of SDE"""
        if not self.reverse:
            return self.scheduler.f(t, x)
        else:
            f = self.scheduler.f(1 - t, x)
            return -f + self.sigma(t) ** 2 * self.score(t, x)

    def v(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        """Velocity of PF-ODE"""
        if not self.reverse:
            f = self.scheduler.f(t, x)
            return f - 0.5 * self.scheduler.sigma(t) ** 2 * self.score(t, x)
        else:
            f = self.scheduler.f(1 - t, x)
            return -f + 0.5 * self.sigma(t) ** 2 * self.score(t, x)

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        """Score function of SDE"""
        if not self.reverse:
            return self.score_model(t, x)
        else:
            return self.score_model(1 - t, x)

    def sigma(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """Diffusion coefficient of SDE"""
        if not self.reverse:
            return self.scheduler.sigma(t)
        else:
            return self.scheduler.sigma(1 - t)


class PaddedProbabilityPath(ProbabilityPathABC):
    """
    A wrapper around ProbabilityPath that pads the input and output to a common dimension. This is useful for MoE where different paths may have different dimensions but we want to combine them in a single tensor.

    len(paths) = len(mask_list) = 2 should hold. The first path will be responsible for original data dimensions, and the second path will be responsible for the extra dimensions (e.g., for control or steering). The masks indicate which dimensions each path is responsible for.
    """

    def __init__(self, paths: List["ProbabilityPath"], mask_list: List[torch.Tensor]):
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
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        out = torch.zeros(x.shape[0], self.sample_size, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].drift_coeff(t, x[:, self.mask_list[i]])
        return out

    def v(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        out = torch.zeros(x.shape[0], self.sample_size, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].v(t, x[:, self.mask_list[i]])

        return out

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        out = torch.zeros(x.shape[0], self.sample_size, device=x.device)
        for i in range(len(self.paths)):
            out[:, self.mask_list[i]] = self.paths[i].score(t, x[:, self.mask_list[i]])
        return out

    def sigma(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        return self.paths[0].sigma(t)


class MoEProbabilityPath(ProbabilityPathABC):
    def __init__(
        self,
        scheduler: SchedulerABC,  # global noise schedule for MoE path, can be different from individual paths' schedulers
        q_list: List["PaddedProbabilityPath"],
        mask_list: List[torch.Tensor],
        exponent_list: List[Callable[[torch.Tensor], torch.Tensor]],
        sample_size: int,
    ):
        super().__init__()
        self.q_list = q_list
        self.mask_list = mask_list
        self.reverse = self.check_reverse()
        self.exponent_list = exponent_list
        self.scheduler = scheduler

        self.sample_size = sample_size

    def check_reverse(self) -> bool:
        for path in self.q_list:
            if path.reverse != self.q_list[0].reverse:
                raise ValueError("All paths must have the same reverse value")
        return self.q_list[0].reverse

    def drift_coeff(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        """Drift coefficient of MoE SDE, calculated as weighted mixture of component-wise velocities and scores."""
        return self.v(t, x) + 0.5 * self.sigma(t) ** 2 * self.score(t, x)

    def v(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        """Velocity of MoE PF-ODE, calculated as weighted mixture of component-wise velocities."""
        gamma: Float[torch.Tensor, "B E 1"] = torch.stack(  # time-varying exponents
            [exponent_fn(t) for exponent_fn in self.exponent_list], dim=1
        )
        v: Float[torch.Tensor, "B E D"] = torch.stack(
            [pad_tensor(q.v(t, x[:, mask]), self.sample_size, dim=1) for q, mask in zip(self.q_list, self.mask_list)],
            dim=1,
        )
        v_star: Float[torch.Tensor, "B D"] = (gamma * v).sum(dim=1)
        return v_star

    def score(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        """Score function of MoE SDE, calculated as weighted mixture of component-wise scores."""
        gamma: Float[torch.Tensor, "B E 1"] = torch.stack(  # time-varying exponents
            [exponent_fn(t) for exponent_fn in self.exponent_list], dim=1
        )
        s: Float[torch.Tensor, "B E D"] = torch.stack(
            [
                pad_tensor(q.score(t, x[:, mask]), self.sample_size, dim=1).clamp(-20, 20)
                for q, mask in zip(self.q_list, self.mask_list)
            ],
            dim=1,
        )
        s_star: Float[torch.Tensor, "B D"] = (gamma * s).sum(dim=1)
        return s_star

    def sigma(self, t: Float[torch.Tensor, "B 1"]) -> Float[torch.Tensor, "B 1"]:
        """Diffusion coefficient of MoE SDE, calculated as diffusion of global scheduler (not a mixture)."""
        if not self.reverse:
            return self.scheduler.sigma(t)
        else:
            return self.scheduler.sigma(1 - t)

    def get_dlogq(
        self,
        t: Float[torch.Tensor, "B 1"],
        x: Float[torch.Tensor, "B D"],
    ):
        """
        Calculate the logq correction term for each expert, which is used for logq correction in drift and resampling.
        """
        # line 8's logq correction term
        v: Float[torch.Tensor, "B E D"] = torch.stack(
            [pad_tensor(q.v(t, x[:, mask]), self.sample_size, dim=1) for q, mask in zip(self.q_list, self.mask_list)],
            dim=1,
        )
        s: Float[torch.Tensor, "B E D"] = torch.stack(
            [
                pad_tensor(q.score(t, x[:, mask]), self.sample_size, dim=1).clamp(-20, 20)
                for q, mask in zip(self.q_list, self.mask_list)
            ],
            dim=1,
        )

        x_subset_list = [x[:, mask] for mask in self.mask_list]
        div_s = torch.stack(
            [divergence_hutchinson(q.score, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)],
            dim=1,
        )
        div_v = torch.stack(
            [divergence_hutchinson(q.v, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)],
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
        x: Float[torch.Tensor, "B D"],
        use_logq: bool,
        logq_tensor: Optional[Float[torch.Tensor, "B E 1"]],
    ) -> Float[torch.Tensor, " B"]:
        """
        Calculate the time derivative of log weight for each expert, which is used for resampling and logq correction.
        """
        gamma: Float[torch.Tensor, "B E 1"] = torch.stack(  # time-varying exponents
            [exponent_fn(t) for exponent_fn in self.exponent_list], dim=1
        )
        v: Float[torch.Tensor, "B E D"] = torch.stack(
            [pad_tensor(q.v(t, x[:, mask]), self.sample_size, dim=1) for q, mask in zip(self.q_list, self.mask_list)],
            dim=1,
        )
        s: Float[torch.Tensor, "B E D"] = torch.stack(
            [
                pad_tensor(q.score(t, x[:, mask]), self.sample_size, dim=1).clamp(-20, 20)
                for q, mask in zip(self.q_list, self.mask_list)
            ],
            dim=1,
        )
        # dlog_weight except for logq correction
        dlog_weight_base: Float[torch.Tensor, "B 1"] = (
            gamma * ((self.v(t, x).unsqueeze(1) - v) * s).sum(dim=2, keepdim=True)
        ).sum(dim=1)

        # update log weight with logq correction if needed
        if use_logq:
            d_gamma = torch.stack(  # time derivatives of exponents
                [
                    torch.func.jacfwd(exponent_fn, argnums=0)(t).squeeze().diag().unsqueeze(1)  # type: ignore
                    for exponent_fn in self.exponent_list
                ],
                dim=1,
            )
            dlog_weight = dlog_weight_base + (d_gamma * logq_tensor).sum(dim=1)
        else:
            dlog_weight = dlog_weight_base

        return dlog_weight.squeeze(-1)  # (B,)

    def f(
        self,
        t: torch.Tensor,
        x: Float[torch.Tensor, "B D"],
        logq_tensor: Float[torch.Tensor, "B E 1"],
        do_use_logq_tensor: bool,
        calc_dlogq_tensor: bool,
    ):
        """
        - Shapes are seemingly... (Batch, Experts, Data_dim)
        - Base drift and score are calculated with weighted mixture of
          component-wise drifts and scores, where the weights are the time-varying exponents.
        """

        # Algo1. line 3: Mixture score
        gamma: Float[torch.Tensor, "B E 1"] = torch.stack(  # time-varying exponents
            [exponent_fn(t) for exponent_fn in self.exponent_list], dim=1
        )
        s: Float[torch.Tensor, "B E D"] = torch.stack(
            [
                pad_tensor(q.score(t, x[:, mask]), self.sample_size, dim=1).clamp(-20, 20)
                for q, mask in zip(self.q_list, self.mask_list)
            ],
            dim=1,
        )
        s_star: Float[torch.Tensor, "B D"] = (gamma * s).sum(dim=1)

        # Algo1. line 4: Drift
        v: Float[torch.Tensor, "B E D"] = torch.stack(
            [pad_tensor(q.v(t, x[:, mask]), self.sample_size, dim=1) for q, mask in zip(self.q_list, self.mask_list)],
            dim=1,
        )
        sigma_expert = torch.stack([q.sigma(t) for q in self.q_list], dim=1)
        v = v + 0.5 * sigma_expert**2 * s
        v_star: Float[torch.Tensor, "B D"] = (gamma * v).sum(dim=1)
        sigma = self.sigma(t)
        mu: Float[torch.Tensor, "B D"] = v_star + sigma**2 / 2 * s_star

        # Algo1. line 9: update log weight (except for logq correction)
        dlog_weight: Float[torch.Tensor, "B 1"] = (
            gamma * ((v_star.unsqueeze(1) - v) * s).sum(dim=2, keepdim=True)
        ).sum(dim=1)

        # Algo1. line 9: update log weight with previous logq correction
        if do_use_logq_tensor:
            d_gamma = torch.stack(  # time derivatives of exponents
                [
                    torch.func.jacfwd(exponent_fn, argnums=0)(t).squeeze().diag().unsqueeze(1)  # type: ignore
                    for exponent_fn in self.exponent_list
                ],
                dim=1,
            )
            dlog_weight = dlog_weight + (d_gamma * logq_tensor).sum(dim=1)

        # Algo1. line 8 logq correction terms
        dlogq_tensor_drift_term: Union[Float[torch.Tensor, "B E 1"], None] = None
        dlogq_tensor_diffusion_term: Union[Float[torch.Tensor, "B E D"], None] = None
        if do_use_logq_tensor and calc_dlogq_tensor:
            # line 8's logq correction term
            x_subset_list = [x[:, mask] for mask in self.mask_list]
            div_s = torch.stack(
                [divergence_hutchinson(q.score, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)],
                dim=1,
            )
            div_v = torch.stack(
                [divergence_hutchinson(q.v, t, x_subset) for q, x_subset in zip(self.q_list, x_subset_list)],
                dim=1,
            )
            sigma = sigma.unsqueeze(1)  # (B, 1, 1)
            div_v = div_v + sigma**2 / 2 * div_s

            # dlogq_A: (B, num_experts, 1), dlogq_B: (B, num_experts, data_dim)
            dlogq_tensor_drift_term = (
                -div_v + ((mu.unsqueeze(1) - v) * s).sum(dim=2, keepdim=True) + 0.5 * sigma**2 * div_s
            )
            dlogq_tensor_diffusion_term = sigma * s  # dW related term

        return mu, dlog_weight, dlogq_tensor_drift_term, dlogq_tensor_diffusion_term


def pad_tensor(x: Float[torch.Tensor, "B D"], pad_size: int, dim: int):
    padding_shape = list(x.shape)
    padding_shape[dim] = pad_size - x.shape[dim]
    return torch.cat([x, torch.zeros(padding_shape, device=x.device)], dim=dim)


def divergence_hutchinson(
    score_fn: ScoreFunctionType,
    t: torch.Tensor,
    x: torch.Tensor,
    n_probe: int = 1,
    rademacher: bool = True,
    create_graph: bool = False,  # sampling usually doesn't need higher-order grads
    t_eps: float = 1e-4,
    jitter: float = 1e-6,
    grad_clip: float = 1e6,
):
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
        eps = torch.randint_like(x, low=0, high=2).float().mul_(2).sub_(1) if rademacher else torch.randn_like(x)

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
    # print(f'div: {div}')
    return div / n_probe
