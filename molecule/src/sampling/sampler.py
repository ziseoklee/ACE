import functools
import logging
from collections.abc import Callable
from math import ceil

import numpy as np
import torch
from jaxtyping import Float, Int
from torch.distributions import Normal
from tqdm import tqdm

from configs.config_sampler import _BaseSamplerConfig
from sampling.probability_path import MoEProbabilityPath

logger = logging.getLogger(__name__)

ParticleState = Float[torch.Tensor, "B data"]
ExpertLogQ = Float[torch.Tensor, "B E 1"]
ParticleLogWeight = Float[torch.Tensor, "B"]  # noqa: F821
Choices = Int[np.ndarray, "B"]  # noqa: F821
CategoryLogits = Float[torch.Tensor, "K"]  # noqa: F821
StateTrajectory = Float[torch.Tensor, "step B data"]
LogWeightTrajectory = Float[torch.Tensor, "step B"]
LogQTrajectory = Float[torch.Tensor, "step B E 1"]
ChoicesTrajectory = Int[np.ndarray, "step B"]

InterleaveFn = Callable[[ParticleState, Choices], ParticleState]
PostprocessFn = Callable[[ParticleState], ParticleState]


def computation_overhead_logger(func):
    """
    A decorator to log the computation time and peak VRAM usage of a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            result = func(*args, **kwargs)
            logger.info("CUDA is not available; skipped VRAM logging.")
            return result

        device = torch.cuda.current_device()

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # pyright: ignore[reportCallIssue]

        try:
            result = func(*args, **kwargs)
        finally:
            end_event.record()  # pyright: ignore[reportCallIssue]
            torch.cuda.synchronize(device)

            elapsed_time_ms = start_event.elapsed_time(end_event)

            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            current_allocated = torch.cuda.memory_allocated(device) / 1024**3
            current_reserved = torch.cuda.memory_reserved(device) / 1024**3

            logger.info(f"Computation time: {elapsed_time_ms / 1000:.4f} s")
            logger.info(f"Peak allocated VRAM: {peak_allocated:.2f} GB")
            logger.info(f"Peak reserved VRAM: {peak_reserved:.2f} GB")
            logger.info(f"Current allocated VRAM: {current_allocated:.2f} GB")
            logger.info(f"Current reserved VRAM: {current_reserved:.2f} GB")

        return result

    return wrapper


class MoEPDESampler:
    """
    A sampler for Mixture-of-Experts Probability Paths (MoE-PDEs) using Euler-Maruyama discretization and optional resampling.
    """

    @staticmethod
    def initialize_particles(
        moe_probability_path: MoEProbabilityPath,
        prior_sbdd: ParticleState,
        batch_size: int,
        device: str,
        seed: int,
    ) -> tuple[ParticleState, ExpertLogQ, ParticleLogWeight]:
        """
        Initialize x0, logq, and log weights tensor for the sampler.
        """
        sample_size: int = moe_probability_path.sample_size
        generator = torch.Generator(device=device).manual_seed(seed)
        x0: ParticleState = torch.randn(
            batch_size,
            sample_size,
            generator=generator,
            device=device,
        ).to(device)
        # !WARNING!: we assume the 4th expert is the DiffSBDD expert.
        mask_sbdd = moe_probability_path.q_list[3].mask_list[0]
        x0[..., mask_sbdd] = prior_sbdd

        standard_normal_dist = Normal(loc=0.0, scale=1.0)
        logq = standard_normal_dist.log_prob(x0)
        num_experts = len(moe_probability_path.q_list)
        logq: ExpertLogQ = logq.sum(dim=-1, keepdim=True).repeat(1, num_experts).unsqueeze(2)

        logweight: ParticleLogWeight = torch.zeros(batch_size, device=device)

        return x0, logq, logweight

    @classmethod
    @computation_overhead_logger
    def sample(
        cls,
        moe_probability_path: MoEProbabilityPath,
        sampler_cfg: _BaseSamplerConfig,
        prior_sbdd: ParticleState,
        interleave_fns: list[InterleaveFn] | None = None,
        postprocess_fns: list[PostprocessFn] | None = None,
    ) -> tuple[ParticleState, StateTrajectory, LogWeightTrajectory, LogQTrajectory, ChoicesTrajectory]:
        """Sample MoE paths with SDE updates before ``ode_start_t`` and ODE updates after it.

        During the final ODE phase, particles are propagated deterministically with
        the PF-ODE velocity. The logq tensor, log weights, and resampling choices
        are intentionally frozen in this phase; this is an approximation used to
        avoid endpoint logq/reweighting instabilities while still applying the
        interleave functions to keep conditioning state aligned.
        """
        batch_size = sampler_cfg.batch_size
        device = sampler_cfg.device
        seed = sampler_cfg.seed
        num_sampling_steps = sampler_cfg.num_sampling_steps
        dt = 1 / num_sampling_steps
        timesteps = torch.arange(0, 1, dt).to(device)

        use_logq = sampler_cfg.use_logq
        dlogq_calc_interval = sampler_cfg.dlogq_calc_interval
        dlogq_noise_scale = sampler_cfg.dlogq_noise_scale

        do_resample = sampler_cfg.do_resample
        resampling_step_interval = sampler_cfg.resampling_step_interval

        # Initializations
        x, logq_tensor, logweight_tensor = cls.initialize_particles(
            moe_probability_path=moe_probability_path,
            prior_sbdd=prior_sbdd,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        x.requires_grad = True

        # For trajectory storage
        x_tensor_list = []
        logweight_tensor_list = []
        logq_tensor_list = []
        choices = []

        log_ode_switched = False  # Flag to indicate if ODE solver has been switched
        ode_start_step = ceil(sampler_cfg.ode_start_t * num_sampling_steps)  # Step at which to switch to ODE solver
        for step, t in enumerate(tqdm(timesteps, desc="MoE Sampling"), start=0):
            if t.dim() == 0:
                t = t * torch.ones(batch_size, 1).to(x.device)

            if step >= ode_start_step:
                if not log_ode_switched:
                    logger.info(
                        f"From step {step} (t={t[0].item():.4f}), switching to ODE solver for deterministic propagation."
                        "It is recommended for numerical stability near t=1."
                    )
                    logger.info(
                        "Note: This will freeze the logq tensor, log weights, and disable resampling for the remaining steps."
                    )
                    log_ode_switched = True

                # ==== ODE Sampling Step =====
                moe_v = moe_probability_path.v(t, x)
                x_next = x + moe_v * dt
                logq_tensor_next = logq_tensor
                logweight_tensor_next = logweight_tensor
                choice = torch.arange(batch_size).numpy()

            else:
                # ===== SDE Sampling Step =====
                # moe_score = probability_path.score(t, x)  # Algorithm  L3 (calculated internally in probability_path's methods)
                moe_mu = moe_probability_path.drift_coeff(t, x)  # Algorithm  L4
                moe_sigma = moe_probability_path.sigma(t)

                # Algorithm L7: Propagate particles with Euler-Maruyama
                dW = torch.randn_like(x) * np.sqrt(dt)  # noqa: N806
                x_next = x + moe_mu * dt + moe_sigma * dW

                # Algorithm L8: Update logq tensor if needed
                # NOTE: This update is done for every (step % dlogq_calc_interval == 0) step where `use_logq` is True.
                # NOTE: But the magnitude of update is scaled by `dlogq_calc_interval` to reflect the fact that we are effectively taking a bigger step in time for logq correction.
                if use_logq and step % dlogq_calc_interval == 0:
                    dlogq_tensor_drift_term, dlogq_tensor_diffusion_term = moe_probability_path.get_dlogq(t, x)

                    logq_tensor_next = logq_tensor + dlogq_tensor_drift_term * (dt * dlogq_calc_interval)
                    # `dlogq_noise_scale` is a hyperparameter that scales the noise added to logq correction. The noise is added to account for the stochasticity in the diffusion term and can help stabilize inference.
                    logq_tensor_next = logq_tensor_next + torch.einsum(
                        "bij,bij->bi",
                        dlogq_tensor_diffusion_term,
                        dW.unsqueeze(1) * np.sqrt(dlogq_calc_interval) * dlogq_noise_scale,
                    ).unsqueeze(2)
                else:
                    logq_tensor_next = logq_tensor

                # Algorithm L9: Update log weights
                dlog_weight = moe_probability_path.get_dlog_weight(t, x, use_logq=use_logq, logq_tensor=logq_tensor)
                logweight_tensor_next = logweight_tensor + dlog_weight * dt

                # Algorithm 11-14: Resampling (if needed)
                if do_resample and step % resampling_step_interval == 0:
                    choice = cls.resample_particles(logweight_tensor_next, batch_size)
                    x_next = x_next[choice]
                    logq_tensor_next = logq_tensor_next[choice]
                    # Algorithm L14: Reset log weights to zero after resampling
                    logweight_tensor_next = torch.zeros_like(logweight_tensor_next)
                else:
                    choice = torch.arange(batch_size).numpy()

            x_next = cls.apply_interleave_fns(x_next, choice, interleave_fns)

            # Store trajectories
            x_tensor_list.append(x_next.detach().cpu())
            logweight_tensor_list.append(logweight_tensor_next.detach().cpu())
            logq_tensor_list.append(logq_tensor_next.detach().cpu())
            choices.append(choice)

            # Update for next iteration
            x = x_next
            logq_tensor = logq_tensor_next
            logweight_tensor = logweight_tensor_next

        x_trajectory = torch.stack(x_tensor_list)
        x1 = cls.apply_postprocess_fns(x, postprocess_fns)
        logweight_trajectory = torch.stack(logweight_tensor_list)
        logq_trajectory = torch.stack(logq_tensor_list)
        choices = np.array(choices)

        return x1.detach(), x_trajectory, logweight_trajectory, logq_trajectory, choices

    @staticmethod
    def apply_interleave_fns(
        x: ParticleState,
        choices: Choices,
        interleave_fns: list[InterleaveFn] | None,
    ) -> ParticleState:
        if interleave_fns is None:
            return x

        for interleave_fn in interleave_fns:
            x = interleave_fn(x, choices)
        return x

    @staticmethod
    def apply_postprocess_fns(
        x: ParticleState,
        postprocess_fns: list[PostprocessFn] | None,
    ) -> ParticleState:
        if postprocess_fns is None:
            return x

        for postprocess_fn in postprocess_fns:
            x = postprocess_fn(x)
        return x

    @staticmethod
    def resample_particles(
        logits: CategoryLogits,
        num_out_particles: int,  # (B,)
        tol: float = 1e-6,
        stratified: bool = True,
    ) -> Choices:
        """
        Draw one categorical sample per row of `logits` using stratified uniforms.
        - Collapses tiny probs (<= tol) to 0 and renormalizes.
        - Uses torch.searchsorted on the CDF to avoid np.digitize monotonicity errors.

        Args:
            logits: shape (K, 1)
            num_out_particles: number of particles to sample (i.e. output batch size), shape (B,)
            tol: probabilities <= tol are collapsed to zero
            stratified: use stratified uniforms across [0,1)

        Returns:
            ids: LongTensor of shape (B,) with chosen category per row
            None: placeholder to match your original signature
        """
        logits = logits.unsqueeze(0).expand(num_out_particles, -1)
        num_rows, num_categories = logits.shape

        # Stable softmax, then collapse tiny probs
        probs = torch.softmax(logits, dim=-1)
        probs = torch.where(probs <= tol, torch.zeros_like(probs), probs)

        # Renormalize (rows that got fully zeroed become uniform)
        row_sum = probs.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / num_categories)
        probs = torch.where(row_sum > 0, probs / row_sum.clamp_min(torch.finfo(probs.dtype).eps), uniform)

        # CDF (non-decreasing; duplicates OK)
        cdf = torch.cumsum(probs, dim=-1).clamp(max=1.0)

        # Stratified uniforms u in [0,1)
        if stratified:
            base = torch.rand((), device=logits.device, dtype=logits.dtype)
            # center within each stratum (optional but nice)
            u = (base + (torch.arange(num_rows, device=logits.device, dtype=logits.dtype) + 0.5) / num_rows) % 1.0
        else:
            u = torch.rand(num_rows, device=logits.device, dtype=logits.dtype)

        # Use torch.searchsorted to find the indices where u would be inserted to maintain order in cdf
        ids = torch.searchsorted(cdf, u.unsqueeze(-1), right=True).squeeze(-1)
        ids = ids.clamp_(0, num_categories - 1)

        return ids.cpu().numpy()
