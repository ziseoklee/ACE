"""ACE/FKC/NR sampling for compositional Stable Diffusion generation.

The three samplers intentionally share one implementation:

* ``nr`` propagates particles with the composed score and no weights.
* ``fkc`` adds Algorithm 2 importance weights with constant exponents.
* ``ace`` adds Algorithm 1 log-density tracking and the endpoint-preserving
  exponent correction ``B t (1-t)``.

HFKC was the development name for FKC/ACE in the original research code.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image

from ace_schedule import (
    quadratic_bump,
    quadratic_bump_derivative,
    scheduled_steps,
    validate_unit_interval,
)
from core import diffusion

logger = logging.getLogger(__name__)


def _sum_except_batch(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).sum(dim=1)


def _rademacher_like(value: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    sample = torch.randint(0, 2, value.shape, device=value.device, generator=generator)
    return sample.to(value.dtype).mul_(2).sub_(1)


class StableDiffusionACEPipelineWrapper:
    """Paper-aligned ACE sampler for Stable Diffusion 1.5 and 2.1."""

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.vae: AutoencoderKL = pipe.vae
        self.backbone: UNet2DConditionModel = pipe.unet
        self.scheduler: DDIMScheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe.scheduler = self.scheduler

        self.device = pipe.device
        self.dtype = self.backbone.dtype
        self.latent_dim = int(self.backbone.config.in_channels)
        self.latent_size = int(pipe.default_sample_size)
        self.image_size = self.latent_size * int(pipe.vae_scale_factor)

    def encode_text(
        self,
        prompt: str | list[str],
        *,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
        )
        if not isinstance(negative_embeds, torch.Tensor):
            raise TypeError("Stable Diffusion did not return unconditional prompt embeddings.")
        return negative_embeds, prompt_embeds

    def _upcast_vae(self) -> torch.dtype:
        if self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False):
            self.pipe.upcast_vae()
        post_quant = getattr(self.vae, "post_quant_conv", None)
        return self.vae.dtype if post_quant is None else post_quant.weight.dtype

    def decode_image(self, latents: torch.Tensor) -> list[Image.Image]:
        dtype = self._upcast_vae()
        latents = latents.to(dtype)
        config = self.vae.config
        mean = getattr(config, "latents_mean", None)
        std = getattr(config, "latents_std", None)
        if mean is not None and std is not None:
            mean_tensor = torch.tensor(mean, device=self.device, dtype=dtype).view(1, 4, 1, 1)
            std_tensor = torch.tensor(std, device=self.device, dtype=dtype).view(1, 4, 1, 1)
            latents = latents * std_tensor / config.scaling_factor + mean_tensor
        else:
            latents = latents / config.scaling_factor
        shift = getattr(config, "shift_factor", 0) or 0
        images = self.vae.decode(latents + shift).sample
        return self.pipe.image_processor.postprocess(images.detach())

    def _build_masks(self, boxes: list[tuple[float, float, float, float]]) -> torch.Tensor:
        masks = []
        for box in boxes:
            if len(box) != 4:
                raise ValueError(f"Each box must contain four coordinates, got {box!r}.")
            x0, y0, x1, y1 = (float(value) for value in box)
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(f"Boxes must be normalized xyxy coordinates, got {box!r}.")
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            mask[
                int(y0 * self.image_size) : math.ceil(y1 * self.image_size),
                int(x0 * self.image_size) : math.ceil(x1 * self.image_size),
            ] = 1.0
            masks.append(mask[None])
        if not masks:
            return torch.empty((0, 1, self.latent_size, self.latent_size), device=self.device, dtype=self.dtype)
        tensor = torch.from_numpy(np.stack(masks)).to(device=self.device, dtype=self.dtype)
        return F.interpolate(tensor, (self.latent_size, self.latent_size), mode="nearest")

    def _predict_pair(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        unconditional_embeds: torch.Tensor,
        conditional_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = torch.cat((latents, latents), dim=0)
        model_input = self.scheduler.scale_model_input(model_input, timestep)
        embeddings = torch.cat((unconditional_embeds, conditional_embeds), dim=0)
        prediction = self.backbone(
            sample=model_input,
            timestep=timestep,
            encoder_hidden_states=embeddings,
        ).sample
        return prediction.chunk(2)

    @staticmethod
    def _ddim_sigma(alpha_t: torch.Tensor, alpha_previous: torch.Tensor, eta: float) -> torch.Tensor:
        if eta == 0.0:
            return torch.zeros_like(alpha_t)
        variance = (1.0 - alpha_previous) / (1.0 - alpha_t).clamp_min(1e-12)
        variance = variance * (1.0 - alpha_t / alpha_previous).clamp_min(0.0)
        return float(eta) * variance.clamp_min(0.0).sqrt()

    @staticmethod
    def _ddim_displacement(
        latents: torch.Tensor,
        epsilon: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_previous: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        beta_t = 1.0 - alpha_t
        predicted_x0 = (latents - beta_t.sqrt() * epsilon) / alpha_t.sqrt()
        epsilon_coefficient = (1.0 - alpha_previous - sigma.square()).clamp_min(0.0).sqrt()
        mean_previous = alpha_previous.sqrt() * predicted_x0 + epsilon_coefficient * epsilon
        return mean_previous - latents

    def _local_log_ratio_divergence(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        masks: torch.Tensor,
        phrase_unconditional: torch.Tensor,
        phrase_conditional: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_previous: torch.Tensor,
        sigma: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Estimate ``div[-(dv_c-dv_u)+sigma²/2(s_c-s_u)]`` per ROI.

        This is the pairwise divergence contribution in Algorithm 1's update
        for ``log q(F_i|c_i) - log q(F_i)``.  A single Rademacher probe gives
        the Hutchinson estimate independently for every ROI and particle.
        """
        num_regions, batch = masks.shape[:2]
        with torch.enable_grad():
            expanded = latents.detach().unsqueeze(0).expand(num_regions, -1, -1, -1, -1).clone()
            expanded.requires_grad_(True)
            masked = expanded * masks
            flat = masked.flatten(0, 1)
            flat_u = phrase_unconditional.flatten(0, 1)
            flat_c = phrase_conditional.flatten(0, 1)
            epsilon_u, epsilon_c = self._predict_pair(flat, timestep, flat_u, flat_c)
            epsilon_u = epsilon_u.view_as(masked)
            epsilon_c = epsilon_c.view_as(masked)

            score_difference = -(epsilon_c - epsilon_u) / (1.0 - alpha_t).sqrt().clamp_min(1e-6)
            score_difference = score_difference * masks
            displacement_u = self._ddim_displacement(masked, epsilon_u, alpha_t, alpha_previous, sigma) * masks
            displacement_c = self._ddim_displacement(masked, epsilon_c, alpha_t, alpha_previous, sigma) * masks
            field = -(displacement_c - displacement_u) + 0.5 * sigma.square() * score_difference

            probe = _rademacher_like(field, generator)
            projected = (field.float() * probe.float()).sum()
            gradient = torch.autograd.grad(projected, expanded, create_graph=False)[0]
            divergence = (gradient * probe).float().flatten(2).sum(dim=2)
        return divergence.detach().view(num_regions, batch)

    def generate(
        self,
        prompt: str,
        tags: list[str],
        phrases: list[str],
        boxes: list[tuple[float, float, float, float]],
        *,
        neg_prompt: str | None = None,
        steps: int = 50,
        cfg: float = 7.5,
        cfgpp: float = 0.0,
        local_guidance: float = 7.5,
        gamma: list[float] | float = 1.0,
        B: float = 5.0,
        generator: torch.Generator | None = None,
        N: int = 3,
        eta: float = 1.5,
        sampler: str = "ace",
        resample: bool = True,
        resample_at: list[float] | tuple[float, ...] = (0.3,),
        resample_mode: str = "scheduled",
        ess_threshold: float = 0.5,
        resample_start: float = 0.0,
        resample_end: float = 1.0,
        feather: int = 0,
    ) -> tuple[list[Image.Image], torch.Tensor]:
        """Generate compositional images with NR, FKC, or ACE.

        ``B=5``, ``N=3``, and one scheduled resampling at ``t=0.3`` are the
        image settings reported in Appendix E.4 of the ACE paper.
        """
        del tags, feather  # accepted for compatibility with the benchmark API
        sampler = sampler.lower()
        if sampler not in {"nr", "fkc", "ace"}:
            raise ValueError(f"sampler must be one of nr/fkc/ace, got {sampler!r}.")
        if resample_mode not in {"scheduled", "ess"}:
            raise ValueError("resample_mode must be 'scheduled' or 'ess'.")
        if cfgpp != 0.0:
            raise ValueError("ACE's Stable Diffusion reproduction path does not implement CFG++.")
        if steps < 1 or N < 1:
            raise ValueError("steps and N must be positive.")
        if eta < 0.0:
            raise ValueError("eta must be non-negative.")
        if len(phrases) != len(boxes) or not boxes:
            raise ValueError("phrases and boxes must have the same non-zero length.")
        if not 0.0 < ess_threshold <= 1.0:
            raise ValueError("ess_threshold must lie in (0, 1].")
        validate_unit_interval(resample_start, name="resample_start")
        validate_unit_interval(resample_end, name="resample_end")
        if resample_start > resample_end:
            raise ValueError("resample_start cannot exceed resample_end.")

        num_regions = len(boxes)
        if isinstance(gamma, (float, int)):
            gamma_values = [float(gamma)] * num_regions
        else:
            gamma_values = [float(value) for value in gamma]
        if len(gamma_values) != num_regions:
            raise ValueError("gamma must be scalar or have one value per region.")

        bump_strength = float(B) if sampler == "ace" else 0.0
        weighted = sampler in {"fkc", "ace"}
        schedule = scheduled_steps(tuple(resample_at), steps) if weighted and resample else ()
        last_scheduled_step = schedule[-1] if schedule and resample_mode == "scheduled" else steps - 1
        logger.info(
            "Sampler=%s, B=%g, eta=%g, particles=%d, resampling=%s",
            sampler.upper(),
            bump_strength,
            eta,
            N,
            schedule if resample_mode == "scheduled" else f"ESS<{ess_threshold}N",
        )

        prompt_u, prompt_c = self.encode_text(
            prompt,
            negative_prompt=neg_prompt,
            num_images_per_prompt=N,
        )
        phrases_u, phrases_c = self.encode_text(
            phrases,
            negative_prompt=neg_prompt,
            num_images_per_prompt=N,
        )
        sequence_length, embedding_dim = phrases_u.shape[-2:]
        phrases_u = phrases_u.view(num_regions, N, sequence_length, embedding_dim)
        phrases_c = phrases_c.view(num_regions, N, sequence_length, embedding_dim)

        masks = self._build_masks(boxes).unsqueeze(1).expand(-1, N, -1, -1, -1).contiguous()
        base_local_strength = torch.tensor(
            gamma_values,
            device=self.device,
            dtype=self.dtype,
        ) * float(local_guidance)
        base_local_strength = base_local_strength.view(num_regions, 1, 1, 1, 1)

        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = torch.randn(
            (N, self.latent_dim, self.latent_size, self.latent_size),
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        ) * self.scheduler.init_noise_sigma
        log_weights = torch.zeros(N, device=self.device, dtype=torch.float32)
        log_ratios = torch.zeros((num_regions, N), device=self.device, dtype=torch.float32)
        dt = 1.0 / float(steps)

        for step_index, timestep in enumerate(timesteps):
            # Algorithm 1 evaluates coefficients at t=k/T and propagates to
            # (k+1)/T, so the final loop iteration uses t=1-dt.
            progress = step_index / float(steps)
            bump = quadratic_bump(progress, bump_strength)
            bump_derivative = quadratic_bump_derivative(progress, bump_strength)
            local_exponents = base_local_strength + bump

            timestep_index = int(timestep.item())
            previous_index = timestep_index - self.scheduler.config.num_train_timesteps // steps
            alpha_t = self.scheduler.alphas_cumprod[timestep_index].to(device=self.device, dtype=self.dtype)
            alpha_previous = (
                self.scheduler.alphas_cumprod[previous_index]
                if previous_index >= 0
                else self.scheduler.final_alpha_cumprod
            ).to(device=self.device, dtype=self.dtype)
            sigma = self._ddim_sigma(alpha_t, alpha_previous, eta)
            score_scale = (1.0 - alpha_t).sqrt().clamp_min(1e-6)

            current_latents = latents
            with torch.no_grad():
                epsilon_global_u, epsilon_global_c = self._predict_pair(latents, timestep, prompt_u, prompt_c)
                score_global_u = -epsilon_global_u / score_scale
                score_global_c = -epsilon_global_c / score_scale
                displacement_global_u = self._ddim_displacement(
                    latents, epsilon_global_u, alpha_t, alpha_previous, sigma
                )
                displacement_global_c = self._ddim_displacement(
                    latents, epsilon_global_c, alpha_t, alpha_previous, sigma
                )

                expanded = latents.unsqueeze(0).expand(num_regions, -1, -1, -1, -1)
                masked = expanded * masks
                epsilon_local_u, epsilon_local_c = self._predict_pair(
                    masked.flatten(0, 1),
                    timestep,
                    phrases_u.flatten(0, 1),
                    phrases_c.flatten(0, 1),
                )
                epsilon_local_u = epsilon_local_u.view_as(masked)
                epsilon_local_c = epsilon_local_c.view_as(masked)
                score_local_u = (-epsilon_local_u / score_scale) * masks
                score_local_c = (-epsilon_local_c / score_scale) * masks
                displacement_local_u = self._ddim_displacement(
                    masked, epsilon_local_u, alpha_t, alpha_previous, sigma
                ) * masks
                displacement_local_c = self._ddim_displacement(
                    masked, epsilon_local_c, alpha_t, alpha_previous, sigma
                ) * masks

                epsilon_composed = cfg * epsilon_global_c + (1.0 - cfg) * epsilon_global_u
                epsilon_composed = epsilon_composed + (
                    local_exponents * (epsilon_local_c - epsilon_local_u) * masks
                ).sum(dim=0)
                score_composed = cfg * score_global_c + (1.0 - cfg) * score_global_u
                score_composed = score_composed + (
                    local_exponents * (score_local_c - score_local_u)
                ).sum(dim=0)
                displacement_composed = self._ddim_displacement(
                    latents, epsilon_composed, alpha_t, alpha_previous, sigma
                )

                correction_active = weighted and step_index <= last_scheduled_step
                if correction_active:
                    potential = cfg * _sum_except_batch(
                        (displacement_composed - displacement_global_c) * score_global_c
                    )
                    potential += (1.0 - cfg) * _sum_except_batch(
                        (displacement_composed - displacement_global_u) * score_global_u
                    )
                    composed_roi = displacement_composed.unsqueeze(0) * masks
                    local_potential = (composed_roi - displacement_local_c) * score_local_c
                    local_potential -= (composed_roi - displacement_local_u) * score_local_u
                    potential += (
                        local_exponents.squeeze(-1).squeeze(-1).squeeze(-1)
                        * local_potential.float().flatten(2).sum(dim=2)
                    ).sum(dim=0)
                    log_weights += potential.float()
                    if sampler == "ace":
                        log_weights += bump_derivative * log_ratios.sum(dim=0) * dt
                    log_weights -= log_weights.max()

                noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
                latents = latents + displacement_composed + sigma * noise

            if correction_active and sampler == "ace":
                divergence = self._local_log_ratio_divergence(
                    current_latents,
                    timestep,
                    masks,
                    phrases_u,
                    phrases_c,
                    alpha_t,
                    alpha_previous,
                    sigma,
                    generator,
                )
                with torch.no_grad():
                    score_difference = score_local_c - score_local_u
                    composed_roi = displacement_composed.unsqueeze(0) * masks
                    non_divergence = (composed_roi - displacement_local_c) * score_local_c
                    non_divergence -= (composed_roi - displacement_local_u) * score_local_u
                    score_interaction = 0.5 * sigma.square() * (
                        score_composed.unsqueeze(0) * masks * score_difference
                    )
                    stochastic = sigma * score_difference * noise.unsqueeze(0)
                    log_ratio_increment = divergence
                    log_ratio_increment += non_divergence.float().flatten(2).sum(dim=2)
                    log_ratio_increment += score_interaction.float().flatten(2).sum(dim=2)
                    log_ratio_increment += stochastic.float().flatten(2).sum(dim=2)
                    log_ratios += log_ratio_increment

            if not correction_active or not resample:
                continue
            normalized_weights = torch.softmax(log_weights, dim=0)
            in_window = resample_start <= progress <= resample_end
            if resample_mode == "scheduled":
                should_resample = step_index in schedule
            else:
                ess = normalized_weights.square().sum().reciprocal()
                should_resample = in_window and bool(ess < ess_threshold * N)
            if should_resample:
                ancestors = torch.multinomial(normalized_weights, N, replacement=True, generator=generator)
                latents = latents[ancestors]
                log_ratios = log_ratios[:, ancestors]
                log_weights.zero_()

        final_weights = torch.softmax(log_weights, dim=0)
        return self.decode_image(latents), final_weights


def _parse_boxes(flat_coordinates: list[float]) -> list[tuple[float, float, float, float]]:
    if len(flat_coordinates) % 4:
        raise ValueError("--boxes must contain a multiple of four coordinates.")
    return [tuple(flat_coordinates[index : index + 4]) for index in range(0, len(flat_coordinates), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description="ACE/FKC/NR compositional Stable Diffusion demo")
    parser.add_argument("--arch", choices=["SD1.5", "SD2.1"], default="SD2.1")
    parser.add_argument("--sampler", choices=["nr", "fkc", "ace"], default="ace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--local-guidance", type=float, default=7.5)
    parser.add_argument("--bump", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=1.5)
    parser.add_argument("--resample-at", type=float, nargs="*", default=[0.3])
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--phrases", nargs="+", required=True)
    parser.add_argument("--boxes", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ace_demo"))
    args = parser.parse_args()

    base_pipe = diffusion.build_pipeline(args.arch)
    pipe = StableDiffusionACEPipelineWrapper(cast(StableDiffusionPipeline, base_pipe))
    generator = torch.Generator(pipe.device).manual_seed(args.seed)
    images, weights = pipe.generate(
        prompt=args.prompt,
        tags=args.tags,
        phrases=args.phrases,
        boxes=_parse_boxes(args.boxes),
        steps=args.steps,
        cfg=args.cfg,
        local_guidance=args.local_guidance,
        B=args.bump,
        generator=generator,
        N=args.N,
        eta=args.eta,
        sampler=args.sampler,
        resample=not args.no_resample,
        resample_at=args.resample_at,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (image, weight) in enumerate(zip(images, weights)):
        image.save(args.output_dir / f"particle_{index:02d}_weight_{weight.item():.6f}.png")


if __name__ == "__main__":
    main()
