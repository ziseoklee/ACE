# Copyright (C) 2025 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import math
import torch
import einops
import numpy as np
import sanghyunjo as shjo
import sanghyunjo.ai_utils as shai

from PIL import Image
from typing import Optional, List, Tuple

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from core import diffusion

from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention_processor import Attention

SD_MODELS = ["StableDiffusionPipeline"]
UNET_GROUP = SD_MODELS

class UnetPipeline: # for UNet-based Diffusion Models
    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe: StableDiffusionPipeline = pipe
        self.pipeline_name = pipe.__class__.__name__

        self.vae: AutoencoderKL = pipe.vae
        self.backbone: UNet2DConditionModel = pipe.unet
        self.scheduler: DDIMScheduler = pipe.scheduler

        self.device = pipe.device
        self.dtype = self.backbone.dtype

        self.latent_dim = self.backbone.config.in_channels # e.g., 4 for SDXL
        self.latent_size = pipe.default_sample_size # e.g., 128 for SDXL
        self.patch_size = pipe.vae_scale_factor # e.g., 8 for SDXL
        self.image_size = self.latent_size * self.patch_size # e.g., 1024 for SDXL

        # related to attention hooks
        self.latent_ratio = (1.0, 1.0) # default square ratio for latents
        self.reset_hook_cache() # reset attention maps
    
    def encode_text(self, prompt, neg_prompt=None, do_cfg=True) -> torch.Tensor:
        """
        Encode textual prompts into embeddings for diffusion guidance.

        Args:
            prompt (str or List[str]): The input text prompt(s).
            neg_prompt (str or List[str], optional): The negative prompt(s) for classifier-free guidance.
            do_cfg (bool): Whether to apply classifier-free guidance (CFG).

        Returns:
            Tuple[Tensor, Optional[Tensor]]: 
                - prompt_embeds: The (possibly CFG-augmented) text embeddings.
                - pooled_prompt_embeds: (Only for SDXL) Global pooled embeddings, or None otherwise.
        """
        # Common parameters passed into the pipeline's `encode_prompt` method
        encode_params = {
            'negative_prompt': neg_prompt,
            'device': self.device,
            'num_images_per_prompt': 1,
            'do_classifier_free_guidance': do_cfg,
        }

        if self.pipeline_name in SD_MODELS:
            # Standard Stable Diffusion (SD 1.x / 2.x)
            prompt_embeds, neg_embeds = self.pipe.encode_prompt(prompt=prompt, **encode_params)
        else:
            raise ValueError(f"Unsupported pipeline: {self.pipeline_name}")

        # Apply classifier-free guidance (duplicate batch with negative guidance)
        if do_cfg:
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0).to(self.device)

        return prompt_embeds

    def upcast_vae(self) -> torch.dtype:
        """
        Upcast VAE to float32 if required (usually from float16).
        Returns the actual dtype of VAE's weights used in encoding/decoding.
        """
        try:
            if self.vae.dtype == torch.float16 and getattr(self.vae.config, 'force_upcast', False):
                self.pipe.upcast_vae()
        except AttributeError:
            pass

        # Return the dtype used by the VAE (typically the post_quant_conv layer's weights)
        post_quant_layer = getattr(self.vae, 'post_quant_conv', None)
        return self.vae.dtype if post_quant_layer is None else post_quant_layer.weight.dtype

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode input image(s) into latent space using the VAE encoder.

        Args:
            images (Tensor): Input images in [B, C, H, W] format.

        Returns:
            Tensor: Latent representation (after scaling and optional shift).
        """
        dtype = self.upcast_vae()
        posterior = self.vae.encode(images.to(dtype=dtype)).latent_dist

        shift = getattr(self.vae.config, 'shift_factor', 0) or 0
        latents = posterior.mean - shift if shift else posterior.mean

        return latents * self.vae.config.scaling_factor
    
    def encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(mask, self.pipe.vae_scale_factor)

    def decode_image(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode latent tensor back into image(s) using the VAE decoder.

        Args:
            latents (Tensor): Latent representation.

        Returns:
            Tensor: Reconstructed image(s).
        """
        dtype = self.upcast_vae()
        latents = latents.to(dtype)

        # Handle normalization stats if present
        cfg = self.vae.config
        mean, std = getattr(cfg, 'latents_mean', None), getattr(cfg, 'latents_std', None)

        if mean is not None and std is not None:
            mean = torch.tensor(mean).view(1, 4, 1, 1).to(self.device, dtype)
            std = torch.tensor(std).view(1, 4, 1, 1).to(self.device, dtype)
            latents = latents * std / cfg.scaling_factor + mean
        else:
            latents = latents / cfg.scaling_factor

        shift = getattr(cfg, 'shift_factor', 0) or 0
        images = self.vae.decode(latents + shift).sample

        return self.pipe.image_processor.postprocess(images.detach())
    
    def get_denoising_params(self, latents: torch.Tensor, timestep: torch.Tensor, do_cfg: bool, prompt_embeds: torch.Tensor) -> dict:
            """
            Construct input dict for UNet forward during diffusion denoising step.
            """
            latent_input = torch.cat([latents] * prompt_embeds.shape[0]) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, timestep)

            params = {
                'sample': latent_input,
                'timestep': timestep,
                'encoder_hidden_states': prompt_embeds,
            }

            return params
    
    def hook_attention(self, hooks: list = ['enc', 'mid', 'dec']) -> None:
        """
        Register `self` as processor for attention layers in the UNet backbone.
        This enables analysis, intervention, or modification during forward pass.

        Args:
            hooks (list): List of UNet regions to hook. 
                        Options: 'enc' (down_blocks), 'mid' (mid_block), 'dec' (up_blocks)
        """
        # Map hook keys to corresponding module blocks in UNet
        hook_targets = {
            'enc': self.backbone.down_blocks,
            'mid': [self.backbone.mid_block],
            'dec': self.backbone.up_blocks,
        }

        def find_attention_layers(module) -> List[Attention]:
            """
            Recursively find CrossAttention blocks and return their self-attention layers.
            """
            attn_layers = []

            # Identify CrossAttn modules by class name (e.g., `CrossAttnDownBlock2D`)
            if 'CrossAttn' in module.__class__.__name__:
                for cross_attn_group in module.attentions:
                    for transformer in cross_attn_group.transformer_blocks:
                        attn_layers.append(transformer.attn1) # self-attention layer
                        attn_layers.append(transformer.attn2) # cross-attention layer
                        
                        """
                        Hook CrossAttnDownBlock2D BasicTransformerBlock
                        ...
                        Hook UNetMidBlock2DCrossAttn BasicTransformerBlock
                        ...
                        Hook CrossAttnUpBlock2D BasicTransformerBlock
                        ...
                        """
                        # print(f"Hook {module.__class__.__name__} {transformer.__class__.__name__}")
            
            return attn_layers

        for target in hooks:
            index = 1
            for block in hook_targets[target]:
                for module in find_attention_layers(block):
                    module.set_processor(self)

                    module._hook_name = f"{target}_{index:04d}" # tag for identification
                    index += 1
        
        # print(f"Hooked attention layers in {self.pipeline_name} backbone.")

    def reset_hook_cache(self):
        """
        Clear any cached attention maps
        """
        self.attn_self: Optional[torch.Tensor] = 0.
        self.count_self = 0

        self.attn_cross: Optional[torch.Tensor] = 0.
        self.count_cross = 0

    def calculate_latent_wh(self, flattened_size):
        rw, rh = self.latent_ratio
        total_ratio = rw * rh
        lwh = (flattened_size / total_ratio) ** 0.5
        return int(rw * lwh), int(rh * lwh)
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        elif attn.norm_cross: encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key = attn.norm_k(key)

        # ⚠️ original attention calculation (not providing attention hooks)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        # ⭐ Attention Hooking using eniops
        attention_probs = torch.softmax(einops.einsum(query, key, 'b h l d, b h m d -> b h l m'), dim=-1)
        
        # reshape self- and cross-attention maps @sanghyun jo
        attn_w, attn_h = self.calculate_latent_wh(query.shape[2])
        attn_dim = attention_probs.shape[-1]

        attn_map = attention_probs.mean(dim=1).permute(0, 2, 1) # B x HW x C to B x C x HW
        attn_map = attn_map.view(batch_size, attn_dim, attn_h, attn_w) # B x C x H x W
        
        # check using cfg for multiple batch images (batch * 2 is the same as using cfg)
        attn_map = attn_map[1 if attn_map.shape[0] > 1 else 0] # for conditional heatmaps
        
        # for cross-attention maps
        if (attn_w * attn_h) != attn_dim:
            if isinstance(self.attn_cross, torch.Tensor):
                attn_map = shai.resize(attn_map, self.attn_cross, mode='bilinear')
            
            self.attn_cross = self.attn_cross + attn_map
            self.count_cross += 1
        
        # for self-attention maps
        else:
            if isinstance(self.attn_self, torch.Tensor):
                lh, lw = attn_map.shape[1:]
                size = self.attn_self.shape[1:]

                attn_map = attn_map.view(lh, lw, lh, lw)
                attn_map = shai.resize(attn_map, size, mode='bilinear')
                attn_map = shai.resize(attn_map.permute(2, 3, 0, 1), size, mode='bilinear')
                attn_map = attn_map.permute(2, 3, 0, 1).view(*self.attn_self.shape)
        
            self.attn_self = self.attn_self + attn_map
            self.count_self += 1
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4: hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection: hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def find_token_indices(self, prompt: str, tag: str, use_special_token=True) -> list:
        """Find token indices for a given tag in the tokenized prompt using multiple tokenizers.

        This function supports both CLIP and T5 text encoders.

        Args:
            prompt (str): The input text prompt.
            tag (str): The tag to search for.

        Returns:
            list: A list of token indices corresponding to the given tag.
        """
        self.tokenizer: CLIPTokenizer = self.pipe.tokenizer

        tokens = self.tokenizer.tokenize(prompt)
        tag_tokens = self.tokenizer.tokenize(tag)

        self.SOT = int(use_special_token) # CLIP skips a special token (e.g., [SOT]/[EOT])

        indices = [
            i + j + self.SOT
            for i in range(len(tokens) - len(tag_tokens) + 1) 
            if tokens[i:i + len(tag_tokens)] == tag_tokens
            for j in range(len(tag_tokens))
        ]
        return indices

    def generate(self, prompt: str, tags: List[str], steps: int = 50, cfg: float = 7.5, cfgpp: float = 0.0, generator=None) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        # Replace scheduler with DDIMScheduler for deterministic inversion
        if not isinstance(self.scheduler, DDIMScheduler) and cfgpp > 0:
            self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(self.pipe._pretrained_path, "scheduler")
        
        # Step 1: Encode text to conditional embeddings
        do_cfg = cfg > 0 or cfgpp > 0
        prompt_embeds = self.encode_text(prompt, do_cfg=do_cfg)

        # extract text token ids per tag
        tag2id = {}
        for tag in tags:
            tag2id[tag] = self.find_token_indices(prompt, tag)

        # Step 2: Prepare initial latents and scheduler
        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = torch.randn(
            (1, self.latent_dim, self.latent_size, self.latent_size),
            generator=generator, dtype=self.dtype, device=self.device
        ) * self.scheduler.init_noise_sigma
        
        accumulated_attn_self = []
        accumulated_attn_cross = []

        # Step 3: Iterative denoising loop
        for t in shjo.progress(timesteps):
            self.reset_hook_cache()

            # a. Build input for UNet
            params = self.get_denoising_params(latents, t, do_cfg, prompt_embeds)

            # b. Predict noise
            noise_pred = self.backbone(**params).sample

            # accumulate self-attention maps
            attn_self_t = (self.attn_self / self.count_self).float().cpu().detach() # [HW, H, W]
            attn_cross_t = (self.attn_cross / self.count_cross).float().cpu().detach() # [L, H, W]

            accumulated_attn_self.append(attn_self_t.clone())
            accumulated_attn_cross.append(attn_cross_t.clone())

            # c. Apply classifier-free guidance
            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + (cfgpp if cfgpp > 0 else cfg) * (noise_cond - noise_uncond)

            # d. Update latents
            if not isinstance(self.scheduler, DDIMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            else:
                a_t = self.scheduler.alphas_cumprod[t]
                prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                a_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod

                beta_t = 1 - a_t
                x0_pred = (latents - beta_t**0.5 * noise_pred) / a_t**0.5

                epsilon = noise_uncond if cfgpp > 0 else noise_pred
                latents = (a_prev**0.5) * x0_pred + (1 - a_prev)**0.5 * epsilon

        # Step 4: Generate attention visualization
        accumulated_attn_self = torch.stack(accumulated_attn_self).mean(dim=0) # [T, HW, H, W] to [HW, H, W]
        accumulated_attn_cross = torch.stack(accumulated_attn_cross).mean(dim=0) # [T, L, H, W] to [L, H, W]
        
        ah, aw = accumulated_attn_self.shape[-2:]
        bgr_self = shai.visualize_pca(
            accumulated_attn_self, # [HW, H, W]
            patch_size=(self.image_size // max(ah, aw)) # scale to original image size
        )

        cross_masks = []
        for tag in tags: # [C]
            cross_masks.append(accumulated_attn_cross[tag2id[tag]].mean(dim=0))
        cross_masks = torch.stack(cross_masks) # [C, H, W]

        vis_masks = []
        for vis in shai.normalize(cross_masks, dim=(1, 2)).cpu().detach().float().numpy():
            vis_masks.append(shjo.colorize(shjo.resize_mask(vis, (self.image_size, self.image_size))))
        bgr_cross = shjo.hstack(*vis_masks)

        # Step 5: Decode final latents into image
        return self.decode_image(latents)[0], bgr_self, bgr_cross
    
    # Heterogeneous CFG without Feynman–Kac
    def generate_hcg(
            self,
            global_prompt: str,
            local_prompts: List[str],
            local_boxes: List[Tuple[int, int, int, int]],  # (x0,y0,x1,y1) in pixel coords
            steps: int = 50,
            cfg: float = 7.5,            # standard CFG scale (use >0 for CFG)
            cfgpp: float = 0.0,          # CFG++ scale (use >0 to enable CFG++)
            hcg_strength: float = 1.0,   # patch guidance strength
            gamma: Optional[List[float]] = None,  # per-patch sign/weight; default = +1
            generator=None,
        ) -> Image.Image:
        """
        Text-to-image with global CFG / CFG++ + Heterogeneous (local) Guidance.
        - Global: CFG or CFG++ (choose by cfg / cfgpp)
        - Local: add-only (cond - uncond) with sign gamma[i] and strength hcg_strength
        """
        self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(self.pipe._pretrained_path, "scheduler")

        # ----- Embeddings (we need cond/uncond separately) -----
        emb_u_global = self.encode_text("", do_cfg=False)                 # uncond (global)
        emb_c_global = self.encode_text(global_prompt, do_cfg=False)      # cond   (global)
        emb_c_patches = [self.encode_text(p, do_cfg=False) for p in local_prompts]

        # ----- Timesteps & latents -----
        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # derive spatial sizes from pipe default
        latents = torch.randn(
            1, self.latent_dim, self.latent_size, self.latent_size,
            generator=generator, dtype=self.dtype, device=self.device
        ) * self.scheduler.init_noise_sigma

        def to_latent_box(box):
            """
            Convert a normalized [0,1] box to latent grid indices.

            Args
            ----
            box : Tuple[float, float, float, float]
                (x0, y0, x1, y1) with each in [0, 1], where x1 > x0 and y1 > y0.

            Returns
            -------
            (lx0, ly0, lx1, ly1) : Tuple[int, int, int, int]
                Latent-space coordinates.
                Right/bottom are EXCLUSIVE (Python slicing friendly).
                Guarantees at least 1px span in both directions.
            """
            import math

            # latent grid size (e.g., 128 for 1024px image with VAE down=8)
            lw = lh = int(self.latent_size)

            # clamp to [0,1]
            x0, y0, x1, y1 = [float(v) for v in box]

            # map to latent indices; use floor for starts, ceil for ends (exclusive)
            lx0 = int(math.floor(x0 * lw))
            ly0 = int(math.floor(y0 * lh))
            lx1 = int(math.ceil (x1 * lw))
            ly1 = int(math.ceil (y1 * lh))

            # clamp to [0, lw]/[0, lh] and ensure at least 1px span
            lx0 = max(0, min(lx0, lw - 1))
            ly0 = max(0, min(ly0, lh - 1))
            lx1 = max(lx0 + 1, min(lx1, lw))
            ly1 = max(ly0 + 1, min(ly1, lh))

            return lx0, ly0, lx1, ly1
        
        # ----- Denoising loop -----
        for t in shjo.progress(timesteps):
            # --- Heterogeneous CFG + Local Ratios (clean version) ------------------------
            # Math (log-domain, up to a constant C):
            #   log p_cfg ∝ (1-g)·log p_u + g·log p_B + Σ_i [ w_i · (log p(X_i|A_i) - log p(X_i)) ]
            # Score/ε-parameterization => linear mix:
            #   Global: (1-g)*ε_u + g*ε_B
            #   Local (per ROI): add w_i * (ε_c - ε_u) on the ROI only (zero-padded context)

            # (1) Scale input to scheduler's expected domain
            x_global = self.scheduler.scale_model_input(latents, t)

            # (2) GLOBAL: standard CFG (or CFG++) on the mean direction
            eps_u_global = self.backbone(sample=x_global, timestep=t, encoder_hidden_states=emb_u_global).sample
            eps_B        = self.backbone(sample=x_global, timestep=t, encoder_hidden_states=emb_c_global).sample

            # Choose CFG weight: g>0 → CFG (scale=g), else use cfgpp as your alternative switch
            g = cfg if cfg > 0.0 else cfgpp

            # Global composed epsilon (algebraically same as eps_u + g*(eps_B - eps_u))
            composed = (1.0 - g) * eps_u_global + g * eps_B

            # (3) LOCAL: ratio (cond - uncond) per ROI using zero-padding (no interpolation)
            # - No overlap handling (assumed none), no feathering, no attention masking.
            if hcg_strength > 0 and len(local_boxes) > 0:
                for i, box in enumerate(local_boxes):
                    lx0, ly0, lx1, ly1 = to_latent_box(box)
                    ph, pw = (ly1 - ly0), (lx1 - lx0)
                    assert ph > 0 and pw > 0, f"Invalid patch size {(ph, pw)} for box {box} at latent size {self.latent_size}"

                    # Zero-padded ROI forward: outside the ROI is exact zero (neutral VAE-latent context).
                    x_roi = torch.zeros_like(x_global)
                    x_roi[:, :, ly0:ly1, lx0:lx1] = x_global[:, :, ly0:ly1, lx0:lx1]

                    # Unconditional/conditional with identical zero context -> stable (cond - uncond) delta
                    eps_u_roi = self.backbone(sample=x_roi, timestep=t, encoder_hidden_states=emb_u_global).sample
                    eps_c_roi = self.backbone(sample=x_roi, timestep=t, encoder_hidden_states=emb_c_patches[i]).sample

                    # Local ratio-like delta strictly within ROI
                    delta_roi = (eps_c_roi - eps_u_roi)[:, :, ly0:ly1, lx0:lx1]

                    # Add local contribution (no area normalization since boxes do not overlap)
                    composed[:, :, ly0:ly1, lx0:lx1] += (hcg_strength * float(gamma[i])) * delta_roi
            # -----------------------------------------------------------------------------

            # (4) Single scheduler step
            if torch.is_tensor(t):
                t_idx = int(t.item()) if t.numel() == 1 else int(t[0].item())
            else:
                t_idx = int(t)

            a_t = self.scheduler.alphas_cumprod[t_idx]
            prev_t = t_idx - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            a_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
            beta_t = 1 - a_t

            # Standard DDIM update using composed epsilon as the mean direction
            x0_pred = (latents - beta_t.sqrt() * composed) / a_t.sqrt()

            # Variance source: use unconditional if cfgpp > 0.0 (CFG++), otherwise follow composed
            eps_for_noise = eps_u_global if cfgpp > 0.0 else composed

            latents = (a_prev.sqrt()) * x0_pred + (1 - a_prev).sqrt() * eps_for_noise
            latents = latents.to(self.dtype)

        # ----- Decode -----
        return self.decode_image(latents)[0]
    
    # Heterogeneous CFG with Feynman–Kac
    def generate_hfkc(
        self,
        global_prompt: str,
        local_prompts: List[str],
        local_boxes: List[Tuple[float, float, float, float]],  # [0,1] ratios
        steps: int = 50,
        cfg: float = 7.5,
        hcg_strength: float = 1.0,
        gamma: Optional[List[float]] = None,   # default -> [cfg]*M
        generator=None,
        N: int = 10,                            # batch size (Algorithm 1: N)
        ess_threshold: float = 0.5,             # Algorithm 1: τ
        resample: bool = True,                  # ESS resampling on/off
    ):
        """
        Heterogeneous CFG with Feynman–Kac (Algorithm 1)
        - Global: (1-g)*eps_u + g*eps_B
        - Local : add g*(eps_c - eps_u) on ROI (zero-padded)
        - Returns: best_image (top-1 by final weight)
        """

        # ---------- utils ----------
        def _tile_to_batch(ctx: torch.Tensor, N: int) -> torch.Tensor:
            """
            Make encoder_hidden_states batch match N by repeating the FIRST sample.
            Ex) [1, S, D] -> [N, S, D]; [K, S, D] -> [N, S, D] using ctx[:1].
            """
            assert ctx.ndim >= 2, f"Unexpected ctx shape: {tuple(ctx.shape)}"
            if ctx.shape[0] != N:
                ctx = ctx[:1].repeat(N, *([1] * (ctx.ndim - 1)))
            return ctx

        import math
        def to_latent_box(box):
            lw = lh = int(self.latent_size)
            x0, y0, x1, y1 = box
            lx0 = int(math.floor(x0 * lw)); ly0 = int(math.floor(y0 * lh))
            lx1 = int(math.ceil (x1 * lw)); ly1 = int(math.ceil (y1 * lh))
            lx0 = max(0, min(lx0, lw - 1)); ly0 = max(0, min(ly0, lh - 1))
            lx1 = max(lx0 + 1, min(lx1, lw)); ly1 = max(ly0 + 1, min(ly1, lh))
            return lx0, ly0, lx1, ly1

        # ---------- Algorithm 1: Require ----------
        device, dtype = self.device, self.dtype
        g = float(cfg)
        M = len(local_prompts)
        gamma = [g] * M if gamma is None else gamma

        # (Text) embeddings
        emb_u_global = self.encode_text("", do_cfg=False)                    # [1, S, D]
        emb_c_global = self.encode_text(global_prompt, do_cfg=False)         # [1, S, D]
        emb_c_patches = [self.encode_text(p, do_cfg=False) for p in local_prompts]  # 각 [1, S, D]

        # match text-condition batch to N
        emb_u_global = _tile_to_batch(emb_u_global, N)
        emb_c_global = _tile_to_batch(emb_c_global, N)
        emb_c_patches = [ _tile_to_batch(e, N) for e in emb_c_patches ]

        # scheduler & time grid
        self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(self.pipe._pretrained_path, "scheduler")
        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps
        dt = 1.0 / float(steps)                                              # [Alg 1: Line 1]

        # initial latents: N particles from one generator/seed
        latents = torch.randn(
            (N, self.latent_dim, self.latent_size, self.latent_size),
            generator=generator, dtype=dtype, device=device
        ) * self.scheduler.init_noise_sigma                                   # [Alg 1: X0]
        logw = torch.zeros(N, 1, device=device)                               # [Alg 1: w0 = 1]

        # ---------- Algorithm 1: Loop Line 2 ----------
        for t in shjo.progress(timesteps):
            # --- Line 3: mixture score (global+local eps) ---
            x_global_t = self.scheduler.scale_model_input(latents, t)
            eps_u = self.backbone(sample=x_global_t, timestep=t, encoder_hidden_states=emb_u_global).sample
            eps_cg = self.backbone(sample=x_global_t, timestep=t, encoder_hidden_states=emb_c_global).sample
            composed = (1.0 - g) * eps_u + g * eps_cg

            # local contributions (ROI zero-padding; embeddings already [N, S, D])
            if M > 0 and hcg_strength:
                for i, box in enumerate(local_boxes):
                    lx0, ly0, lx1, ly1 = to_latent_box(box)
                    x_roi = torch.zeros_like(x_global_t)
                    x_roi[:, :, ly0:ly1, lx0:lx1] = x_global_t[:, :, ly0:ly1, lx0:lx1]
                    eps_u_r = self.backbone(sample=x_roi, timestep=t, encoder_hidden_states=emb_u_global).sample
                    eps_c_r = self.backbone(sample=x_roi, timestep=t, encoder_hidden_states=emb_c_patches[i]).sample
                    delta = (eps_c_r - eps_u_r)[:, :, ly0:ly1, lx0:lx1]
                    composed[:, :, ly0:ly1, lx0:lx1] += hcg_strength * float(gamma[i]) * delta

            # --- Line 4–5: propagate (DDIM deterministic) ---
            t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
            a_t = self.scheduler.alphas_cumprod[t_idx]
            prev_t = t_idx - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            a_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
            beta_t = 1.0 - a_t
            sigma_t = (1.0 - a_t).sqrt() # for score conversion

            # cache x_t before update
            latents_t = latents

            # x0 estimates at x_t
            x0_star = (latents_t - beta_t.sqrt() * composed) / a_t.sqrt()
            x0_u    = (latents_t - beta_t.sqrt() * eps_u)   / a_t.sqrt()

            # standard DDIM noise source (CFG++ removed)
            use_cfgpp_noise = False # i.e., CFG everywhere
            eps_noise = eps_u if use_cfgpp_noise else composed

            mean_next_star = a_prev.sqrt() * x0_star + (1.0 - a_prev).sqrt() * eps_noise
            mean_next_u    = a_prev.sqrt() * x0_u    + (1.0 - a_prev).sqrt() * eps_u     # ref drift (uncond)

            # drifts must be computed against x_t
            v_star = (mean_next_star - latents_t) / dt
            v_hat  = (mean_next_u    - latents_t) / dt

            # take step
            latents = mean_next_star.to(dtype)

            # --- Line 6: update log-weights at x_t (pre-update state) ---
            delta_logw = torch.zeros(N, 1, device=device)
            if M > 0:
                for i, box in enumerate(local_boxes):
                    lx0, ly0, lx1, ly1 = to_latent_box(box)
                    x_roi_t = torch.zeros_like(x_global_t)
                    x_roi_t[:, :, ly0:ly1, lx0:lx1] = x_global_t[:, :, ly0:ly1, lx0:lx1]
                    eps_u_r = self.backbone(sample=x_roi_t, timestep=t, encoder_hidden_states=emb_u_global).sample
                    eps_c_r = self.backbone(sample=x_roi_t, timestep=t, encoder_hidden_states=emb_c_patches[i]).sample
                    s_i = -(eps_c_r - eps_u_r) / (sigma_t + 1e-8)
                    dv = (v_star - v_hat)[:, :, ly0:ly1, lx0:lx1].float()
                    si = s_i[:, :, ly0:ly1, lx0:lx1].float()
                    delta_logw += (dv * si).flatten(1).sum(dim=1, keepdim=True) * float(gamma[i]) * dt

            logw += delta_logw

            # --- Line 7: weight normalization ---
            w = torch.softmax(logw.squeeze(-1), dim=0)

            # --- Line 8–12: ESS check & (optional) resample ---
            if resample:
                ess = (w.sum() ** 2) / (w.pow(2).sum() + 1e-12)
                if ess.item() < ess_threshold * N:
                    idx = torch.multinomial(w, num_samples=N, replacement=True)
                    latents = latents[idx]
                    logw.zero_()
                    w.fill_(1.0 / N)

        # ---------- top-1 particle ----------
        top_idx = int(torch.argmax(w).item())
        best_img = self.decode_image(latents[top_idx:top_idx+1])[0]
        return best_img


"""
source ../250806_Text-to-Image_Diffusion_Attention/venv/bin/activate

# 9 seconds -> 
CUDA_VISIBLE_DEVICES=0 python3 demo_sd_HFKC.py --arch SD2.1 --global_prompt "A photo of a cat and a dog on a grass field, high quality, 4k" --tags cat dog --steps 50 --cfg 7.5 --cfgpp 0.0 --seed 0
CUDA_VISIBLE_DEVICES=0 python3 demo_sd_HFKC.py --arch SD2.1 --global_prompt "A photo of a cat and a dog on a grass field, high quality, 4k" --tags cat dog --steps 50 --cfg 7.5 --cfgpp 0.0 --seed 0 --hcg 7.5
CUDA_VISIBLE_DEVICES=0 python3 demo_sd_HFKC.py --arch SD2.1 --global_prompt "A photo of a cat and a dog on a grass field, high quality, 4k" --tags cat dog --steps 50 --cfg 7.5 --cfgpp 0.0 --seed 0 --hcg 7.5

CUDA_VISIBLE_DEVICES=2 python3 demo_sd_HFKC.py --arch SD2.1 --global_prompt "A photo of a cow, a horse, and a sheep on a grass field, high quality, 4k" --tags cow horse sheep --steps 50 --cfg 7.5 --cfgpp 0.0 --seed 0
CUDA_VISIBLE_DEVICES=0 python3 demo_sd_HFKC.py --arch SD2.1 --global_prompt "A photo of a cow, a horse, and a sheep on a grass field, high quality, 4k" --tags cow horse sheep --steps 50 --cfg 7.5 --cfgpp 0.0 --seed 0 --hcg 7.5 \
--local_prompts "A photo of a cow" "A photo of a horse" "A photo of a sheep" \
--local_boxes "(0.050, 0.550, 0.280, 0.900)" "(0.360, 0.520, 0.640, 0.900)" "(0.720, 0.540, 0.950, 0.900)"

CUDA_VISIBLE_DEVICES=2 python3 demo_sd_HFKC.py --arch SD2.1 --global_prompt "A photo of six cups on a wooden table, high quality, 4k" --tags cups table --steps 50 --cfg 7.5 --cfgpp 0.0 --seed 0
CUDA_VISIBLE_DEVICES=0 python3 demo_sd_HFKC.py --arch SD2.1 --global_prompt "A photo of six cups on a wooden table, high quality, 4k" --tags cups table --steps 50 --cfg 7.5 --cfgpp 0.0 --seed 0 --hcg 7.5 \
--local_prompts "A photo of a cup" "A photo of a cup" "A photo of a cup" "A photo of a cup" "A photo of a cup" "A photo of a cup" \
--local_boxes "(0.040, 0.050, 0.320, 0.407)" "(0.360, 0.050, 0.640, 0.407)" "(0.680, 0.050, 0.960, 0.407)" "(0.040, 0.593, 0.320, 0.950)" "(0.360, 0.593, 0.640, 0.950)" "(0.680, 0.593, 0.960, 0.950)"
"""
# 250913_Heterogeneous_Feynman-Kac_Corrector
if __name__ == '__main__':
    args = shjo.Parser(
        {
            'arch': 'SD2.1',

            'global_prompt': 'A photo of a cat and a dog on a grass field, high quality, 4k',
            'tags': ['cat', 'dog'],
            
            'seed': 0,
            'steps': 50,
            'cfg': 7.5, # classifier-free guidance scale (conditional epsilon)
            'cfgpp': 0.0, # classifier-free guidance scale (unconditional epsilon)

            'hcg': 0.0, # heterogeneous (local) guidance (add-only epsilon)
            'local_prompts': ["A photo of a dog", "A photo of a cat"],
            'local_boxes': [
                "(0.050781250, 0.142578125, 0.470703125, 0.857421875)",
                "(0.529296875, 0.142578125, 0.949218750, 0.857421875)"
            ],
            'N': 10, # number of particles
            'ess_threshold': 0.5, # effective sample size threshold (ratio)
            'resample': False,
        }
    )

    pipe = diffusion.build_pipeline(args.arch) # official pipeline provided by diffusers
    pipe = UnetPipeline(pipe) # customized pipeline
    
    with torch.no_grad():
        if args.hcg > 0:
            def parse_local_boxes(data: List[str]) -> List[Tuple[float, float, float, float]]:
                """Parse boxes given as strings or sequences into [0,1] ratio tuples."""
                import ast
                out = []
                for item in data:
                    # 1) to list[float] of length 4
                    if isinstance(item, str):
                        s = item.strip()
                        if s and s[0] not in "([{":
                            s = f"({s})"
                        try:
                            vals = ast.literal_eval(s)
                        except Exception:
                            vals = [float(v) for v in s.replace("(", "").replace(")", "").split(",")]
                    elif isinstance(item, (list, tuple)) and len(item) == 4:
                        vals = [float(v) for v in item]
                    else:
                        raise ValueError(f"Unsupported box: {item}")

                    if len(vals) != 4:
                        raise ValueError(f"Box must have 4 values: {item}")

                    # 2) clamp to [0,1] and ensure positive area
                    x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in vals]
                    eps = 1e-6
                    if x1 <= x0: x1 = min(1.0, x0 + eps)
                    if y1 <= y0: y1 = min(1.0, y0 + eps)

                    out.append((x0, y0, x1, y1))
                return out

            local_boxes = parse_local_boxes(args.local_boxes)
            generated_image = pipe.generate_hfkc(
                args.global_prompt, args.local_prompts, local_boxes,
                steps=args.steps, cfg=args.cfg, # cfgpp=args.cfgpp, 
                hcg_strength=args.hcg, gamma=[1.0] * len(local_boxes), 
                generator=shai.set_seed(args.seed), 
                
                N=args.N, ess_threshold=args.ess_threshold, resample=args.resample, # for HFKC
            )

            cfg_tag = f'CFG={args.cfg:.01f}' if args.cfg > 0 else f'CFGpp={args.cfgpp:.01f}'

            fig_dir = shjo.makedir(f'./figures/{args.arch}_{args.global_prompt}_{cfg_tag}_HFKC@N={args.N:02d}_ESS={args.ess_threshold:.02f}_Resample={args.resample}/')
            cv_image = shjo.pil2cv(generated_image)
            shjo.imwrite(fig_dir + 'text-to-image_hfkc.jpg', cv_image)
            
            for box in local_boxes:
                shjo.draw_rect(cv_image, [int(v * pipe.image_size) for v in box], dashed=True, thickness=4)
            shjo.imwrite(fig_dir + f'text-to-image_hfkc_box.jpg', cv_image)
        else:
            pipe.hook_attention(['enc', 'mid', 'dec'])

            generated_image, bgr_self, bgr_cross = pipe.generate(
                args.global_prompt, args.tags,
                steps=args.steps, cfg=args.cfg, cfgpp=args.cfgpp,
                generator=shai.set_seed(args.seed),
            )

            cfg_tag = f'CFG={args.cfg:.01f}' if args.cfg > 0 else f'CFGpp={args.cfgpp:.01f}'

            fig_dir = shjo.makedir(f'./figures/{args.arch}_{args.global_prompt}_{cfg_tag}/')
            shjo.imwrite(fig_dir + 'text-to-image.jpg', shjo.pil2cv(generated_image))
            shjo.imwrite(fig_dir + 'self.jpg', bgr_self)
            shjo.imwrite(fig_dir + 'cross.jpg', bgr_cross)

