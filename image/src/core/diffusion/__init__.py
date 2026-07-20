# Copyright (C) 2025 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

from typing import Any

import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    FluxFillPipeline,
    StableDiffusionXLPipeline,
)

model_dict = {
    # image diffusion
    "SD1.4": {  # CVPR'22 (Oral)
        "pretrained_model_or_path": "CompVis/stable-diffusion-v1-4",
        "torch_dtype": torch.float16,
    },
    "SD1.4+TokenCompose": {  # CVPR'24 / https://huggingface.co/mlpc-lab/TokenCompose_SD14_A
        "pretrained_model_or_path": "mlpc-lab/TokenCompose_SD14_A",
        "torch_dtype": torch.float32,
    },
    "SD1.5": {
        "pretrained_model_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "use_safetensors": True,
        "variant": "fp16",
        "torch_dtype": torch.float16,
    },
    "SD2.0": {
        "pretrained_model_or_path": "stabilityai/stable-diffusion-2-base",
        "use_auth_token": True,
        "torch_dtype": torch.float16,
    },
    "SD2.1": {  # CVPR'22 (Oral) & https://arxiv.org/abs/2112.10752
        "pretrained_model_or_path": "sd2-community/stable-diffusion-2-1-base",
        "use_safetensors": True,
        "variant": "fp16",
        "torch_dtype": torch.float16,
    },
    "SD2.1+TokenCompose": {  # CVPR'24 / https://huggingface.co/mlpc-lab/TokenCompose_SD14_A
        "pretrained_model_or_path": "mlpc-lab/TokenCompose_SD21_B",
        "use_auth_token": True,
        "torch_dtype": torch.float32,
    },
    "SDXL": {  # 2307
        "pretrained_model_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "use_safetensors": True,
        "torch_dtype": torch.float16,
        "variant": "fp16",
    },
    "SDXL+IterComp": {  # ICLR'25 (fine-tuning on SDXL)
        "pretrained_model_or_path": "comin/IterComp",
        "torch_dtype": torch.float16,
    },
    "PixArt-Alpha": {  # ICLR'24 Spotlight / https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS (steps: 20 & cfg: 4.5)
        "pretrained_model_or_path": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "torch_dtype": torch.float16,
    },
    "PixArt-Sigma": {  # https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS (steps: 20 & cfg: 4.5)
        "pretrained_model_or_path": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "torch_dtype": torch.float16,
    },
    "Playground-v2.5": {  # 2402
        "pretrained_model_or_path": "playgroundai/playground-v2.5-1024px-aesthetic",
        "torch_dtype": torch.float16,
        "variant": "fp16",
    },
    "SD3.0-M": {  # https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers (steps: 28 & cfg: 7.0)
        "pretrained_model_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
    },
    "FLUX.1-dev": {  # https://huggingface.co/black-forest-labs/FLUX.1-dev (steps: 50 & cfg: 3.5)
        "pretrained_model_or_path": "black-forest-labs/FLUX.1-dev",
        "torch_dtype": torch.bfloat16,
    },
    "SD3.5-M": {  # https://huggingface.co/stabilityai/stable-diffusion-3.5-medium (steps: 40 & cfg: 4.5)
        "pretrained_model_or_path": "stabilityai/stable-diffusion-3.5-medium",
        "torch_dtype": torch.bfloat16,
    },
    "SD3.5-L": {  # https://huggingface.co/stabilityai/stable-diffusion-3.5-large (steps: 28 & cfg: 3.5)
        "pretrained_model_or_path": "stabilityai/stable-diffusion-3.5-large",
        "torch_dtype": torch.bfloat16,
    },
    "Sana-4K": {  # https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers (steps: 20 & cfg: 5.0)
        "pretrained_model_or_path": "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers",
        "torch_dtype": torch.bfloat16,
    },
    "FLUX.1-Kontext-dev": {  # https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev (steps: 28 & cfg: 2.5)
        "pretrained_model_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
        "torch_dtype": torch.bfloat16,
    },
    "Qwen-Image": {  # https://huggingface.co/Qwen/Qwen-Image (steps: 50 & true_cfg: 4.0 & guidance: 1.0)
        "pretrained_model_name_or_path": "Qwen/Qwen-Image",
        "torch_dtype": torch.bfloat16,
    },
    # + consistency distillation
    "SD-Turbo": {
        "pretrained_model_or_path": "stabilityai/sd-turbo",
        "torch_dtype": torch.float16,
        "variant": "fp16",
    },
    "SDXL-Turbo": {
        "pretrained_model_or_path": "stabilityai/sdxl-turbo",
        "torch_dtype": torch.float16,
        "variant": "fp16",
    },
    "FLUX.1-schnell": {"pretrained_model_or_path": "black-forest-labs/FLUX.1-schnell", "torch_dtype": torch.bfloat16},
    "SD3.5-L-Turbo": {  # https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo (steps: 4 & cfg: 0.)
        "pretrained_model_or_path": "stabilityai/stable-diffusion-3.5-large-turbo",
        "torch_dtype": torch.bfloat16,
    },
}


def build_pipeline(arch, device=torch.device("cuda"), **kwargs) -> DiffusionPipeline:
    params: dict[str, Any] = model_dict[arch].copy()
    params.update(
        {
            "safety_checker": None,
            "requires_safety_checker": False,
        }
    )

    if "pag_applied_layers" in kwargs:
        params.update({"enable_pag": True, "pag_applied_layers": kwargs["pag_applied_layers"]})

    # Determine pipeline class
    if "Inpainting" in arch:
        pipeline_class = {
            "FLUX": FluxFillPipeline,
        }.get(arch[:4], AutoPipelineForInpainting)
    else:
        if arch == "Qwen-Image":  # if Qwen-Image import fails
            from diffusers import QwenImagePipeline

            pipeline_class = QwenImagePipeline
        else:
            pipeline_class = AutoPipelineForText2Image

    # Instantiate pipeline
    pipe = pipeline_class.from_pretrained(**params)

    # Apply VAE fix for SDXL black image issue
    if isinstance(pipe, StableDiffusionXLPipeline):
        pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Optional memory optimizations
    if kwargs.get("cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    if kwargs.get("xformers"):
        pipe.enable_xformers_memory_efficient_attention()

    # Ensure default sample size exists
    if not hasattr(pipe, "default_sample_size"):
        pipe.default_sample_size = 64

    # Store original pretrained path
    pipe._pretrained_path = params.get("pretrained_model_or_path", params.get("pretrained_model_name_or_path"))

    return pipe.to(device)
