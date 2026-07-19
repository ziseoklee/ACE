"""Pipeline construction for the ACE paper's SD1.5/SD2.1 experiments."""

from typing import Any, cast

import torch
from diffusers import StableDiffusionPipeline

from baselines import StableDiffusionACEPipelineWrapper
from configs.config_method import BaseMethodConfig


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported torch dtype: {dtype_name!r}.") from error


def create_pipeline(method_config: BaseMethodConfig, device: torch.device) -> StableDiffusionACEPipelineWrapper:
    supported_methods = {
        "SD1.5+NR",
        "SD1.5+FKC",
        "SD1.5+ACE",
        "SD2.1+NR",
        "SD2.1+FKC",
        "SD2.1+ACE",
    }
    if method_config.arch not in supported_methods:
        raise ValueError(f"The ACE benchmark driver does not support {method_config.arch!r}.")

    pipeline_kwargs: dict[str, Any] = {
        "torch_dtype": get_torch_dtype(method_config.pipeline_kwargs.get("torch_dtype", "float16")),
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    for optional_key in ("use_safetensors", "variant"):
        if optional_key in method_config.pipeline_kwargs:
            pipeline_kwargs[optional_key] = method_config.pipeline_kwargs[optional_key]

    base_pipeline = StableDiffusionPipeline.from_pretrained(
        method_config.pipeline_kwargs["model_path"],
        **pipeline_kwargs,
    ).to(device)
    if not hasattr(base_pipeline, "default_sample_size"):
        base_pipeline.default_sample_size = 64
    return StableDiffusionACEPipelineWrapper(cast(StableDiffusionPipeline, base_pipeline))
