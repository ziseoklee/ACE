"""Backward-compatible import for ACE's former HFKC development name."""

from .demo_sd_ace import StableDiffusionACEPipelineWrapper

StableDiffusionHFKCPipelineWrapper = StableDiffusionACEPipelineWrapper

__all__ = ["StableDiffusionHFKCPipelineWrapper"]
