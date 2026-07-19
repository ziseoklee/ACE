"""Method configurations for the six Stable Diffusion rows in Table E.10."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseMethodConfig:
    name: str = ""
    arch: str = ""
    steps: int = 50
    cfg: float = 7.5
    cfgpp: float = 0.0
    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    compile_forward: bool = False

    def set_efficiency_mode(self) -> None:
        self.compile_forward = False
        self.pipeline_kwargs["torch_dtype"] = "float32"
        self.pipeline_kwargs.pop("variant", None)


def _paper_sampler_kwargs(sampler: str, bump: float) -> dict[str, Any]:
    """Appendix E.4 settings shared by SD1.5 and SD2.1."""
    return {
        "sampler": sampler,
        "local_guidance": 7.5,
        "gamma": 1.0,
        "B": bump,
        "N": 3,
        "eta": 1.5,
        "resample": sampler != "nr",
        "resample_mode": "scheduled",
        "resample_at": [0.3],
    }


def _sd15_pipeline_kwargs() -> dict[str, Any]:
    return {
        "model_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "use_safetensors": True,
        "variant": "fp16",
        "torch_dtype": "float16",
    }


def _sd21_pipeline_kwargs() -> dict[str, Any]:
    return {
        "model_path": "stabilityai/stable-diffusion-2-1-base",
        "use_safetensors": True,
        "torch_dtype": "float16",
    }


@dataclass
class SD15NRConfig(BaseMethodConfig):
    name: str = "NR (SD1.5)"
    arch: str = "SD1.5+NR"
    generate_kwargs: dict[str, Any] = field(default_factory=lambda: _paper_sampler_kwargs("nr", 0.0))
    pipeline_kwargs: dict[str, Any] = field(default_factory=_sd15_pipeline_kwargs)


@dataclass
class SD15FKCConfig(BaseMethodConfig):
    name: str = "FKC (SD1.5)"
    arch: str = "SD1.5+FKC"
    generate_kwargs: dict[str, Any] = field(default_factory=lambda: _paper_sampler_kwargs("fkc", 0.0))
    pipeline_kwargs: dict[str, Any] = field(default_factory=_sd15_pipeline_kwargs)


@dataclass
class SD15ACEConfig(BaseMethodConfig):
    name: str = "ACE B5 (SD1.5)"
    arch: str = "SD1.5+ACE"
    generate_kwargs: dict[str, Any] = field(default_factory=lambda: _paper_sampler_kwargs("ace", 5.0))
    pipeline_kwargs: dict[str, Any] = field(default_factory=_sd15_pipeline_kwargs)


@dataclass
class SD21NRConfig(BaseMethodConfig):
    name: str = "NR (SD2.1)"
    arch: str = "SD2.1+NR"
    generate_kwargs: dict[str, Any] = field(default_factory=lambda: _paper_sampler_kwargs("nr", 0.0))
    pipeline_kwargs: dict[str, Any] = field(default_factory=_sd21_pipeline_kwargs)


@dataclass
class SD21FKCConfig(BaseMethodConfig):
    name: str = "FKC (SD2.1)"
    arch: str = "SD2.1+FKC"
    generate_kwargs: dict[str, Any] = field(default_factory=lambda: _paper_sampler_kwargs("fkc", 0.0))
    pipeline_kwargs: dict[str, Any] = field(default_factory=_sd21_pipeline_kwargs)


@dataclass
class SD21ACEConfig(BaseMethodConfig):
    name: str = "ACE B5 (SD2.1)"
    arch: str = "SD2.1+ACE"
    generate_kwargs: dict[str, Any] = field(default_factory=lambda: _paper_sampler_kwargs("ace", 5.0))
    pipeline_kwargs: dict[str, Any] = field(default_factory=_sd21_pipeline_kwargs)
