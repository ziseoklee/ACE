"""Immutable response types for the v1 capabilities contract."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

UnavailableReason = Literal[
    "cuda_unavailable",
    "cuda_incompatible",
    "checkpoint_missing",
    "dependency_unavailable",
    "feature_not_enabled",
]
FeatureName = Literal["inference", "druglikeness", "scaffold_preservation", "docking"]


class AvailableFeature(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    available: Literal[True] = True
    reason: None = None


class UnavailableFeature(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    available: Literal[False] = False
    reason: UnavailableReason


FeatureAvailability = Annotated[AvailableFeature | UnavailableFeature, Field(discriminator="available")]


class OperationalLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_file_bytes: int = Field(default=10485760, ge=1, le=10485760)
    max_request_bytes: int = Field(default=33554432, ge=1, le=33554432)
    max_config_bytes: int = Field(default=16384, ge=1, le=16384)
    max_pending_jobs: int = Field(default=4, ge=0, le=4)
    max_running_jobs: int = Field(default=1, ge=1, le=1)
    max_num_samples: int = Field(default=16, ge=1, le=16)
    max_num_sampling_steps: int = Field(default=2000, ge=10, le=2000)
    max_num_ligand_atoms: int = Field(default=128, ge=1, le=128)
    max_pocket_atoms: int = Field(default=10000, ge=1, le=10000)
    job_timeout_seconds: int = Field(default=3600, ge=1, le=3600)

    @model_validator(mode="after")
    def validate_request_limits(self) -> Self:
        if max(self.max_file_bytes, self.max_config_bytes) > self.max_request_bytes:
            raise ValueError("File and config limits cannot exceed the whole request limit")
        return self


class EvaluationCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    druglikeness: FeatureAvailability
    scaffold_preservation: FeatureAvailability
    docking: FeatureAvailability


class Capabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    api_version: Literal["1.0.0"] = "1.0.0"
    inference: FeatureAvailability
    evaluation: EvaluationCapabilities
    inference_presets: tuple[Literal["ace_scaffold_v1"], ...] = ("ace_scaffold_v1",)
    limits: OperationalLimits
