"""Evaluation requests, persisted input snapshots, and metric reports for API v1."""

from typing import Annotated, Generic, Literal, Self, TypeVar
from uuid import UUID

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from ace_backend.jobs_schema import Error, FrozenModel, InferenceResult, WarningMessage

MetricName = Literal["druglikeness", "scaffold_preservation", "docking"]


class UploadSource(FrozenModel):
    type: Literal["upload"]


class InferenceJobSource(FrozenModel):
    type: Literal["inference_job"]
    job_id: str
    sample_ids: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=1, max_length=16)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return str(UUID(value))

    @field_validator("sample_ids")
    @classmethod
    def unique_samples(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Sample IDs must be unique.")
        return value


EvaluationSource = Annotated[UploadSource | InferenceJobSource, Field(discriminator="type")]


class DockingConfig(FrozenModel):
    seed: int = Field(ge=0, le=2147483647)
    exhaustiveness: int = Field(ge=1, le=32)
    num_modes: int = Field(ge=1, le=20)
    padding_angstrom: float = Field(gt=0, le=20)


class EvaluationConfig(FrozenModel):
    source: EvaluationSource
    metrics: tuple[MetricName, ...] = Field(min_length=1, max_length=3)
    docking: DockingConfig | None

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Metrics must be unique.")
        if ("docking" in self.metrics) != (self.docking is not None):
            raise ValueError("Docking settings are required exactly when docking is requested.")
        return self


EVALUATION_CONFIG_EXAMPLE = {"source": {"type": "upload"}, "metrics": ["druglikeness"], "docking": None}


class EvaluationInput(FrozenModel):
    sample_id: int = Field(ge=0)
    ligand_sdf: str


class SourceArtifact(FrozenModel):
    artifact_id: str
    relative_path: str
    sha256: str


class EvaluationPreparation(FrozenModel):
    samples: tuple[EvaluationInput, ...]
    fragment_sdf: str | None = None
    pocket_pdb: str | None = None
    reference_ligand_sdf: str | None = None
    rdkit_version: str
    normalization: str = "Preserve originals and coordinates; sanitize copies; RDKit RemoveHs for topology. Persist normalized SDFs as V3000 with 17 decimal places; keep unsanitizable ligand bytes unchanged."
    source_artifacts: tuple[SourceArtifact, ...] = ()


T = TypeVar("T")


class SuccessfulReport(FrozenModel, Generic[T]):  # noqa: UP046 -- Python 3.11 is supported.
    status: Literal["succeeded"] = "succeeded"
    value: T


class FailedReport(FrozenModel):
    status: Literal["failed"] = "failed"
    error: Error


class SkippedReport(FrozenModel):
    status: Literal["skipped"] = "skipped"
    reason: Literal["invalid_molecule"] = "invalid_molecule"


BooleanReport = Annotated[SuccessfulReport[bool] | FailedReport | SkippedReport, Field(discriminator="status")]
NumberReport = Annotated[SuccessfulReport[float] | FailedReport | SkippedReport, Field(discriminator="status")]
UnitIntervalReport = Annotated[
    SuccessfulReport[Annotated[float, Field(ge=0, le=1)]] | FailedReport | SkippedReport, Field(discriminator="status")
]
LipinskiReport = Annotated[
    SuccessfulReport[Annotated[float, Field(ge=0.2, le=1)]] | FailedReport | SkippedReport,
    Field(discriminator="status"),
]


class DruglikenessReports(FrozenModel):
    validity: BooleanReport
    qed: UnitIntervalReport
    sa_normalized: UnitIntervalReport
    logp: NumberReport
    lipinski_legacy: LipinskiReport


class ScaffoldValue(FrozenModel):
    contains_fragment: bool
    method: Literal["rdkit_substructure_v1"] = "rdkit_substructure_v1"


class DockingValue(FrozenModel):
    affinity_kcal_mol: float
    num_poses: int = Field(ge=1)


ScaffoldReport = Annotated[
    SuccessfulReport[ScaffoldValue] | FailedReport | SkippedReport, Field(discriminator="status")
]
DockingReport = Annotated[SuccessfulReport[DockingValue] | FailedReport | SkippedReport, Field(discriminator="status")]


class MetricGroups(FrozenModel):
    druglikeness: DruglikenessReports | None = None
    scaffold_preservation: ScaffoldReport | None = None
    docking: DockingReport | None = None

    @model_serializer(mode="wrap")
    def requested_groups_only(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {name: value for name, value in handler(self).items() if value is not None}


class EvaluationSample(FrozenModel):
    sample_id: int = Field(ge=0)
    metrics: MetricGroups


class EvaluationResult(FrozenModel):
    job_id: str
    kind: Literal["evaluation"] = "evaluation"
    source: EvaluationSource
    metrics: tuple[MetricName, ...]
    samples: tuple[EvaluationSample, ...]
    warnings: tuple[WarningMessage, ...]


JobResult = Annotated[InferenceResult | EvaluationResult, Field(discriminator="kind")]
RESULT_ADAPTER: TypeAdapter[JobResult] = TypeAdapter(JobResult)
